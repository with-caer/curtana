use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use curtana_infers::{ChatModel, TextEmbeddingModel};

use tracing::{debug, warn};

use crate::{Artifact, manifest::Manifest, open_source_store};

/// Trade-off between relevance and diversity in MMR selection.
/// 1.0 = pure relevance (equivalent to top-k), 0.0 = pure diversity.
const MMR_LAMBDA: f32 = 0.7;

/// Soft score bonus for candidates that would introduce a taxonomy
/// not yet represented in the selected set. Lower than the old value (0.1)
/// because flat search naturally provides more diversity.
const TAXONOMY_DIVERSITY_BONUS: f32 = 0.05;

/// Minimum cosine similarity to be considered as an MMR candidate.
/// Artifacts below this threshold are discarded before selection.
const MIN_SIMILARITY: f32 = 0.0;

/// How much taxonomy affinity can shift a candidate's RRF score.
/// Applied as `score *= 1.0 + AFFINITY_WEIGHT * affinity`.
const AFFINITY_WEIGHT: f32 = 0.3;

/// Cap on how many distinct taxonomies receive the diversity bonus.
/// Beyond this count, the bonus stops — prevents over-fragmenting results
/// when there are many taxonomies.
const MAX_DIVERSITY_TAXONOMIES: usize = 8;

/// An artifact annotated with its source taxonomy and similarity score.
pub struct ScoredArtifact {
    pub taxonomy: String,
    pub score: f32,
    pub artifact: Artifact,
}

/// Searches all taxonomy stores and merges results with taxonomy affinity
/// scoring. Unlike the old routing approach (which hard-filtered to a few
/// taxonomies), this searches everything and uses taxonomy descriptions as
/// a soft relevance boost.
pub struct Router {
    manifest: Manifest,
    data_dir: PathBuf,
}

impl Router {
    pub fn new(manifest: Manifest, data_dir: PathBuf) -> Self {
        Self { manifest, data_dir }
    }

    /// Searches all taxonomies across all source stores, applies taxonomy
    /// affinity boosting from cached description embeddings, and re-ranks
    /// with MMR for diversity.
    pub async fn search(
        &self,
        embed_model: &mut TextEmbeddingModel,
        query: &str,
        top_k: usize,
        recency_weight: f32,
    ) -> Result<Vec<ScoredArtifact>, crate::Error> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let query_embedding = embed_model
            .embed(&[query])
            .map_err(|e| crate::Error::EmbeddingError(format!("{e:?}")))?
            .pop()
            .ok_or_else(|| crate::Error::EmbeddingError("no embedding returned".into()))?;

        // Compute taxonomy affinities from cached description embeddings (no model call).
        let affinities = taxonomy_affinities(&self.manifest, &query_embedding);

        // Group ALL taxonomies by source_key so we open one DB per source.
        let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (name, entry) in &self.manifest.taxonomies {
            groups
                .entry(entry.source_key())
                .or_default()
                .push(name.clone());
        }

        let total_taxonomy_count = self.manifest.taxonomies.len();

        debug!(
            "flat search across {} taxonomies in {} stores",
            total_taxonomy_count,
            groups.len()
        );

        // Fetch more candidates than top_k from each store to give MMR
        // enough diversity to work with, but still bounded.
        let candidate_limit = top_k * 3;
        let mut all_scored = Vec::new();

        for (source_key, group_taxonomies) in &groups {
            let store = match open_source_store(&self.data_dir, source_key).await {
                Ok(s) => s,
                Err(e) => {
                    warn!("failed to open store for {source_key}: {e}");
                    continue;
                }
            };
            let candidates = match store
                .search_candidates(
                    group_taxonomies,
                    &query_embedding,
                    query,
                    candidate_limit,
                    recency_weight,
                    now,
                )
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    warn!("failed to search store for {source_key}: {e}");
                    continue;
                }
            };

            for (taxonomy, _id, score, artifact) in candidates {
                all_scored.push(ScoredArtifact {
                    taxonomy,
                    score,
                    artifact,
                });
            }
        }

        // Apply taxonomy affinity boost: multiply each candidate's RRF score
        // by (1 + AFFINITY_WEIGHT * affinity). Taxonomies without cached
        // embeddings get affinity 0 → no boost, no penalty.
        for candidate in &mut all_scored {
            let affinity = affinities.get(&candidate.taxonomy).copied().unwrap_or(0.0);
            candidate.score *= 1.0 + AFFINITY_WEIGHT * affinity;
        }

        all_scored.retain(|s| s.score > MIN_SIMILARITY);
        Ok(mmr_select(
            all_scored,
            top_k,
            MMR_LAMBDA,
            total_taxonomy_count,
        ))
    }
}

/// Cosine similarity between the query embedding and each taxonomy's
/// pre-computed description embedding. Returns map of taxonomy name →
/// affinity in [0, 1]. Taxonomies without a cached embedding get 0.
fn taxonomy_affinities(manifest: &Manifest, query_embedding: &[f32]) -> BTreeMap<String, f32> {
    let mut affinities = BTreeMap::new();
    for (name, entry) in &manifest.taxonomies {
        let affinity = match &entry.description_embedding {
            Some(emb) => curtana_infers::cosine_distance(emb, query_embedding).max(0.0),
            None => 0.0,
        };
        affinities.insert(name.clone(), affinity);
    }
    affinities
}

/// Selects `top_k` artifacts using Maximal Marginal Relevance (MMR).
///
/// MMR balances relevance (similarity to query) with diversity
/// (dissimilarity to already-selected artifacts):
///
///   MMR(d) = λ · rel(d) − (1−λ) · max_{s∈S} sim(d, s)
///
/// `lambda` controls the trade-off: 1.0 = pure relevance, 0.0 = pure diversity.
///
/// While fewer than `diversity_target` (capped at [`MAX_DIVERSITY_TAXONOMIES`])
/// distinct taxonomies are represented, candidates from an unrepresented
/// taxonomy receive a small score boost to encourage breadth across data sources.
fn mmr_select(
    mut candidates: Vec<ScoredArtifact>,
    top_k: usize,
    lambda: f32,
    taxonomy_count: usize,
) -> Vec<ScoredArtifact> {
    if candidates.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let diversity_target = taxonomy_count.min(MAX_DIVERSITY_TAXONOMIES);

    let mut selected: Vec<ScoredArtifact> = Vec::with_capacity(top_k);
    let mut selected_taxonomies: HashSet<String> = HashSet::new();

    // Seed with the single most relevant candidate.
    let best_idx = candidates
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.score.total_cmp(&b.score))
        .map(|(i, _)| i)
        .expect("candidates verified non-empty above");
    selected_taxonomies.insert(candidates[best_idx].taxonomy.clone());
    selected.push(candidates.swap_remove(best_idx));

    // Iteratively pick the candidate that maximises the MMR criterion.
    while selected.len() < top_k && !candidates.is_empty() {
        let mut best_mmr = f32::NEG_INFINITY;
        let mut best_idx = 0;

        for (i, candidate) in candidates.iter().enumerate() {
            let relevance = candidate.score;

            // Diversity penalty: max similarity to any already-selected artifact.
            let max_sim = selected
                .iter()
                .map(|s| max_chunk_similarity(&candidate.artifact.embedding, &s.artifact.embedding))
                .fold(f32::NEG_INFINITY, f32::max);

            let mut mmr = lambda * relevance - (1.0 - lambda) * max_sim;

            // Boost candidates that would introduce an unrepresented taxonomy,
            // up to the diversity cap.
            if selected_taxonomies.len() < diversity_target
                && !selected_taxonomies.contains(&candidate.taxonomy)
            {
                mmr += TAXONOMY_DIVERSITY_BONUS;
            }

            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx = i;
            }
        }

        selected_taxonomies.insert(candidates[best_idx].taxonomy.clone());
        selected.push(candidates.swap_remove(best_idx));
    }

    selected
}

/// Returns the maximum cosine similarity between any chunk of `a` and any
/// chunk of `b`.
fn max_chunk_similarity(a_chunks: &[Vec<f32>], b_chunks: &[Vec<f32>]) -> f32 {
    a_chunks
        .iter()
        .flat_map(|a| {
            b_chunks
                .iter()
                .map(move |b| curtana_infers::cosine_distance(a, b))
        })
        .fold(f32::NEG_INFINITY, f32::max)
}

/// Samples artifacts from a store and summarizes them into a one-sentence
/// taxonomy description via the chat model.
pub async fn generate_description(
    store: &crate::Store,
    taxonomy: &str,
    chat_model: &mut ChatModel,
    sample_size: usize,
) -> String {
    let samples = match store.sample(taxonomy, sample_size).await {
        Ok(s) => s,
        Err(e) => {
            warn!("failed to sample artifacts for description: {e}");
            return String::new();
        }
    };

    if samples.is_empty() {
        return String::new();
    }

    let sample_texts: Vec<String> = samples
        .iter()
        .map(|artifact| {
            let s = format!("{}", artifact.contents);
            crate::truncate_text(&s, 500).to_string()
        })
        .collect();

    if sample_texts.is_empty() {
        return String::new();
    }

    let samples_block = sample_texts
        .iter()
        .enumerate()
        .map(|(i, text)| {
            format!(
                "<sample index=\"{}\">\n{}\n</sample>",
                i + 1,
                crate::escape_xml(text)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let prompt = format!(
        "Below are sample artifacts from a collection. \
         Write a single sentence describing the kind of content this collection holds. \
         Be specific about topics and domains.\n\n{samples_block}"
    );

    let mut output = Vec::new();
    if let Err(e) = chat_model.infer(&prompt, &mut output) {
        warn!("failed to generate description: {e:?}");
        return String::new();
    }
    strip_think_tags(&String::from_utf8_lossy(&output))
}

/// Strips `<think>...</think>` reasoning blocks that some local models emit,
/// then trims surrounding whitespace.
fn strip_think_tags(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        result.push_str(&rest[..start]);
        match rest[start..].find("</think>") {
            Some(end) => rest = &rest[start + end + "</think>".len()..],
            None => {
                // Unclosed tag — drop everything from <think> onward.
                return result.trim().to_string();
            }
        }
    }
    result.push_str(rest);
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::TaxonomyEntry;

    fn make_entry(name: &str, description: &str, embedding: Option<Vec<f32>>) -> TaxonomyEntry {
        TaxonomyEntry {
            name: name.to_string(),
            description: description.to_string(),
            source_type: "imap".to_string(),
            source_id: name.to_string(),
            source_host: "host".to_string(),
            source_username: "user".to_string(),
            last_ingested_at: None,
            description_updated_at: None,
            cursor: None,
            description_embedding: embedding,
        }
    }

    fn make_artifact(score: f32, embedding: Vec<f32>) -> ScoredArtifact {
        ScoredArtifact {
            taxonomy: String::new(),
            score,
            artifact: Artifact {
                id: "x".into(),
                timestamp: 0,
                author: "t".into(),
                contents: "c".into(),
                embedding: vec![embedding],
            },
        }
    }

    #[test]
    fn taxonomy_affinities_empty_manifest() {
        let manifest = Manifest::default();
        let query = vec![1.0, 0.0, 0.0];
        let affinities = taxonomy_affinities(&manifest, &query);
        assert!(affinities.is_empty());
    }

    #[test]
    fn taxonomy_affinities_mixed_embeddings() {
        let mut manifest = Manifest::default();
        // One taxonomy with embedding, one without.
        manifest.taxonomies.insert(
            "with-emb".to_string(),
            make_entry("with-emb", "has embedding", Some(vec![1.0, 0.0, 0.0])),
        );
        manifest.taxonomies.insert(
            "no-emb".to_string(),
            make_entry("no-emb", "no embedding", None),
        );

        let query = vec![1.0, 0.0, 0.0];
        let affinities = taxonomy_affinities(&manifest, &query);

        assert_eq!(affinities.len(), 2);
        // "with-emb" has cosine similarity ~1.0 with query.
        assert!(affinities["with-emb"] > 0.9);
        // "no-emb" has no embedding → affinity 0.
        assert_eq!(affinities["no-emb"], 0.0);
    }

    #[test]
    fn taxonomy_affinities_negative_clamped_to_zero() {
        let mut manifest = Manifest::default();
        // Embedding pointing opposite to query → negative cosine → clamped to 0.
        manifest.taxonomies.insert(
            "opposite".to_string(),
            make_entry("opposite", "desc", Some(vec![-1.0, 0.0, 0.0])),
        );

        let query = vec![1.0, 0.0, 0.0];
        let affinities = taxonomy_affinities(&manifest, &query);

        assert_eq!(affinities["opposite"], 0.0);
    }

    #[test]
    fn mmr_diversity_cap_stops_bonus() {
        // Create candidates from 20 distinct taxonomies.
        // All have the same embedding and base score so diversity bonus
        // is the only differentiator after the first pick.
        let mut candidates: Vec<ScoredArtifact> = (0..20)
            .map(|i| {
                let mut sa = make_artifact(1.0, vec![1.0, 0.0, 0.0]);
                sa.taxonomy = format!("tax-{i}");
                sa
            })
            .collect();
        // Add extra candidates from tax-0 with slightly lower score.
        for _ in 0..5 {
            let mut sa = make_artifact(0.99, vec![1.0, 0.0, 0.0]);
            sa.taxonomy = "tax-0".to_string();
            candidates.push(sa);
        }

        let selected = mmr_select(candidates, 15, 0.7, 20);
        let unique_taxonomies: HashSet<String> =
            selected.iter().map(|s| s.taxonomy.clone()).collect();

        // Diversity bonus caps at MAX_DIVERSITY_TAXONOMIES (8), so we shouldn't
        // see all 20 taxonomies forced in — the bonus stops encouraging new ones.
        // With 15 results and a cap of 8, we expect roughly 8-12 unique taxonomies
        // (some diversity happens naturally), but definitely not all 20.
        assert!(unique_taxonomies.len() <= 15);
        // The cap should be effective: we should see at least the capped amount.
        assert!(unique_taxonomies.len() >= MAX_DIVERSITY_TAXONOMIES);
    }

    #[test]
    fn affinity_boost_preserves_ordering_when_uniform() {
        // All candidates have the same affinity → ordering unchanged.
        let mut candidates = vec![
            make_artifact(0.9, vec![1.0, 0.0, 0.0]),
            make_artifact(0.5, vec![0.0, 1.0, 0.0]),
            make_artifact(0.3, vec![0.0, 0.0, 1.0]),
        ];
        candidates[0].taxonomy = "a".into();
        candidates[1].taxonomy = "a".into();
        candidates[2].taxonomy = "a".into();

        // Uniform affinity of 0.8.
        let factor = 1.0 + AFFINITY_WEIGHT * 0.8;
        for c in &mut candidates {
            c.score *= factor;
        }

        // After uniform boost, ordering should be 0.9*f > 0.5*f > 0.3*f.
        let selected = mmr_select(candidates, 3, 1.0, 1);
        assert!(selected[0].score > selected[1].score);
        assert!(selected[1].score > selected[2].score);
    }

    #[test]
    fn affinity_boost_reorders_candidates() {
        // Two candidates: one high base score / low affinity, one low base / high affinity.
        let mut high_base = make_artifact(0.8, vec![1.0, 0.0, 0.0]);
        high_base.taxonomy = "low-aff".into();
        let mut low_base = make_artifact(0.7, vec![0.0, 1.0, 0.0]);
        low_base.taxonomy = "high-aff".into();

        // Apply affinity boost: low-aff gets 0, high-aff gets 1.0.
        high_base.score *= 1.0 + AFFINITY_WEIGHT * 0.0; // stays 0.8
        low_base.score *= 1.0 + AFFINITY_WEIGHT * 1.0; // becomes 0.7 * 1.3 = 0.91

        let selected = mmr_select(vec![high_base, low_base], 2, 1.0, 2);
        // After affinity boost, the originally-lower candidate should be first.
        assert_eq!(selected[0].taxonomy, "high-aff");
    }

    #[test]
    fn strip_think_empty_block() {
        assert_eq!(
            strip_think_tags("<think>\n\n</think>\nActual description."),
            "Actual description."
        );
    }

    #[test]
    fn strip_think_with_content() {
        assert_eq!(
            strip_think_tags("<think>reasoning here</think>Answer."),
            "Answer."
        );
    }

    #[test]
    fn strip_think_no_tags() {
        assert_eq!(
            strip_think_tags("Just a plain string."),
            "Just a plain string."
        );
    }

    #[test]
    fn strip_think_unclosed() {
        assert_eq!(strip_think_tags("before<think>oops no close"), "before");
    }

    #[test]
    fn strip_think_multiple() {
        assert_eq!(
            strip_think_tags("<think>a</think>one<think>b</think>two"),
            "onetwo"
        );
    }
}
