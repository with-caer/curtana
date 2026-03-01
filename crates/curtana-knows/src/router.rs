use std::collections::HashSet;
use std::path::PathBuf;

use curtana_infers::{ChatModel, TextEmbeddingModel};

use tracing::debug;

use crate::{Artifact, best_chunk_score, manifest::Manifest, open_taxonomy_store};

/// Trade-off between relevance and diversity in MMR selection.
/// 1.0 = pure relevance (equivalent to top-k), 0.0 = pure diversity.
const MMR_LAMBDA: f32 = 0.7;

/// Soft score bonus for candidates that would introduce a taxonomy
/// not yet represented in the selected set.
const TAXONOMY_DIVERSITY_BONUS: f32 = 0.1;

/// Minimum cosine similarity to be considered as an MMR candidate.
/// Artifacts below this threshold are discarded before selection.
const MIN_SIMILARITY: f32 = 0.0;

/// Maximum number of taxonomies to search after embedding-based routing.
const MAX_ROUTED_TAXONOMIES: usize = 5;

/// An artifact annotated with its source taxonomy and similarity score.
pub struct ScoredArtifact {
    pub taxonomy: String,
    pub score: f32,
    pub artifact: Artifact,
}

/// Routes queries to the relevant taxonomy stores and merges results.
pub struct Router {
    manifest: Manifest,
    data_dir: PathBuf,
}

impl Router {
    pub fn new(manifest: Manifest, data_dir: PathBuf) -> Self {
        Self { manifest, data_dir }
    }

    /// Routes a query to the most relevant taxonomies by comparing the
    /// query embedding against each taxonomy's description embedding.
    /// Returns up to [`MAX_ROUTED_TAXONOMIES`] names, or all taxonomies
    /// if there are fewer than that.
    pub fn route(
        &self,
        embed_model: &mut TextEmbeddingModel,
        query_embedding: &[f32],
    ) -> Vec<String> {
        let entries: Vec<(&String, &str)> = self
            .manifest
            .taxonomies
            .iter()
            .filter(|(_, entry)| !entry.description.is_empty())
            .map(|(name, entry)| (name, entry.description.as_str()))
            .collect();

        if entries.len() <= MAX_ROUTED_TAXONOMIES {
            debug!(
                "routed to all {} taxonomies (below threshold)",
                entries.len()
            );
            return self.manifest.taxonomies.keys().cloned().collect();
        }

        let descriptions: Vec<&str> = entries.iter().map(|(_, desc)| *desc).collect();
        let desc_embeddings = embed_model.embed(&descriptions).unwrap();

        let mut scored: Vec<(&String, f32)> = entries
            .iter()
            .zip(desc_embeddings.iter())
            .map(|((name, _), emb)| (*name, curtana_infers::cosine_distance(emb, query_embedding)))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1).reverse());
        scored.truncate(MAX_ROUTED_TAXONOMIES);

        let names: Vec<String> = scored.iter().map(|(name, _)| (*name).clone()).collect();
        debug!(
            "routed to: {}",
            scored
                .iter()
                .map(|(name, score)| format!("{name} ({score:.4})"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        names
    }

    /// Routes the query to the most relevant taxonomies, then searches
    /// and reranks using MMR.
    pub async fn search(
        &self,
        embed_model: &mut TextEmbeddingModel,
        query: &str,
        top_k: usize,
    ) -> Vec<ScoredArtifact> {
        let query_embedding = embed_model.embed(&[query]).unwrap().pop().unwrap();
        let taxonomy_names = self.route(embed_model, &query_embedding);

        let mut all_scored = Vec::new();

        for name in &taxonomy_names {
            let store = open_taxonomy_store(&self.data_dir, name).await;
            let artifacts = store.all_embedded().await;

            for artifact in artifacts {
                let score = best_chunk_score(&artifact.embedding, &query_embedding);
                all_scored.push(ScoredArtifact {
                    taxonomy: name.clone(),
                    score,
                    artifact,
                });
            }
        }

        all_scored.retain(|s| s.score > MIN_SIMILARITY);
        mmr_select(all_scored, top_k, MMR_LAMBDA, taxonomy_names.len())
    }
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
/// While fewer than `taxonomy_count` distinct taxonomies are represented,
/// candidates from an unrepresented taxonomy receive a small score boost
/// to encourage breadth across data sources.
fn mmr_select(
    mut candidates: Vec<ScoredArtifact>,
    top_k: usize,
    lambda: f32,
    taxonomy_count: usize,
) -> Vec<ScoredArtifact> {
    if candidates.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let mut selected: Vec<ScoredArtifact> = Vec::with_capacity(top_k);
    let mut selected_taxonomies: HashSet<String> = HashSet::new();

    // Seed with the single most relevant candidate.
    let best_idx = candidates
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.score.total_cmp(&b.score))
        .map(|(i, _)| i)
        .unwrap();
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

            // Boost candidates that would introduce an unrepresented taxonomy.
            if selected_taxonomies.len() < taxonomy_count
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
    chat_model: &mut ChatModel,
    sample_size: usize,
) -> String {
    let samples = store.sample(sample_size).await;

    if samples.is_empty() {
        return String::new();
    }

    let sample_texts: Vec<String> = samples
        .iter()
        .map(|artifact| {
            let s = format!("{}", artifact.contents);
            if s.len() > 500 {
                s[..500].to_string()
            } else {
                s
            }
        })
        .collect();

    if sample_texts.is_empty() {
        return String::new();
    }

    let samples_block = sample_texts
        .iter()
        .enumerate()
        .map(|(i, text)| format!("--- Sample {} ---\n{text}", i + 1))
        .collect::<Vec<_>>()
        .join("\n\n");

    let prompt = format!(
        "Below are sample artifacts from a collection. \
         Write a single sentence describing the kind of content this collection holds. \
         Be specific about topics and domains.\n\n{samples_block}"
    );

    let mut output = Vec::new();
    chat_model.infer(&prompt, &mut output).unwrap();
    String::from_utf8_lossy(&output).trim().to_string()
}
