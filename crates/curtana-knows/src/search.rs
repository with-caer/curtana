use std::collections::HashMap;

use crate::Error;

/// (taxonomy, artifact_id, score)
pub(crate) type ScoredTaxonomyId = (String, String, f32);

/// Half-life in days for the recency exponential decay.
/// After 14 days, a recency signal of 1.0 decays to 0.5.
pub(crate) const RECENCY_HALF_LIFE_DAYS: f64 = 14.0;

/// Reciprocal Rank Fusion constant. Higher values dampen the contribution
/// of lower-ranked results.
const RRF_K: f32 = 60.0;

// ---------------------------------------------------------------------------
// Vector scoring
// ---------------------------------------------------------------------------

/// Scores artifacts across multiple taxonomies by cosine similarity with
/// optional recency blending. Returns `(taxonomy, id, score)` tuples sorted
/// descending.
pub(crate) fn vector_score_multi(
    conn: &duckdb::Connection,
    taxonomies: &[String],
    query_literal: &str,
    recency_weight: f32,
    now: u64,
    limit: usize,
) -> Result<Vec<ScoredTaxonomyId>, Error> {
    let rw = recency_weight.clamp(0.0, 1.0);

    let placeholders: String = taxonomies.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = if rw > 0.0 {
        let decay = f64::ln(2.0) / RECENCY_HALF_LIFE_DAYS;
        format!(
            "SELECT e.taxonomy, e.artifact_id, \
                (1.0 - {rw}) * MAX(list_cosine_similarity(e.embedding, {query_literal})) \
                + {rw} * exp(-(CAST(? AS DOUBLE) - COALESCE(a.timestamp, 0)) / 86400.0 * {decay}) \
                AS score \
             FROM artifact_embeddings e \
             JOIN artifacts a ON a.taxonomy = e.taxonomy AND a.id = e.artifact_id \
             WHERE e.taxonomy IN ({placeholders}) \
             GROUP BY e.taxonomy, e.artifact_id, a.timestamp \
             HAVING score > 0.0 \
             ORDER BY score DESC \
             LIMIT ?",
        )
    } else {
        format!(
            "SELECT e.taxonomy, e.artifact_id, \
                MAX(list_cosine_similarity(e.embedding, {query_literal})) AS score \
             FROM artifact_embeddings e \
             WHERE e.taxonomy IN ({placeholders}) \
             GROUP BY e.taxonomy, e.artifact_id \
             HAVING score > 0.0 \
             ORDER BY score DESC \
             LIMIT ?",
        )
    };

    let mut statement = conn
        .prepare(&sql)
        .map_err(|e| Error::DatabaseError(e.to_string()))?;

    let mut params_vec: Vec<Box<dyn duckdb::types::ToSql>> = Vec::new();
    if rw > 0.0 {
        params_vec.push(Box::new(now as i64));
    }
    for t in taxonomies {
        params_vec.push(Box::new(t.clone()));
    }
    params_vec.push(Box::new(limit as i64));
    let param_refs: Vec<&dyn duckdb::types::ToSql> =
        params_vec.iter().map(|b| b.as_ref()).collect();

    let rows = statement
        .query_map(param_refs.as_slice(), |row| {
            let taxonomy: String = row.get(0)?;
            let id: String = row.get(1)?;
            let score: f64 = row.get(2)?;
            Ok((taxonomy, id, score as f32))
        })
        .map_err(|e| Error::DatabaseError(e.to_string()))?;

    Ok(rows.flatten().collect())
}

// ---------------------------------------------------------------------------
// BM25 scoring
// ---------------------------------------------------------------------------

/// Scores artifacts across multiple taxonomies using BM25.
/// Returns empty if the FTS index hasn't been built yet.
pub(crate) fn bm25_score_multi(
    conn: &duckdb::Connection,
    taxonomies: &[String],
    query_text: &str,
    limit: usize,
) -> Result<Vec<ScoredTaxonomyId>, Error> {
    if !has_fts_index(conn) {
        return Ok(vec![]);
    }

    let escaped = query_text.replace('\'', "''");
    let placeholders: String = taxonomies.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT a.taxonomy, a.id, fts_main_artifacts.match_bm25(a.fts_key, '{escaped}') AS score \
         FROM artifacts a \
         WHERE score IS NOT NULL AND a.taxonomy IN ({placeholders}) \
         ORDER BY score DESC \
         LIMIT ?"
    );

    let mut statement = conn
        .prepare(&sql)
        .map_err(|e| Error::DatabaseError(e.to_string()))?;

    let mut params_vec: Vec<Box<dyn duckdb::types::ToSql>> = Vec::new();
    for t in taxonomies {
        params_vec.push(Box::new(t.clone()));
    }
    params_vec.push(Box::new(limit as i64));
    let param_refs: Vec<&dyn duckdb::types::ToSql> =
        params_vec.iter().map(|b| b.as_ref()).collect();

    let rows = statement
        .query_map(param_refs.as_slice(), |row| {
            let taxonomy: String = row.get(0)?;
            let id: String = row.get(1)?;
            let score: f64 = row.get(2)?;
            Ok((taxonomy, id, score as f32))
        })
        .map_err(|e| Error::DatabaseError(e.to_string()))?;

    Ok(rows.flatten().collect())
}

/// Returns true if the FTS index schema exists.
fn has_fts_index(conn: &duckdb::Connection) -> bool {
    conn.query_row(
        "SELECT COUNT(*) FROM information_schema.schemata \
         WHERE schema_name = 'fts_main_artifacts'",
        [],
        |row| row.get::<_, i64>(0),
    )
    .unwrap_or(0)
        > 0
}

// ---------------------------------------------------------------------------
// Reciprocal Rank Fusion
// ---------------------------------------------------------------------------

/// Fuses multi-taxonomy vector and BM25 results using RRF.
/// Keys on `(taxonomy, id)` to avoid collisions across taxonomies.
pub(crate) fn rrf_fuse_multi(
    vector_results: &[ScoredTaxonomyId],
    bm25_results: &[ScoredTaxonomyId],
) -> Vec<ScoredTaxonomyId> {
    let mut scores: HashMap<(String, String), f32> = HashMap::new();
    for (rank, (taxonomy, id, _)) in vector_results.iter().enumerate() {
        *scores.entry((taxonomy.clone(), id.clone())).or_default() +=
            1.0 / (RRF_K + rank as f32 + 1.0);
    }
    for (rank, (taxonomy, id, _)) in bm25_results.iter().enumerate() {
        *scores.entry((taxonomy.clone(), id.clone())).or_default() +=
            1.0 / (RRF_K + rank as f32 + 1.0);
    }
    let mut fused: Vec<ScoredTaxonomyId> = scores
        .into_iter()
        .map(|((taxonomy, id), score)| (taxonomy, id, score))
        .collect();
    fused.sort_by(|a, b| a.2.total_cmp(&b.2).reverse());
    fused
}

// ---------------------------------------------------------------------------
// FTS index management
// ---------------------------------------------------------------------------

/// Rebuilds the DuckDB FTS index on the `artifacts` table.
/// Uses `overwrite=1` so this is idempotent.
pub(crate) fn rebuild_fts_index(conn: &duckdb::Connection) -> Result<(), Error> {
    conn.execute("INSTALL 'fts'", []).ok(); // no-op if bundled/already installed
    conn.execute("LOAD 'fts'", [])
        .map_err(|e| Error::DatabaseError(e.to_string()))?;
    conn.execute(
        "PRAGMA create_fts_index('artifacts', 'fts_key', 'contents', \
         stemmer='porter', overwrite=1)",
        [],
    )
    .map_err(|e| Error::DatabaseError(e.to_string()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_fuse_multi_keys_on_taxonomy_and_id() {
        // Same id in different taxonomies should not collide.
        let vector = vec![("tax1".to_string(), "a".to_string(), 0.9)];
        let bm25 = vec![("tax2".to_string(), "a".to_string(), 5.0)];
        let result = rrf_fuse_multi(&vector, &bm25);
        assert_eq!(result.len(), 2);
        let taxonomies: Vec<&str> = result.iter().map(|(t, _, _)| t.as_str()).collect();
        assert!(taxonomies.contains(&"tax1"));
        assert!(taxonomies.contains(&"tax2"));
    }

    #[test]
    fn rrf_fuse_multi_overlap_ranks_highest() {
        let vector = vec![
            ("t".to_string(), "a".to_string(), 0.9),
            ("t".to_string(), "shared".to_string(), 0.8),
        ];
        let bm25 = vec![
            ("t".to_string(), "b".to_string(), 5.0),
            ("t".to_string(), "shared".to_string(), 3.0),
        ];
        let result = rrf_fuse_multi(&vector, &bm25);
        assert_eq!(result[0].1, "shared");
    }
}
