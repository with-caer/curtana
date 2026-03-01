extern crate alloc;

pub mod manifest;
pub mod router;
pub mod tools;

use std::{fmt, path::Path, sync::Arc};

use codas::{
    codec::{Decodable, ReadsDecodable, WritesEncodable},
    types::Text,
};
use curtana_infers::TextEmbeddingModel;
use duckdb::params;
use tokio::sync::Mutex;
use tracing::{info, trace, warn};

codas_macros::export_coda!("crates/curtana-knows/src/coda.md");

/// Sort order for browsing artifacts.
pub enum BrowseOrder {
    Asc,
    Desc,
}

impl BrowseOrder {
    fn sql(&self) -> &str {
        match self {
            BrowseOrder::Asc => "ASC",
            BrowseOrder::Desc => "DESC",
        }
    }
}

/// Datastore for knowledge graph storage and retrieval.
#[derive(Clone)]
pub struct Store {
    connection: Arc<Mutex<duckdb::Connection>>,
}

impl Store {
    pub async fn new(path: Option<&str>) -> Result<Self, Error> {
        let connection = match path {
            Some(path) => {
                duckdb::Connection::open(path).map_err(|e| Error::DatabaseError(e.to_string()))?
            }
            None => duckdb::Connection::open_in_memory()
                .map_err(|e| Error::DatabaseError(e.to_string()))?,
        };

        let this = Self {
            connection: Arc::new(Mutex::new(connection)),
        };

        let this_copy = this.clone();
        tokio::task::spawn_blocking(move || -> Result<(), Error> {
            let connection = this_copy.connection.blocking_lock();

            connection
                .execute(
                    "CREATE TABLE IF NOT EXISTS artifacts (id VARCHAR PRIMARY KEY, data BLOB);",
                    [],
                )
                .map_err(|e| Error::DatabaseError(e.to_string()))?;

            // Backwards-compatible schema migration: add structured columns
            // for SQL-level filtering. These are denormalized from the blob.
            let _ = connection.execute(
                "ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS timestamp BIGINT",
                [],
            );
            let _ = connection.execute(
                "ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS author VARCHAR",
                [],
            );

            // Migration: add embedding column for DB-level similarity search.
            let _ = connection.execute(
                "ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS embedding BLOB",
                [],
            );

            Ok(())
        })
        .await
        .map_err(|e| Error::SpawnError(e.to_string()))??;

        Ok(this)
    }

    pub async fn upsert(&self, artifact: &Artifact) -> Result<(), Error> {
        let this = self.clone();

        let id = format!("{}", artifact.id);
        let timestamp = artifact.timestamp as i64;
        let author = format!("{}", artifact.author);

        let mut data_bytes = vec![];
        data_bytes
            .write_data(artifact)
            .map_err(|e| Error::SerializationError(format!("{e:?}")))?;

        // Serialize embedding chunks as a flat blob: [n_chunks: u32][dim: u32][f32 data...]
        let embedding_blob = serialize_embedding(&artifact.embedding);

        tokio::task::spawn_blocking(move || -> Result<(), Error> {
            let connection = this.connection.blocking_lock();

            connection
                .execute(
                    "INSERT OR REPLACE INTO artifacts (id, data, timestamp, author, embedding) VALUES (?, ?, ?, ?, ?)",
                    params![id.as_str(), data_bytes, timestamp, author.as_str(), embedding_blob],
                )
                .map_err(|e| Error::DatabaseError(e.to_string()))?;

            Ok(())
        })
        .await
        .map_err(|e| Error::SpawnError(e.to_string()))?
    }

    pub async fn get(&self, id: Text, data: &mut impl Decodable) -> Result<(), Error> {
        let this = self.clone();

        let data_bytes: Vec<u8> = tokio::task::spawn_blocking(move || {
            let connection = this.connection.blocking_lock();
            connection
                .query_row(
                    "SELECT data FROM artifacts WHERE id=(?)",
                    params![id.as_str()],
                    |row| row.get(0),
                )
                .ok()
        })
        .await
        .map_err(|e| Error::SpawnError(e.to_string()))?
        .ok_or(Error::NotFound)?;

        data_bytes
            .as_slice()
            .read_data_into(data)
            .map_err(|e| Error::SerializationError(format!("{e:?}")))?;

        Ok(())
    }

    /// Embeds all artifacts currently stored in the datastore
    /// that don't already have embeddings. If the text exceeds
    /// the model's context window, it is split into chunks and
    /// each chunk is embedded separately.
    pub async fn embed_pending(
        &self,
        model: &mut TextEmbeddingModel,
        mut on_progress: impl FnMut(usize, usize),
    ) -> Result<(), Error> {
        let this = self.clone();

        // Find all unembedded artifacts.
        let unembedded: Vec<(String, Artifact)> = tokio::task::spawn_blocking(move || {
            let connection = this.connection.blocking_lock();

            let mut statement = match connection.prepare(
                "SELECT id, data FROM artifacts WHERE embedding IS NULL OR LENGTH(embedding) = 0",
            ) {
                Ok(s) => s,
                Err(e) => {
                    warn!("failed to prepare embed_pending query: {e}");
                    return vec![];
                }
            };
            let rows = match statement.query_map([], |row| {
                let id: String = row.get(0)?;
                let data: Vec<u8> = row.get(1)?;
                Ok((id, data))
            }) {
                Ok(r) => r,
                Err(e) => {
                    warn!("failed to query artifacts: {e}");
                    return vec![];
                }
            };

            rows.into_iter()
                .filter_map(|row| {
                    let (id, data) = row.ok()?;
                    let artifact: Artifact = match data.as_slice().read_data() {
                        Ok(a) => a,
                        Err(_) => return None,
                    };
                    Some((id, artifact))
                })
                .collect::<Vec<_>>()
        })
        .await
        .map_err(|e| Error::SpawnError(e.to_string()))?;

        let total = unembedded.len();
        info!("embedding {} artifacts", total);
        on_progress(0, total);

        for (i, (_artifact_id, mut artifact)) in unembedded.into_iter().enumerate() {
            let text = format!("{}", artifact.contents);
            artifact.embedding = embed_with_chunking(model, &text, 0);
            self.upsert(&artifact).await?;
            on_progress(i + 1, total);

            if i % 50 == 0 {
                info!("embedded {i} artifacts...");
            }
        }

        Ok(())
    }

    /// Returns all artifacts that have embeddings.
    pub async fn all_embedded(&self) -> Result<Vec<Artifact>, Error> {
        let this = self.clone();

        tokio::task::spawn_blocking(move || -> Result<Vec<Artifact>, Error> {
            let connection = this.connection.blocking_lock();

            let mut statement = connection
                .prepare(
                    "SELECT data FROM artifacts WHERE embedding IS NOT NULL AND LENGTH(embedding) > 0",
                )
                .map_err(|e| Error::DatabaseError(e.to_string()))?;
            let rows = statement
                .query_map([], |row| {
                    let data: Vec<u8> = row.get(0)?;
                    Ok(data)
                })
                .map_err(|e| Error::DatabaseError(e.to_string()))?;

            Ok(rows
                .into_iter()
                .filter_map(|r| {
                    let data = r.ok()?;
                    data.as_slice().read_data().ok()
                })
                .collect())
        })
        .await
        .map_err(|e| Error::SpawnError(e.to_string()))?
    }

    /// Returns a random sample of up to `limit` artifacts from the store.
    pub async fn sample(&self, limit: usize) -> Result<Vec<Artifact>, Error> {
        let this = self.clone();

        tokio::task::spawn_blocking(move || -> Result<Vec<Artifact>, Error> {
            let connection = this.connection.blocking_lock();

            let mut statement = connection
                .prepare("SELECT data FROM artifacts ORDER BY RANDOM() LIMIT ?")
                .map_err(|e| Error::DatabaseError(e.to_string()))?;
            let rows = statement
                .query_map(params![limit], |row| {
                    let data: Vec<u8> = row.get(0)?;
                    Ok(data)
                })
                .map_err(|e| Error::DatabaseError(e.to_string()))?;

            Ok(rows
                .into_iter()
                .filter_map(|r| {
                    let data = r.ok()?;
                    data.as_slice().read_data().ok()
                })
                .collect())
        })
        .await
        .map_err(|e| Error::SpawnError(e.to_string()))?
    }

    /// Finds the `top_k` artifacts most similar to `query` in the datastore.
    ///
    /// Uses a two-phase approach to avoid loading all artifact data into memory:
    /// 1. Load only (id, embedding_blob) pairs — compact
    /// 2. Compute similarity in Rust, pick top-k IDs
    /// 3. Load full artifact data only for the top-k
    pub async fn search(
        &self,
        model: &mut TextEmbeddingModel,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<Artifact>, Error> {
        // Embed the query.
        let query_embedding = model
            .embed(&[query])
            .map_err(|e| Error::EmbeddingError(format!("{e:?}")))?
            .pop()
            .ok_or_else(|| Error::EmbeddingError("no embedding returned".into()))?;

        // Phase 1: Load only (id, embedding_blob) — much smaller than full data.
        let this = self.clone();
        let scored_ids: Vec<(String, f32)> = tokio::task::spawn_blocking(move || -> Result<Vec<(String, f32)>, Error> {
            let connection = this.connection.blocking_lock();

            let mut statement = connection
                .prepare("SELECT id, embedding FROM artifacts WHERE embedding IS NOT NULL AND LENGTH(embedding) > 0")
                .map_err(|e| Error::DatabaseError(e.to_string()))?;
            let rows = statement
                .query_map([], |row| {
                    let id: String = row.get(0)?;
                    let emb_blob: Vec<u8> = row.get(1)?;
                    Ok((id, emb_blob))
                })
                .map_err(|e| Error::DatabaseError(e.to_string()))?;

            let mut scored = Vec::new();
            for (id, emb_blob) in rows.flatten() {
                let chunks = deserialize_embedding(&emb_blob);
                if !chunks.is_empty() {
                    let score = best_chunk_score(&chunks, &query_embedding);
                    scored.push((id, score));
                }
            }

            // Sort and truncate to top-k.
            scored.sort_by(|a, b| a.1.total_cmp(&b.1).reverse());
            scored.truncate(top_k);
            Ok(scored)
        })
        .await
        .map_err(|e| Error::SpawnError(e.to_string()))??;

        if scored_ids.is_empty() {
            return Ok(vec![]);
        }

        // Phase 2: Load full artifact data only for top-k IDs.
        let ids: Vec<String> = scored_ids.iter().map(|(id, _)| id.clone()).collect();
        let this = self.clone();
        let mut artifacts: Vec<Artifact> =
            tokio::task::spawn_blocking(move || -> Result<Vec<Artifact>, Error> {
                let connection = this.connection.blocking_lock();

                let placeholders: String = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                let sql = format!("SELECT data FROM artifacts WHERE id IN ({placeholders})");
                let mut statement = connection
                    .prepare(&sql)
                    .map_err(|e| Error::DatabaseError(e.to_string()))?;

                let param_refs: Vec<&dyn duckdb::types::ToSql> =
                    ids.iter().map(|s| s as &dyn duckdb::types::ToSql).collect();
                let rows = statement
                    .query_map(param_refs.as_slice(), |row| {
                        let data: Vec<u8> = row.get(0)?;
                        Ok(data)
                    })
                    .map_err(|e| Error::DatabaseError(e.to_string()))?;

                Ok(rows
                    .into_iter()
                    .filter_map(|r| {
                        let data = r.ok()?;
                        data.as_slice().read_data().ok()
                    })
                    .collect())
            })
            .await
            .map_err(|e| Error::SpawnError(e.to_string()))??;

        // Re-sort by score (DB may return in different order).
        let id_to_score: std::collections::HashMap<String, f32> = scored_ids.into_iter().collect();
        artifacts.sort_by(|a, b| {
            let sa = id_to_score
                .get(format!("{}", a.id).as_str())
                .copied()
                .unwrap_or(0.0);
            let sb = id_to_score
                .get(format!("{}", b.id).as_str())
                .copied()
                .unwrap_or(0.0);
            sa.total_cmp(&sb).reverse()
        });

        Ok(artifacts)
    }

    /// Returns scored artifacts for MMR re-ranking. Loads only embedding blobs
    /// from the DB, scores them, returns the top candidates with their embeddings
    /// still attached (needed for MMR inter-similarity computation).
    pub async fn search_candidates(
        &self,
        query_embedding: &[f32],
        candidate_limit: usize,
    ) -> Result<Vec<(String, f32, Artifact)>, Error> {
        let this = self.clone();
        let query_emb = query_embedding.to_vec();

        // Phase 1: Score all embeddings, keep top candidates.
        let scored_ids: Vec<(String, f32)> = tokio::task::spawn_blocking(move || -> Result<Vec<(String, f32)>, Error> {
            let connection = this.connection.blocking_lock();

            let mut statement = connection
                .prepare("SELECT id, embedding FROM artifacts WHERE embedding IS NOT NULL AND LENGTH(embedding) > 0")
                .map_err(|e| Error::DatabaseError(e.to_string()))?;
            let rows = statement
                .query_map([], |row| {
                    let id: String = row.get(0)?;
                    let emb_blob: Vec<u8> = row.get(1)?;
                    Ok((id, emb_blob))
                })
                .map_err(|e| Error::DatabaseError(e.to_string()))?;

            let mut scored = Vec::new();
            for (id, emb_blob) in rows.flatten() {
                let chunks = deserialize_embedding(&emb_blob);
                if !chunks.is_empty() {
                    let score = best_chunk_score(&chunks, &query_emb);
                    scored.push((id, score));
                }
            }
            scored.sort_by(|a, b| a.1.total_cmp(&b.1).reverse());
            scored.truncate(candidate_limit);
            Ok(scored)
        })
        .await
        .map_err(|e| Error::SpawnError(e.to_string()))??;

        if scored_ids.is_empty() {
            return Ok(vec![]);
        }

        // Phase 2: Load full artifact data for candidates.
        let ids: Vec<String> = scored_ids.iter().map(|(id, _)| id.clone()).collect();
        let this = self.clone();
        let artifacts: Vec<Artifact> =
            tokio::task::spawn_blocking(move || -> Result<Vec<Artifact>, Error> {
                let connection = this.connection.blocking_lock();

                let placeholders: String = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                let sql = format!("SELECT data FROM artifacts WHERE id IN ({placeholders})");
                let mut statement = connection
                    .prepare(&sql)
                    .map_err(|e| Error::DatabaseError(e.to_string()))?;

                let param_refs: Vec<&dyn duckdb::types::ToSql> =
                    ids.iter().map(|s| s as &dyn duckdb::types::ToSql).collect();
                let rows = statement
                    .query_map(param_refs.as_slice(), |row| {
                        let data: Vec<u8> = row.get(0)?;
                        Ok(data)
                    })
                    .map_err(|e| Error::DatabaseError(e.to_string()))?;

                Ok(rows
                    .into_iter()
                    .filter_map(|r| {
                        let data = r.ok()?;
                        data.as_slice().read_data().ok()
                    })
                    .collect())
            })
            .await
            .map_err(|e| Error::SpawnError(e.to_string()))??;

        // Build result with scores.
        let id_to_score: std::collections::HashMap<String, f32> = scored_ids.into_iter().collect();
        let mut result: Vec<(String, f32, Artifact)> = artifacts
            .into_iter()
            .map(|a| {
                let id_str = format!("{}", a.id);
                let score = id_to_score.get(&id_str).copied().unwrap_or(0.0);
                (id_str, score, a)
            })
            .collect();
        result.sort_by(|a, b| a.1.total_cmp(&b.1).reverse());
        Ok(result)
    }

    /// Returns the total number of artifacts in the store.
    pub async fn count(&self) -> Result<usize, Error> {
        let this = self.clone();

        tokio::task::spawn_blocking(move || -> Result<usize, Error> {
            let connection = this.connection.blocking_lock();
            let count = connection
                .query_row("SELECT COUNT(*) FROM artifacts", [], |row| {
                    row.get::<_, i64>(0)
                })
                .map_err(|e| Error::DatabaseError(e.to_string()))?;
            Ok(count as usize)
        })
        .await
        .map_err(|e| Error::SpawnError(e.to_string()))?
    }

    /// Browses artifacts ordered by timestamp.
    pub async fn browse(
        &self,
        offset: usize,
        limit: usize,
        order: BrowseOrder,
    ) -> Result<Vec<Artifact>, Error> {
        let this = self.clone();
        let order_sql = order.sql().to_string();

        tokio::task::spawn_blocking(move || -> Result<Vec<Artifact>, Error> {
            let connection = this.connection.blocking_lock();

            let sql = format!(
                "SELECT data FROM artifacts ORDER BY COALESCE(timestamp, 0) {} LIMIT ? OFFSET ?",
                order_sql,
            );
            let mut statement = connection
                .prepare(&sql)
                .map_err(|e| Error::DatabaseError(e.to_string()))?;
            let rows = statement
                .query_map(params![limit as i64, offset as i64], |row| {
                    let data: Vec<u8> = row.get(0)?;
                    Ok(data)
                })
                .map_err(|e| Error::DatabaseError(e.to_string()))?;

            Ok(rows
                .into_iter()
                .filter_map(|r| {
                    let data = r.ok()?;
                    data.as_slice().read_data().ok()
                })
                .collect())
        })
        .await
        .map_err(|e| Error::SpawnError(e.to_string()))?
    }

    /// Filters artifacts by author and/or time range.
    pub async fn filter(
        &self,
        author: Option<String>,
        after: Option<u64>,
        before: Option<u64>,
        limit: usize,
    ) -> Result<Vec<Artifact>, Error> {
        let this = self.clone();

        tokio::task::spawn_blocking(move || -> Result<Vec<Artifact>, Error> {
            let connection = this.connection.blocking_lock();

            let has_author = author.is_some();
            let author_val = author.unwrap_or_default();
            let has_after = after.is_some();
            let after_val = after.map(|v| v as i64).unwrap_or(0);
            let has_before = before.is_some();
            let before_val = before.map(|v| v as i64).unwrap_or(0);

            let mut statement = connection
                .prepare(
                    "SELECT data FROM artifacts \
                     WHERE (NOT ? OR author = ?) \
                     AND (NOT ? OR timestamp >= ?) \
                     AND (NOT ? OR timestamp <= ?) \
                     ORDER BY COALESCE(timestamp, 0) DESC LIMIT ?",
                )
                .map_err(|e| Error::DatabaseError(e.to_string()))?;

            let rows = statement
                .query_map(
                    params![
                        has_author,
                        author_val.as_str(),
                        has_after,
                        after_val,
                        has_before,
                        before_val,
                        limit as i64,
                    ],
                    |row| {
                        let data: Vec<u8> = row.get(0)?;
                        Ok(data)
                    },
                )
                .map_err(|e| Error::DatabaseError(e.to_string()))?;

            Ok(rows
                .into_iter()
                .filter_map(|r| {
                    let data = r.ok()?;
                    data.as_slice().read_data().ok()
                })
                .collect())
        })
        .await
        .map_err(|e| Error::SpawnError(e.to_string()))?
    }
}

/// Maximum recursion depth for embedding chunking.
const MAX_CHUNK_DEPTH: usize = 8;

/// Embeds `text`, chunking only if the model's context window is exceeded.
/// Returns one embedding per chunk.
fn embed_with_chunking(model: &mut TextEmbeddingModel, text: &str, depth: usize) -> Vec<Vec<f32>> {
    if depth >= MAX_CHUNK_DEPTH {
        warn!("chunking depth limit ({MAX_CHUNK_DEPTH}) reached, returning unchunked");
        return vec![];
    }

    match model.embed(&[text]) {
        Ok(embeddings) => embeddings,
        Err(curtana_infers::Error::ContextSize { maximum, actual })
        | Err(curtana_infers::Error::MicrobatchSize { maximum, actual }) => {
            // Split text into chunks proportional to the overrun.
            // Use character-based splitting as a rough proxy for tokens.
            let num_chunks = actual.div_ceil(maximum);
            let chunk_size = text.len().div_ceil(num_chunks);
            let chunks: Vec<&str> = char_chunks(text, chunk_size);

            trace!(
                "text too long ({actual} tokens, max {maximum}), splitting into {} chunks",
                chunks.len()
            );

            chunks
                .iter()
                .flat_map(|chunk| {
                    if chunk.is_empty() {
                        return vec![];
                    }
                    // Recurse: character-based splitting is approximate,
                    // so some chunks may still exceed the token limit.
                    embed_with_chunking(model, chunk, depth + 1)
                })
                .collect()
        }
        Err(e) => {
            warn!("embedding failed: {e:?}");
            vec![]
        }
    }
}

/// Serializes chunk embeddings to a flat blob format.
fn serialize_embedding(chunks: &[Vec<f32>]) -> Vec<u8> {
    if chunks.is_empty() {
        return vec![];
    }
    let n_chunks = chunks.len() as u32;
    let dim = chunks.first().map(|c| c.len() as u32).unwrap_or(0);
    let mut blob = Vec::with_capacity(8 + (n_chunks as usize) * (dim as usize) * 4);
    blob.extend_from_slice(&n_chunks.to_le_bytes());
    blob.extend_from_slice(&dim.to_le_bytes());
    for chunk in chunks {
        for &val in chunk {
            blob.extend_from_slice(&val.to_le_bytes());
        }
    }
    blob
}

/// Deserializes chunk embeddings from the flat blob format.
fn deserialize_embedding(blob: &[u8]) -> Vec<Vec<f32>> {
    if blob.len() < 8 {
        return vec![];
    }
    let n_chunks = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]) as usize;
    let dim = u32::from_le_bytes([blob[4], blob[5], blob[6], blob[7]]) as usize;

    let Some(expected_len) = n_chunks
        .checked_mul(dim)
        .and_then(|x| x.checked_mul(4))
        .and_then(|x| x.checked_add(8))
    else {
        return vec![];
    };
    if blob.len() < expected_len {
        return vec![];
    }

    let mut chunks = Vec::with_capacity(n_chunks);
    let mut offset = 8;
    for _ in 0..n_chunks {
        let mut vec = Vec::with_capacity(dim);
        for _ in 0..dim {
            let bytes = [
                blob[offset],
                blob[offset + 1],
                blob[offset + 2],
                blob[offset + 3],
            ];
            vec.push(f32::from_le_bytes(bytes));
            offset += 4;
        }
        chunks.push(vec);
    }
    chunks
}

/// Truncates `text` to at most `max_bytes` bytes, breaking on a UTF-8
/// character boundary. Returns the full string if it already fits.
pub fn truncate_text(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        text
    } else {
        &text[..text.floor_char_boundary(max_bytes)]
    }
}

/// Splits `text` into chunks of approximately `byte_budget` bytes,
/// always breaking on a UTF-8 character boundary.
fn char_chunks(text: &str, byte_budget: usize) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let end = (start + byte_budget).min(text.len());
        // Walk back to a char boundary.
        let end = text.floor_char_boundary(end);
        if end <= start {
            break;
        }
        chunks.push(&text[start..end]);
        start = end;
    }
    chunks
}

/// Returns the highest cosine similarity between any chunk embedding
/// and the query embedding.
pub fn best_chunk_score(chunk_embeddings: &[Vec<f32>], query: &[f32]) -> f32 {
    chunk_embeddings
        .iter()
        .map(|chunk| curtana_infers::cosine_distance(chunk, query))
        .fold(f32::NEG_INFINITY, f32::max)
}

/// Opens a taxonomy-specific store at `{data_dir}/{taxonomy_name}.duckdb`.
///
/// `taxonomy_name` must be non-empty and consist only of alphanumeric
/// characters, hyphens, or underscores to prevent path traversal.
pub async fn open_taxonomy_store(data_dir: &Path, taxonomy_name: &str) -> Result<Store, Error> {
    if taxonomy_name.is_empty()
        || !taxonomy_name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        return Err(Error::DatabaseError(format!(
            "invalid taxonomy name: {taxonomy_name:?}"
        )));
    }
    let path = data_dir.join(format!("{taxonomy_name}.duckdb"));
    let path_str = path
        .to_str()
        .ok_or_else(|| Error::DatabaseError("non-UTF8 path".into()))?;
    Store::new(Some(path_str)).await
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Error {
    NotFound,
    DatabaseError(String),
    SerializationError(String),
    EmbeddingError(String),
    SpawnError(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotFound => write!(f, "not found"),
            Error::DatabaseError(e) => write!(f, "database error: {e}"),
            Error::SerializationError(e) => write!(f, "serialization error: {e}"),
            Error::EmbeddingError(e) => write!(f, "embedding error: {e}"),
            Error::SpawnError(e) => write!(f, "task spawn error: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn truncate_text_respects_char_boundaries() {
        // ASCII: no boundary issues.
        assert_eq!(truncate_text("hello", 3), "hel");
        assert_eq!(truncate_text("hello", 10), "hello");
        assert_eq!(truncate_text("hello", 5), "hello");

        // Multi-byte: 'é' is 2 bytes, so cutting at byte 1 must round down.
        let s = "é"; // 2 bytes
        assert_eq!(truncate_text(s, 1), "");
        assert_eq!(truncate_text(s, 2), "é");

        // 3-byte character (em-dash '—' = E2 80 94).
        let s = "a—b"; // 1 + 3 + 1 = 5 bytes
        assert_eq!(truncate_text(s, 2), "a"); // can't fit partial '—'
        assert_eq!(truncate_text(s, 4), "a—");
        assert_eq!(truncate_text(s, 5), "a—b");

        // 4-byte character (emoji '🦀' = F0 9F A6 80).
        let s = "hi🦀!"; // 2 + 4 + 1 = 7 bytes
        assert_eq!(truncate_text(s, 3), "hi"); // can't fit partial emoji
        assert_eq!(truncate_text(s, 6), "hi🦀");
        assert_eq!(truncate_text(s, 7), "hi🦀!");

        // Realistic: truncate a string with mixed multi-byte content at a budget
        // that lands in the middle of a multi-byte sequence.
        let s = "café résumé"; // each 'é' is 2 bytes
        let truncated = truncate_text(s, 5);
        // "caf" = 3 bytes, "é" = 2 bytes → "café" = 5 bytes, fits exactly.
        assert_eq!(truncated, "caf\u{e9}");
    }

    #[tokio::test]
    async fn smoke() {
        let store = Store::new(None).await.unwrap();

        let artifact = Artifact {
            id: "abc".into(),
            timestamp: 1700000000,
            author: "tester".into(),
            contents: "hello, testy".into(),
            embedding: vec![],
        };

        store.upsert(&artifact).await.unwrap();

        let mut found = Artifact {
            id: Text::EMPTY,
            timestamp: 0,
            author: Text::EMPTY,
            contents: Text::EMPTY,
            embedding: vec![],
        };
        assert!(store.get("abc".into(), &mut found).await.is_ok());
        assert_eq!(format!("{}", found.contents), "hello, testy");
        assert_eq!(found.timestamp, 1700000000);
        assert_eq!(format!("{}", found.author), "tester");

        let mut not_found = Text::EMPTY;
        assert_eq!(
            Err(Error::NotFound),
            store.get("xyz".into(), &mut not_found).await
        );
    }

    #[tokio::test]
    async fn count_browse_filter() {
        let store = Store::new(None).await.unwrap();

        // Insert a few artifacts.
        for i in 0..3 {
            let artifact = Artifact {
                id: format!("art-{i}").into(),
                timestamp: 1000 + i as u64,
                author: if i == 0 { "alice".into() } else { "bob".into() },
                contents: format!("content {i}").into(),
                embedding: vec![],
            };
            store.upsert(&artifact).await.unwrap();
        }

        assert_eq!(store.count().await.unwrap(), 3);

        // Browse descending.
        let browsed = store.browse(0, 10, BrowseOrder::Desc).await.unwrap();
        assert_eq!(browsed.len(), 3);
        assert_eq!(browsed[0].timestamp, 1002);

        // Browse ascending.
        let browsed = store.browse(0, 10, BrowseOrder::Asc).await.unwrap();
        assert_eq!(browsed[0].timestamp, 1000);

        // Browse with offset/limit.
        let browsed = store.browse(1, 1, BrowseOrder::Desc).await.unwrap();
        assert_eq!(browsed.len(), 1);
        assert_eq!(browsed[0].timestamp, 1001);

        // Filter by author.
        let filtered = store
            .filter(Some("alice".to_string()), None, None, 10)
            .await
            .unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(format!("{}", filtered[0].id), "art-0");

        // Filter by time range.
        let filtered = store
            .filter(None, Some(1001), Some(1002), 10)
            .await
            .unwrap();
        assert_eq!(filtered.len(), 2);

        // Filter with limit.
        let filtered = store.filter(None, None, None, 2).await.unwrap();
        assert_eq!(filtered.len(), 2);
    }
}
