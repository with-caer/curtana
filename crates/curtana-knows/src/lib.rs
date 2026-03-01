extern crate alloc;

pub mod manifest;
pub mod router;
pub mod tools;

use std::{path::Path, sync::Arc};

use codas::{
    codec::{Decodable, ReadsDecodable, WritesEncodable},
    types::Text,
};
use curtana_infers::TextEmbeddingModel;
use duckdb::params;
use tokio::sync::Mutex;
use tracing::{info, trace};

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
    pub async fn new(path: Option<&str>) -> Self {
        let connection = match path {
            Some(path) => duckdb::Connection::open(path).unwrap(),
            None => duckdb::Connection::open_in_memory().unwrap(),
        };

        let this = Self {
            connection: Arc::new(Mutex::new(connection)),
        };

        let this_copy = this.clone();
        tokio::task::spawn_blocking(move || {
            let connection = this_copy.connection.blocking_lock();

            connection
                .execute(
                    "CREATE TABLE IF NOT EXISTS artifacts (id VARCHAR PRIMARY KEY, data BLOB);",
                    [],
                )
                .unwrap();

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
        })
        .await
        .unwrap();

        this
    }

    pub async fn upsert(&self, artifact: &Artifact) {
        let this = self.clone();

        let id = format!("{}", artifact.id);
        let timestamp = artifact.timestamp as i64;
        let author = format!("{}", artifact.author);

        let mut data_bytes = vec![];
        data_bytes.write_data(artifact).unwrap();

        tokio::task::spawn_blocking(move || {
            let connection = this.connection.blocking_lock();

            let affected_rows = connection
                .execute(
                    "INSERT OR REPLACE INTO artifacts (id, data, timestamp, author) VALUES (?, ?, ?, ?)",
                    params![id.as_str(), data_bytes, timestamp, author.as_str()],
                )
                .unwrap();

            assert_eq!(1, affected_rows);
        })
        .await
        .unwrap()
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
        .unwrap()
        .ok_or(Error::NotFound)?;

        data_bytes.as_slice().read_data_into(data).unwrap();

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
    ) {
        let this = self.clone();

        // Find all unembedded artifacts.
        let unembedded: Vec<(String, Artifact)> = tokio::task::spawn_blocking(move || {
            let connection = this.connection.blocking_lock();

            let mut statement = connection
                .prepare("SELECT id, data FROM artifacts")
                .unwrap();
            let rows = statement
                .query_map([], |row| {
                    let id: String = row.get(0).unwrap();
                    let data: Vec<u8> = row.get(1).unwrap();
                    let artifact: Artifact = data.as_slice().read_data().unwrap();

                    if artifact.embedding.is_empty() {
                        Ok(Some((id, artifact)))
                    } else {
                        Ok(None)
                    }
                })
                .unwrap();

            rows.into_iter()
                .filter_map(|row| {
                    if let Ok(Some(row)) = row {
                        Some(row)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .await
        .unwrap();

        let total = unembedded.len();
        info!("embedding {} artifacts", total);
        on_progress(0, total);

        for (i, (_artifact_id, mut artifact)) in unembedded.into_iter().enumerate() {
            let text = format!("{}", artifact.contents);
            artifact.embedding = embed_with_chunking(model, &text);
            self.upsert(&artifact).await;
            on_progress(i + 1, total);

            if i % 50 == 0 {
                info!("embedded {i} artifacts...");
            }
        }
    }

    /// Returns all artifacts that have embeddings.
    pub async fn all_embedded(&self) -> Vec<Artifact> {
        let this = self.clone();

        tokio::task::spawn_blocking(move || {
            let connection = this.connection.blocking_lock();

            let mut statement = connection.prepare("SELECT data FROM artifacts").unwrap();
            let rows = statement
                .query_map([], |row| {
                    let data: Vec<u8> = row.get(0).unwrap();
                    let artifact: Artifact = data.as_slice().read_data().unwrap();
                    Ok(artifact)
                })
                .unwrap();

            rows.into_iter()
                .filter_map(|r| r.ok())
                .filter(|a| !a.embedding.is_empty())
                .collect()
        })
        .await
        .unwrap()
    }

    /// Returns a random sample of up to `limit` artifacts from the store.
    pub async fn sample(&self, limit: usize) -> Vec<Artifact> {
        let this = self.clone();

        tokio::task::spawn_blocking(move || {
            let connection = this.connection.blocking_lock();

            let mut statement = connection
                .prepare("SELECT data FROM artifacts ORDER BY RANDOM() LIMIT ?")
                .unwrap();
            let rows = statement
                .query_map(params![limit], |row| {
                    let data: Vec<u8> = row.get(0).unwrap();
                    let artifact: Artifact = data.as_slice().read_data().unwrap();
                    Ok(artifact)
                })
                .unwrap();

            rows.into_iter().filter_map(|r| r.ok()).collect()
        })
        .await
        .unwrap()
    }

    /// Finds the `top_k` artifacts most similar to `query` in the datastore.
    pub async fn search(
        &self,
        model: &mut TextEmbeddingModel,
        query: &str,
        top_k: usize,
    ) -> Vec<Artifact> {
        let this = self.clone();

        // Find all embedded artifacts.
        let embedded: Vec<Artifact> = tokio::task::spawn_blocking(move || {
            let connection = this.connection.blocking_lock();

            let mut statement = connection.prepare("SELECT data FROM artifacts").unwrap();
            let rows = statement
                .query_map([], |row| {
                    let data: Vec<u8> = row.get(0).unwrap();
                    let artifact: Artifact = data.as_slice().read_data().unwrap();
                    Ok(artifact)
                })
                .unwrap();

            rows.into_iter()
                .filter_map(|r| r.ok())
                .filter(|a| !a.embedding.is_empty())
                .collect::<Vec<_>>()
        })
        .await
        .unwrap();

        // Embed the query.
        let query_embedding = model.embed(&[query]).unwrap().pop().unwrap();

        // Score each artifact by its best-matching chunk embedding.
        let mut artifacts: Vec<_> = embedded
            .into_iter()
            .map(|artifact| {
                let score = best_chunk_score(&artifact.embedding, &query_embedding);
                (score, artifact)
            })
            .collect();
        artifacts.sort_by(|a, b| a.0.total_cmp(&b.0).reverse());

        // Truncate to top-k.
        artifacts.truncate(top_k);
        artifacts
            .into_iter()
            .map(|(_, artifact)| artifact)
            .collect()
    }

    /// Returns the total number of artifacts in the store.
    pub async fn count(&self) -> usize {
        let this = self.clone();

        tokio::task::spawn_blocking(move || {
            let connection = this.connection.blocking_lock();
            connection
                .query_row("SELECT COUNT(*) FROM artifacts", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap() as usize
        })
        .await
        .unwrap()
    }

    /// Browses artifacts ordered by timestamp.
    pub async fn browse(&self, offset: usize, limit: usize, order: BrowseOrder) -> Vec<Artifact> {
        let this = self.clone();
        let order_sql = order.sql().to_string();

        tokio::task::spawn_blocking(move || {
            let connection = this.connection.blocking_lock();

            let sql = format!(
                "SELECT data FROM artifacts ORDER BY COALESCE(timestamp, 0) {} LIMIT ? OFFSET ?",
                order_sql,
            );
            let mut statement = connection.prepare(&sql).unwrap();
            let rows = statement
                .query_map(params![limit as i64, offset as i64], |row| {
                    let data: Vec<u8> = row.get(0).unwrap();
                    let artifact: Artifact = data.as_slice().read_data().unwrap();
                    Ok(artifact)
                })
                .unwrap();

            rows.into_iter().filter_map(|r| r.ok()).collect()
        })
        .await
        .unwrap()
    }

    /// Filters artifacts by author and/or time range.
    pub async fn filter(
        &self,
        author: Option<String>,
        after: Option<u64>,
        before: Option<u64>,
        limit: usize,
    ) -> Vec<Artifact> {
        let this = self.clone();

        tokio::task::spawn_blocking(move || {
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
                .unwrap();

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
                        let data: Vec<u8> = row.get(0).unwrap();
                        let artifact: Artifact = data.as_slice().read_data().unwrap();
                        Ok(artifact)
                    },
                )
                .unwrap();

            rows.into_iter().filter_map(|r| r.ok()).collect()
        })
        .await
        .unwrap()
    }
}

/// Embeds `text`, chunking only if the model's context window is exceeded.
/// Returns one embedding per chunk.
fn embed_with_chunking(model: &mut TextEmbeddingModel, text: &str) -> Vec<Vec<f32>> {
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
                    embed_with_chunking(model, chunk)
                })
                .collect()
        }
        Err(e) => panic!("embedding failed: {e:?}"),
    }
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
fn best_chunk_score(chunk_embeddings: &[Vec<f32>], query: &[f32]) -> f32 {
    chunk_embeddings
        .iter()
        .map(|chunk| curtana_infers::cosine_distance(chunk, query))
        .fold(f32::NEG_INFINITY, f32::max)
}

/// Opens a taxonomy-specific store at `{data_dir}/{taxonomy_name}.duckdb`.
pub async fn open_taxonomy_store(data_dir: &Path, taxonomy_name: &str) -> Store {
    let path = data_dir.join(format!("{taxonomy_name}.duckdb"));
    Store::new(Some(path.to_str().unwrap())).await
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Error {
    NotFound,
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
        let store = Store::new(None).await;

        let artifact = Artifact {
            id: "abc".into(),
            timestamp: 1700000000,
            author: "tester".into(),
            contents: "hello, testy".into(),
            embedding: vec![],
        };

        store.upsert(&artifact).await;

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
        let store = Store::new(None).await;

        // Insert a few artifacts.
        for i in 0..3 {
            let artifact = Artifact {
                id: format!("art-{i}").into(),
                timestamp: 1000 + i as u64,
                author: if i == 0 { "alice".into() } else { "bob".into() },
                contents: format!("content {i}").into(),
                embedding: vec![],
            };
            store.upsert(&artifact).await;
        }

        assert_eq!(store.count().await, 3);

        // Browse descending.
        let browsed = store.browse(0, 10, BrowseOrder::Desc).await;
        assert_eq!(browsed.len(), 3);
        assert_eq!(browsed[0].timestamp, 1002);

        // Browse ascending.
        let browsed = store.browse(0, 10, BrowseOrder::Asc).await;
        assert_eq!(browsed[0].timestamp, 1000);

        // Browse with offset/limit.
        let browsed = store.browse(1, 1, BrowseOrder::Desc).await;
        assert_eq!(browsed.len(), 1);
        assert_eq!(browsed[0].timestamp, 1001);

        // Filter by author.
        let filtered = store
            .filter(Some("alice".to_string()), None, None, 10)
            .await;
        assert_eq!(filtered.len(), 1);
        assert_eq!(format!("{}", filtered[0].id), "art-0");

        // Filter by time range.
        let filtered = store.filter(None, Some(1001), Some(1002), 10).await;
        assert_eq!(filtered.len(), 2);

        // Filter with limit.
        let filtered = store.filter(None, None, None, 2).await;
        assert_eq!(filtered.len(), 2);
    }
}
