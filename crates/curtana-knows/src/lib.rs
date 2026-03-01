extern crate alloc;

pub mod manifest;
pub mod router;

use std::{path::Path, sync::Arc};

use codas::{
    codec::{Decodable, Encodable, ReadsDecodable, WritesEncodable},
    types::Text,
};
use curtana_infers::TextEmbeddingModel;
use duckdb::params;
use tokio::sync::Mutex;
use tracing::{info, trace};

codas_macros::export_coda!("crates/curtana-knows/src/coda.md");

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
        })
        .await
        .unwrap();

        this
    }

    pub async fn upsert(&self, id: Text, data: &impl Encodable) {
        let this = self.clone();

        let mut data_bytes = vec![];
        data_bytes.write_data(data).unwrap();

        tokio::task::spawn_blocking(move || {
            let connection = this.connection.blocking_lock();

            let affected_rows = connection
                .execute(
                    "INSERT OR REPLACE INTO artifacts (id, data) VALUES (?, ?)",
                    params![id.as_str(), data_bytes],
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
    pub async fn embed_pending(&self, model: &mut TextEmbeddingModel) {
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

        info!("embedding {} artifacts", unembedded.len());

        for (i, (artifact_id, mut artifact)) in unembedded.into_iter().enumerate() {
            let text = format!("{}", artifact.contents);
            artifact.embedding = embed_with_chunking(model, &text);
            self.upsert(artifact_id.into(), &artifact).await;

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

    #[tokio::test]
    async fn smoke() {
        let store = Store::new(None).await;

        let data = Text::from("hello, testy");

        store.upsert("abc".into(), &data).await;

        let mut found_data = Text::EMPTY;
        assert!(store.get("abc".into(), &mut found_data).await.is_ok());

        assert_eq!(data, found_data);
        assert_eq!(
            Err(Error::NotFound),
            store.get("xyz".into(), &mut found_data).await
        );
    }
}
