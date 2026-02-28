extern crate alloc;

use std::sync::Arc;

use codas::{
    codec::{Decodable, Encodable, ReadsDecodable, WritesEncodable},
    types::Text,
};
use curtana_infers::TextEmbeddingModel;
use duckdb::params;
use tokio::sync::Mutex;

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
    /// that have text contents and don't already have embeddings.
    pub async fn embed_pending(&self, model: &mut TextEmbeddingModel) {
        let this = self.clone();

        // Find all unembedded artifacts.
        let unembedded: Vec<(String, Artifact)> = tokio::task::spawn_blocking(move || {
            let connection = this.connection.blocking_lock();

            let mut statement = connection.prepare("SELECT id, data FROM artifacts").unwrap();
            let rows = statement
                .query_map([], |row| {
                    let id: String = row.get(0).unwrap();
                    let data: Vec<u8> = row.get(1).unwrap();

                    let artifact: Artifact = data.as_slice().read_data().unwrap();

                    if artifact.embedding.is_none() {
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

        eprintln!("embedding {} artifacts", unembedded.len());

        for (i, (artifact_id, mut artifact)) in unembedded.into_iter().enumerate() {
            if let codas::types::dynamic::Unspecified::Text(ref text) = artifact.contents {
                let mut embedding = model.embed(&[format!("{}", text)]).unwrap();
                artifact.embedding = Some(embedding.pop().unwrap());
                self.upsert(artifact_id.into(), &artifact).await;
            }

            if i % 50 == 0 {
                eprintln!("embedded {i} artifacts...");
            }
        }
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
        let embedded: Vec<(String, Artifact)> = tokio::task::spawn_blocking(move || {
            let connection = this.connection.blocking_lock();

            let mut statement = connection.prepare("SELECT id, data FROM artifacts").unwrap();
            let rows = statement
                .query_map([], |row| {
                    let id: String = row.get(0).unwrap();
                    let data: Vec<u8> = row.get(1).unwrap();

                    let artifact: Artifact = data.as_slice().read_data().unwrap();

                    if artifact.embedding.is_some() {
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

        // Embed the query.
        let query_embedding = model.embed(&[format!("{}", query)]).unwrap().pop().unwrap();

        // Calculate cosines and rank artifacts by similarity.
        let mut artifacts: Vec<_> = embedded
            .into_iter()
            .map(|(id, artifact)| {
                (
                    curtana_infers::cosine_distance(
                        artifact.embedding.as_ref().unwrap(),
                        &query_embedding,
                    ),
                    id,
                    artifact,
                )
            })
            .collect();
        artifacts.sort_by(|a, b| a.0.total_cmp(&b.0).reverse());

        // Truncate to top-k.
        artifacts.truncate(top_k);
        artifacts.into_iter().map(|(_, _, artifact)| artifact).collect()
    }
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
