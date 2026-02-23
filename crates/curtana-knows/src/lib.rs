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

/// Datastore for retrieval-augmented interactive chat.
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
                    "CREATE TABLE IF NOT EXISTS chats (id VARCHAR PRIMARY KEY, data BLOB);",
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
                    "INSERT OR REPLACE INTO chats (id, data) VALUES (?, ?)",
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
                    "SELECT data FROM chats WHERE id=(?)",
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

    /// Embeds the bodies of all messages currently stored
    /// in the datastore that don't already have embeddings.
    pub async fn embed_pending(&self, model: &mut TextEmbeddingModel) {
        let this = self.clone();

        // Find all unembedded chats.
        let unembedded_chats: Vec<(String, ChatMessage)> = tokio::task::spawn_blocking(move || {
            let connection = this.connection.blocking_lock();

            // Scan all rows.
            let mut statement = connection.prepare("SELECT id, data FROM chats").unwrap();
            let unembedded_chats = statement
                .query_map([], |row| {
                    let id: String = row.get(0).unwrap();
                    let data: Vec<u8> = row.get(1).unwrap();

                    let chat: ChatMessage = data.as_slice().read_data().unwrap();

                    if chat.embedding.is_none() {
                        Ok(Some((id, chat)))
                    } else {
                        Ok(None)
                    }
                })
                .unwrap();

            // Return all unembedded chats.
            unembedded_chats
                .into_iter()
                .filter_map(|chat| {
                    if let Ok(Some(chat)) = chat {
                        Some(chat)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .await
        .unwrap();

        // Notify operator.
        eprintln!("embedding {} chats", unembedded_chats.len());

        // Embed all the chats' bodies.
        for (i, (chat_id, mut chat)) in unembedded_chats.into_iter().enumerate() {
            let mut embedding = model.embed(&[format!("{}", chat.body)]).unwrap();
            chat.embedding = Some(embedding.pop().unwrap());
            self.upsert(chat_id.into(), &chat).await;

            if i % 50 == 0 {
                eprintln!("embedded {i} chats...");
            }
        }
    }

    /// Finds the `top_k` chats most similar to `query` in the datastore.
    pub async fn search(
        &self,
        model: &mut TextEmbeddingModel,
        query: &str,
        top_k: usize,
    ) -> Vec<ChatMessage> {
        let this = self.clone();

        // Find all embedded chats.
        let embedded_chats: Vec<(String, ChatMessage)> = tokio::task::spawn_blocking(move || {
            let connection = this.connection.blocking_lock();

            // Scan all rows that have embeddings.
            let mut statement = connection.prepare("SELECT id, data FROM chats").unwrap();
            let embedded_chats = statement
                .query_map([], |row| {
                    let id: String = row.get(0).unwrap();
                    let data: Vec<u8> = row.get(1).unwrap();

                    let chat: ChatMessage = data.as_slice().read_data().unwrap();

                    if chat.embedding.is_some() {
                        Ok(Some((id, chat)))
                    } else {
                        Ok(None)
                    }
                })
                .unwrap();

            // Return all embedded chats.
            embedded_chats
                .into_iter()
                .filter_map(|chat| {
                    if let Ok(Some(chat)) = chat {
                        Some(chat)
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

        // Calculate cosines and rank chats by cosines.
        let mut chats: Vec<_> = embedded_chats
            .into_iter()
            .map(|(id, chat)| {
                (
                    curtana_infers::cosine_distance(
                        chat.embedding.as_ref().unwrap(),
                        &query_embedding,
                    ),
                    id,
                    chat,
                )
            })
            .collect();
        chats.sort_by(|a, b| a.0.total_cmp(&b.0).reverse());

        // Truncate to top-k.
        chats.truncate(top_k);
        chats.into_iter().map(|(_, _, chat)| chat).collect()
    }

    /// Finds all chats in the same thread(s) as the chats in `chats`.
    pub async fn find_related(&self, chats: &[ChatMessage]) -> Vec<ChatMessage> {
        let threads: Vec<Text> = chats
            .iter()
            .filter_map(|chat| chat.thread_id.clone())
            .collect();

        if threads.is_empty() {
            return vec![];
        }

        let this = self.clone();

        // Find all related chats.
        let mut related_chats: Vec<ChatMessage> = tokio::task::spawn_blocking(move || {
            let connection = this.connection.blocking_lock();

            // Scan all rows.
            let mut statement = connection.prepare("SELECT data FROM chats").unwrap();
            let related_chats = statement
                .query_map([], |row| {
                    let data: Vec<u8> = row.get(0).unwrap();

                    let chat: ChatMessage = data.as_slice().read_data().unwrap();

                    if let Some(thread_id) = chat.thread_id.as_ref() {
                        if threads.contains(thread_id) {
                            return Ok(Some(chat));
                        }
                    }

                    Ok(None)
                })
                .unwrap();

            // Return all related chats.
            related_chats
                .into_iter()
                .filter_map(|chat| {
                    if let Ok(Some(chat)) = chat {
                        Some(chat)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .await
        .unwrap();

        // Deduplicate chats.
        related_chats.retain(|chat| !chats.iter().any(|other| other.body == chat.body));

        related_chats
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
