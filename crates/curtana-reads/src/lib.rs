use serde::Deserialize;

pub mod imap;

#[derive(Deserialize)]
pub struct ImapConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
}

/// A raw message read from an external source.
///
/// This is a source-agnostic representation that callers
/// can convert into their own internal types.
#[derive(Debug, Clone)]
pub struct RawMessage {
    pub id: String,
    pub timestamp: i64,
    pub author: String,
    pub body: String,
    pub thread_id: Option<String>,
}
