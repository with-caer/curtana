use std::path::Path;

use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    /// Data sources to discover and ingest from.
    pub source: Vec<SourceConfig>,
    /// Directory for taxonomy databases and manifest.
    /// Defaults to `"./.curtana"`.
    pub data_dir: Option<String>,
    /// Path to a chat model GGUF file.
    pub chat_model: String,
    /// Path to a text embedding model GGUF file.
    pub embed_model: String,
    /// Enable agent mode for queries (tool-use loop).
    /// Defaults to `true`.
    pub agent_mode: Option<bool>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum SourceConfig {
    #[serde(rename = "imap")]
    Imap(curtana_reads::imap::ImapConfig),
}

impl Config {
    pub fn load(path: &Path) -> Self {
        let contents = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("failed to read config at {}: {e}", path.display()));
        toml::from_str(&contents)
            .unwrap_or_else(|e| panic!("failed to parse config at {}: {e}", path.display()))
    }

    pub fn data_dir(&self) -> &str {
        self.data_dir.as_deref().unwrap_or("./.curtana")
    }

    pub fn use_agent_mode(&self) -> bool {
        self.agent_mode.unwrap_or(true)
    }
}
