use std::{fmt, path::Path};

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
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path).map_err(ConfigError::Io)?;
        toml::from_str(&contents).map_err(ConfigError::Parse)
    }

    pub fn data_dir(&self) -> &str {
        self.data_dir.as_deref().unwrap_or("./.curtana")
    }

    pub fn use_agent_mode(&self) -> bool {
        self.agent_mode.unwrap_or(true)
    }
}

#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    Parse(toml::de::Error),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Io(e) => write!(f, "config I/O error: {e}"),
            ConfigError::Parse(e) => write!(f, "config parse error: {e}"),
        }
    }
}
