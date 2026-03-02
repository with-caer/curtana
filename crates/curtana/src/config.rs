use std::path::{Path, PathBuf};
use std::{fmt, io};

use serde::Deserialize;

use crate::commands::setup;

#[derive(Deserialize)]
pub struct Config {
    /// Data sources to explore and read from.
    #[serde(default)]
    pub source: Vec<SourceConfig>,
    /// Directory for taxonomy databases and manifest.
    pub data_dir: Option<String>,
    /// Path to a chat model GGUF file.
    pub chat_model: Option<String>,
    /// Path to a text embedding model GGUF file.
    pub embed_model: Option<String>,
    /// Enable agent mode for queries (tool-use loop).
    /// Defaults to `true`.
    pub agent_mode: Option<bool>,

    /// Path to the config file that was loaded (not serialized).
    #[serde(skip)]
    config_path: Option<PathBuf>,
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
        let mut config: Config = toml::from_str(&contents).map_err(ConfigError::Parse)?;
        config.config_path = path
            .canonicalize()
            .ok()
            .or_else(|| Some(path.to_path_buf()));
        Ok(config)
    }

    /// Returns the data directory as a `PathBuf`.
    ///
    /// Resolution order:
    /// 1. Explicit `data_dir` from config
    /// 2. Parent directory of the loaded config file
    /// 3. Fallback `./.curtana`
    pub fn data_dir(&self) -> PathBuf {
        if let Some(ref dir) = self.data_dir {
            return PathBuf::from(dir);
        }
        if let Some(ref config_path) = self.config_path
            && let Some(parent) = config_path.parent()
        {
            return parent.to_path_buf();
        }
        PathBuf::from("./.curtana")
    }

    /// Returns the resolved path to the chat model GGUF file.
    pub fn chat_model_path(&self) -> PathBuf {
        if let Some(ref path) = self.chat_model {
            PathBuf::from(path)
        } else {
            self.data_dir()
                .join("models")
                .join(setup::DEFAULT_CHAT_MODEL_FILENAME)
        }
    }

    /// Returns the resolved path to the embedding model GGUF file.
    pub fn embed_model_path(&self) -> PathBuf {
        if let Some(ref path) = self.embed_model {
            PathBuf::from(path)
        } else {
            self.data_dir()
                .join("models")
                .join(setup::DEFAULT_EMBED_MODEL_FILENAME)
        }
    }

    pub fn use_agent_mode(&self) -> bool {
        self.agent_mode.unwrap_or(true)
    }
}

/// Resolves the config file path.
///
/// 1. Explicit `-c` path → use directly
/// 2. `./Curtana.toml` exists → use it (backwards compat)
/// 3. `~/.curtana/Curtana.toml` exists → use it
/// 4. Error with "run `curtana setup`" hint
pub fn resolve_config_path(explicit: Option<&Path>) -> Result<PathBuf, ConfigError> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }

    let local = PathBuf::from("Curtana.toml");
    if local.exists() {
        return Ok(local);
    }

    if let Ok(home_dir) = setup::home_curtana_dir() {
        let home_config = home_dir.join("Curtana.toml");
        if home_config.exists() {
            return Ok(home_config);
        }
    }

    Err(ConfigError::NotFound)
}

#[derive(Debug)]
pub enum ConfigError {
    Io(io::Error),
    Parse(toml::de::Error),
    NotFound,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Io(e) => write!(f, "config I/O error: {e}"),
            ConfigError::Parse(e) => write!(f, "config parse error: {e}"),
            ConfigError::NotFound => write!(
                f,
                "no Curtana.toml found. Run `curtana setup` to get started."
            ),
        }
    }
}
