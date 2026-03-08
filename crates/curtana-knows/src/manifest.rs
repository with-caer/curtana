use std::{collections::BTreeMap, fmt, path::Path};

use serde::{Deserialize, Serialize};

/// Persistent record of all discovered taxonomies and their metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub taxonomies: BTreeMap<String, TaxonomyEntry>,
}

/// Metadata for a single taxonomy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaxonomyEntry {
    /// Human-readable name (same as the BTreeMap key).
    pub name: String,
    /// LLM-generated description of the taxonomy's contents.
    /// Empty until first ingestion and description generation.
    pub description: String,
    /// Source type that produced this taxonomy (e.g. `"imap"`).
    pub source_type: String,
    /// Source-specific identifier (e.g. IMAP folder name).
    pub source_id: String,
    /// Source host for reconnection context (no password stored).
    pub source_host: String,
    /// Source username for reconnection context.
    pub source_username: String,
    /// Unix timestamp of the last successful ingestion.
    pub last_ingested_at: Option<u64>,
    /// Unix timestamp of the last description generation.
    pub description_updated_at: Option<u64>,
    /// Opaque cursor for incremental source fetching.
    /// Format is source-specific (e.g. JSON-encoded IMAP UID state).
    #[serde(default)]
    pub cursor: Option<String>,
}

/// Errors that can occur when loading or saving a manifest.
#[derive(Debug)]
pub enum ManifestError {
    Io(std::io::Error),
    Parse(toml::de::Error),
    Serialize(toml::ser::Error),
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ManifestError::Io(e) => write!(f, "manifest I/O error: {e}"),
            ManifestError::Parse(e) => write!(f, "manifest parse error: {e}"),
            ManifestError::Serialize(e) => write!(f, "manifest serialize error: {e}"),
        }
    }
}

impl From<std::io::Error> for ManifestError {
    fn from(e: std::io::Error) -> Self {
        ManifestError::Io(e)
    }
}

impl From<toml::de::Error> for ManifestError {
    fn from(e: toml::de::Error) -> Self {
        ManifestError::Parse(e)
    }
}

impl From<toml::ser::Error> for ManifestError {
    fn from(e: toml::ser::Error) -> Self {
        ManifestError::Serialize(e)
    }
}

/// Lowercases and replaces non-alphanumeric characters with hyphens,
/// collapsing consecutive hyphens and trimming leading/trailing hyphens.
fn sanitize_key_part(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

impl TaxonomyEntry {
    /// Produces a filesystem-safe key from `source_type`, `source_host`, and
    /// `source_username`. Taxonomies sharing the same source key can share a
    /// single DuckDB file, with a `taxonomy` column discriminating rows.
    pub fn source_key(&self) -> String {
        format!(
            "{}-{}-{}",
            sanitize_key_part(&self.source_type),
            sanitize_key_part(&self.source_host),
            sanitize_key_part(&self.source_username),
        )
    }
}

impl Manifest {
    /// Loads a manifest from `path`. Returns an empty manifest if the file
    /// does not exist.
    pub fn load(path: &Path) -> Result<Self, ManifestError> {
        match std::fs::read_to_string(path) {
            Ok(contents) => Ok(toml::from_str(&contents)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.into()),
        }
    }

    /// Persists the manifest to `path`, creating parent directories
    /// if necessary. Uses atomic write (write to temp, then rename)
    /// to avoid partial writes.
    pub fn save(&self, path: &Path) -> Result<(), ManifestError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = toml::to_string_pretty(self)?;
        let tmp_path = path.with_extension("toml.tmp");
        std::fs::write(&tmp_path, contents)?;
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut manifest = Manifest::default();
        manifest.taxonomies.insert(
            "imap-knowledge".to_string(),
            TaxonomyEntry {
                name: "imap-knowledge".to_string(),
                description: "Technical reference material.".to_string(),
                source_type: "imap".to_string(),
                source_id: "knowledge".to_string(),
                source_host: "imap.example.com".to_string(),
                source_username: "user@example.com".to_string(),
                last_ingested_at: Some(1700000000),
                description_updated_at: Some(1700000000),
                cursor: Some(r#"{"uid_validity":42,"max_uid":100}"#.to_string()),
            },
        );

        let dir = std::env::temp_dir().join("curtana-manifest-test");
        let path = dir.join("manifest.toml");

        manifest.save(&path).unwrap();
        let loaded = Manifest::load(&path).unwrap();
        assert_eq!(manifest, loaded);

        // Clean up.
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_missing_returns_empty() {
        let path = std::env::temp_dir().join("curtana-does-not-exist.toml");
        let manifest = Manifest::load(&path).unwrap();
        assert!(manifest.taxonomies.is_empty());
    }

    #[test]
    fn source_key_basic() {
        let entry = TaxonomyEntry {
            name: "test".to_string(),
            description: String::new(),
            source_type: "imap".to_string(),
            source_id: "INBOX".to_string(),
            source_host: "imap.gmail.com".to_string(),
            source_username: "user@gmail.com".to_string(),
            last_ingested_at: None,
            description_updated_at: None,
            cursor: None,
        };
        assert_eq!(entry.source_key(), "imap-imap-gmail-com-user-gmail-com");
    }

    #[test]
    fn source_key_collapses_special_chars() {
        let entry = TaxonomyEntry {
            name: "test".to_string(),
            description: String::new(),
            source_type: "imap".to_string(),
            source_id: "INBOX".to_string(),
            source_host: "mail..example...com".to_string(),
            source_username: "u@e".to_string(),
            last_ingested_at: None,
            description_updated_at: None,
            cursor: None,
        };
        assert_eq!(entry.source_key(), "imap-mail-example-com-u-e");
    }

    #[test]
    fn sanitize_key_part_edge_cases() {
        assert_eq!(super::sanitize_key_part(""), "");
        assert_eq!(super::sanitize_key_part("---"), "");
        assert_eq!(super::sanitize_key_part("ABC"), "abc");
        assert_eq!(super::sanitize_key_part("a..b@@c"), "a-b-c");
    }
}
