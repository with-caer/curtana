use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use curtana_infers::ModelRegistry;
use curtana_knows::manifest::Manifest;
use curtana_knows::{Artifact, open_taxonomy_store, router};
use curtana_reads::ToMarkdown;
use curtana_reads::imap;

use tracing::{info, warn};

use crate::config::{Config, SourceConfig};

/// For each taxonomy in the manifest: fetches artifacts from its source,
/// upserts into the taxonomy store, embeds pending artifacts, and
/// generates a description if empty or stale.
pub async fn run(config: &Config) {
    let data_dir = Path::new(config.data_dir());
    let manifest_path = data_dir.join("manifest.toml");
    let mut manifest = Manifest::load(&manifest_path).unwrap();

    if manifest.taxonomies.is_empty() {
        warn!("no taxonomies in manifest — run `curtana discover` first");
        return;
    }

    let registry = ModelRegistry::new().unwrap();
    let mut embed_model = registry
        .load_text_embedding_model(&config.embed_model)
        .unwrap();
    let mut chat_model = registry
        .load_chat_model(
            &config.chat_model,
            "You are a helpful assistant that summarizes collections of documents.",
        )
        .unwrap();

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let taxonomy_names: Vec<String> = manifest.taxonomies.keys().cloned().collect();

    for taxonomy_name in &taxonomy_names {
        let entry = manifest.taxonomies.get(taxonomy_name).unwrap().clone();
        info!("ingesting taxonomy: {taxonomy_name}");

        // Fetch artifacts from the source.
        let artifacts = match find_source(config, &entry.source_type, &entry.source_id) {
            Some(source) => fetch_artifacts(source, &entry.source_id).await,
            None => {
                warn!("no matching source config found, skipping");
                continue;
            }
        };

        info!("fetched {} artifacts", artifacts.len());

        // Upsert into taxonomy store.
        let store = open_taxonomy_store(data_dir, taxonomy_name).await;
        for artifact in &artifacts {
            store.upsert(artifact.id.clone(), artifact).await;
        }

        // Embed pending.
        store.embed_pending(&mut embed_model).await;

        // Update ingestion timestamp.
        if let Some(entry) = manifest.taxonomies.get_mut(taxonomy_name) {
            entry.last_ingested_at = Some(now);
        }

        // Generate description if empty.
        if let Some(entry) = manifest.taxonomies.get_mut(taxonomy_name)
            && entry.description.is_empty()
        {
            info!("generating description...");
            let description = router::generate_description(&store, &mut chat_model, 10).await;
            entry.description = description;
            entry.description_updated_at = Some(now);
        }
    }

    manifest.save(&manifest_path).unwrap();
    info!("manifest updated at {}", manifest_path.display());
}

/// Finds the source config that matches the given source type and folder.
fn find_source<'a>(
    config: &'a Config,
    source_type: &str,
    _source_id: &str,
) -> Option<&'a SourceConfig> {
    config
        .source
        .iter()
        .find(|s| matches!((source_type, s), ("imap", SourceConfig::Imap(_))))
}

/// Fetches artifacts from a source and converts them to the Artifact type.
async fn fetch_artifacts(source: &SourceConfig, folder_name: &str) -> Vec<Artifact> {
    match source {
        SourceConfig::Imap(base_config) => {
            // Create a config targeting the specific folder.
            let folder_config = imap::ImapConfig {
                host: base_config.host.clone(),
                port: base_config.port,
                username: base_config.username.clone(),
                password: base_config.password.clone(),
                mailbox: Some(folder_name.to_string()),
                sequence: base_config.sequence.clone(),
            };

            let emails = imap::fetch_emails(&folder_config).await;

            emails
                .into_iter()
                .map(|email| Artifact {
                    id: email.message_id.clone().into(),
                    timestamp: email.timestamp as u64,
                    author: email.from.clone().into(),
                    contents: email.to_markdown().into(),
                    embedding: Vec::new(),
                })
                .collect()
        }
    }
}
