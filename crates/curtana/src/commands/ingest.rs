use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use curtana_knows::manifest::Manifest;
use curtana_knows::{Artifact, open_taxonomy_store, router};
use curtana_reads::ToMarkdown;
use curtana_reads::imap;
use tokio::sync::mpsc;

use crate::config::{Config, SourceConfig};
use crate::event::{CommandResult, Event};

use super::Models;

/// Runs the full ingest pipeline, sending progress as `Event::Token`.
pub(crate) async fn run(
    config: &Config,
    models: &mut Models,
    tx: &mpsc::UnboundedSender<Event>,
) {
    let data_dir = Path::new(config.data_dir());
    let manifest_path = data_dir.join("manifest.toml");

    let mut manifest = match Manifest::load(&manifest_path) {
        Ok(m) => m,
        Err(e) => {
            tx.send(Event::Error(format!("failed to load manifest: {e}")))
                .ok();
            return;
        }
    };

    if manifest.taxonomies.is_empty() {
        tx.send(Event::CommandDone(CommandResult::Message(
            "No taxonomies in manifest \u{2014} run /discover first.".into(),
        )))
        .ok();
        return;
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let taxonomy_names: Vec<String> = manifest.taxonomies.keys().cloned().collect();

    for taxonomy_name in &taxonomy_names {
        let entry = manifest.taxonomies.get(taxonomy_name).unwrap().clone();

        send_progress(tx, &format!("Ingesting {taxonomy_name}...\n"));

        // Fetch artifacts from source.
        let artifacts = match find_source(config, &entry.source_type) {
            Some(source) => fetch_artifacts(source, &entry.source_id).await,
            None => {
                send_progress(tx, &format!("  No matching source config, skipping.\n"));
                continue;
            }
        };

        send_progress(tx, &format!("  Fetched {} artifacts.\n", artifacts.len()));

        // Upsert into taxonomy store.
        let store = open_taxonomy_store(data_dir, taxonomy_name).await;
        for artifact in &artifacts {
            store.upsert(artifact.id.clone(), artifact).await;
        }

        // Embed pending artifacts.
        send_progress(tx, "  Embedding...\n");
        store.embed_pending(&mut models.embed).await;

        // Update ingestion timestamp.
        if let Some(entry) = manifest.taxonomies.get_mut(taxonomy_name) {
            entry.last_ingested_at = Some(now);
        }

        // Generate description if empty.
        if let Some(entry) = manifest.taxonomies.get_mut(taxonomy_name)
            && entry.description.is_empty()
        {
            send_progress(tx, "  Generating description...\n");
            let description = router::generate_description(&store, &mut models.chat, 10).await;
            entry.description = description;
            entry.description_updated_at = Some(now);
        }

        send_progress(tx, &format!("  Done.\n"));
    }

    if let Err(e) = manifest.save(&manifest_path) {
        tx.send(Event::Error(format!("failed to save manifest: {e}")))
            .ok();
        return;
    }

    tx.send(Event::CommandDone(CommandResult::Message(
        "Ingestion complete.".into(),
    )))
    .ok();
}

fn send_progress(tx: &mpsc::UnboundedSender<Event>, text: &str) {
    tx.send(Event::Token(text.to_string())).ok();
}

/// Finds the source config matching a given source type.
fn find_source<'a>(config: &'a Config, source_type: &str) -> Option<&'a SourceConfig> {
    config
        .source
        .iter()
        .find(|s| matches!((source_type, s), ("imap", SourceConfig::Imap(_))))
}

/// Fetches artifacts from a source folder.
async fn fetch_artifacts(source: &SourceConfig, folder_name: &str) -> Vec<Artifact> {
    match source {
        SourceConfig::Imap(base_config) => {
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
