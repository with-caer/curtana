use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use curtana_knows::manifest::Manifest;
use curtana_knows::{Artifact, open_taxonomy_store, router};
use curtana_reads::ToMarkdown;
use curtana_reads::imap;
use tokio::sync::mpsc;

use crate::config::{Config, SourceConfig};
use crate::event::{CommandResult, Event, Progress};

use super::Models;

/// Runs the full ingest pipeline, updating header status and progress bar.
pub(crate) async fn run(config: &Config, models: &mut Models, tx: &mpsc::UnboundedSender<Event>) {
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
    let mut summary_lines: Vec<String> = Vec::new();

    for taxonomy_name in &taxonomy_names {
        let entry = manifest.taxonomies.get(taxonomy_name).unwrap().clone();

        set_status(tx, &format!("Fetching {taxonomy_name}..."));

        // Fetch artifacts from source.
        let artifacts = match find_source(
            config,
            &entry.source_type,
            &entry.source_host,
            &entry.source_username,
        ) {
            Some(source) => match fetch_artifacts(source, &entry.source_id).await {
                Ok(a) => a,
                Err(e) => {
                    summary_lines.push(format!("- **{taxonomy_name}**: error ({e})"));
                    continue;
                }
            },
            None => {
                summary_lines.push(format!("- **{taxonomy_name}**: skipped (no source config)"));
                continue;
            }
        };

        let count = artifacts.len();

        // Upsert into taxonomy store.
        let store = match open_taxonomy_store(data_dir, taxonomy_name).await {
            Ok(s) => s,
            Err(e) => {
                summary_lines.push(format!("- **{taxonomy_name}**: error opening store ({e})"));
                continue;
            }
        };
        for artifact in &artifacts {
            if let Err(e) = store.upsert(artifact).await {
                summary_lines.push(format!("- **{taxonomy_name}**: upsert error ({e})"));
                continue;
            }
        }

        // Embed pending artifacts.
        let embed_label = format!("Embedding {taxonomy_name}");
        if let Err(e) = store
            .embed_pending(&mut models.embed, |current, total| {
                tx.send(Event::Progress(Progress {
                    current,
                    total,
                    label: embed_label.clone(),
                }))
                .ok();
            })
            .await
        {
            summary_lines.push(format!("- **{taxonomy_name}**: embedding error ({e})"));
            continue;
        }

        // Update ingestion timestamp.
        if let Some(entry) = manifest.taxonomies.get_mut(taxonomy_name) {
            entry.last_ingested_at = Some(now);
        }

        // Generate description if empty.
        if let Some(entry) = manifest.taxonomies.get_mut(taxonomy_name)
            && entry.description.is_empty()
        {
            set_status(
                tx,
                &format!("Generating description for {taxonomy_name}..."),
            );
            let description = router::generate_description(&store, &mut models.chat, 10).await;
            entry.description = description;
            entry.description_updated_at = Some(now);
        }

        let artifact_word = if count == 1 { "artifact" } else { "artifacts" };
        summary_lines.push(format!("- **{taxonomy_name}**: {count} {artifact_word}"));
    }

    if let Err(e) = manifest.save(&manifest_path) {
        tx.send(Event::Error(format!("failed to save manifest: {e}")))
            .ok();
        return;
    }

    let summary = format!("## Ingestion complete\n\n{}", summary_lines.join("\n"));
    tx.send(Event::CommandDone(CommandResult::Message(summary)))
        .ok();
}

fn set_status(tx: &mpsc::UnboundedSender<Event>, text: &str) {
    tx.send(Event::StatusText(text.to_string())).ok();
}

/// Finds the source config matching a given source type, host, and username.
fn find_source<'a>(
    config: &'a Config,
    source_type: &str,
    source_host: &str,
    source_username: &str,
) -> Option<&'a SourceConfig> {
    config.source.iter().find(|s| match (source_type, s) {
        ("imap", SourceConfig::Imap(c)) => c.host == source_host && c.username == source_username,
        _ => false,
    })
}

/// Fetches artifacts from a source folder.
async fn fetch_artifacts(
    source: &SourceConfig,
    folder_name: &str,
) -> Result<Vec<Artifact>, String> {
    match source {
        SourceConfig::Imap(base_config) => {
            let folder_config = imap::ImapConfig {
                host: base_config.host.clone(),
                port: base_config.port,
                username: base_config.username.clone(),
                password: base_config.password.clone(),
                mailbox: Some(folder_name.to_string()),
                sequence: base_config.sequence.clone(),
                accept_invalid_certs: base_config.accept_invalid_certs,
            };

            let emails = imap::fetch_emails(&folder_config)
                .await
                .map_err(|e| e.to_string())?;

            Ok(emails
                .into_iter()
                .map(|email| Artifact {
                    id: email.message_id.clone().into(),
                    timestamp: email.timestamp as u64,
                    author: email.from.clone().into(),
                    contents: email.to_markdown().into(),
                    embedding: Vec::new(),
                })
                .collect())
        }
    }
}
