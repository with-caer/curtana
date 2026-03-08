use std::time::{SystemTime, UNIX_EPOCH};

use curtana_knows::manifest::Manifest;
use curtana_knows::{Artifact, open_source_store, router};
use curtana_reads::imap;
use tokio::sync::mpsc;

use crate::config::{Config, SourceConfig};
use crate::event::{CommandResult, Event, Progress};

use super::Models;

/// Runs the full read pipeline, updating header status and progress bar.
pub(crate) async fn run(config: &Config, models: &mut Models, tx: &mpsc::UnboundedSender<Event>) {
    let data_dir = config.data_dir();
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
            "No taxonomies in manifest \u{2014} run /explore first.".into(),
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

        // Open source store (DB-per-source).
        let source_key = entry.source_key();
        let store = match open_source_store(&data_dir, &source_key).await {
            Ok(s) => s,
            Err(e) => {
                summary_lines.push(format!("- **{taxonomy_name}**: error opening store ({e})"));
                continue;
            }
        };

        // Migration safeguard: if the taxonomy had a cursor (incremental mode)
        // but the new source store has zero artifacts for this taxonomy, the
        // old per-taxonomy DB was orphaned. Reset cursor to force a full re-fetch.
        let mut effective_cursor = entry.cursor.clone();
        if effective_cursor.is_some() {
            match store.count(taxonomy_name).await {
                Ok(0) => {
                    eprintln!(
                        "warning: taxonomy {taxonomy_name} has cursor but 0 artifacts in \
                         source store {source_key} — resetting cursor for full re-fetch"
                    );
                    effective_cursor = None;
                    if let Some(e) = manifest.taxonomies.get_mut(taxonomy_name) {
                        e.cursor = None;
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("warning: failed to count artifacts for {taxonomy_name}: {e}");
                }
            }
        }

        // Fetch artifacts from source.
        let (artifacts, new_cursor) = match find_source(
            config,
            &entry.source_type,
            &entry.source_host,
            &entry.source_username,
        ) {
            Some(source) => {
                match fetch_artifacts(source, &entry.source_id, effective_cursor.as_deref()).await {
                    Ok(result) => result,
                    Err(e) => {
                        summary_lines.push(format!("- **{taxonomy_name}**: error ({e})"));
                        continue;
                    }
                }
            }
            None => {
                summary_lines.push(format!("- **{taxonomy_name}**: skipped (no source config)"));
                continue;
            }
        };

        let is_incremental = effective_cursor.is_some();
        let count = artifacts.len();

        // Upsert into source store with taxonomy discriminator.
        for artifact in &artifacts {
            if let Err(e) = store.upsert(taxonomy_name, artifact).await {
                summary_lines.push(format!("- **{taxonomy_name}**: upsert error ({e})"));
                continue;
            }
        }

        // Embed pending artifacts.
        let embed_label = format!("Embedding {taxonomy_name}");
        if let Err(e) = store
            .embed_pending(taxonomy_name, &mut models.embed, |current, total| {
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

        // Rebuild FTS index for BM25 search (non-fatal on failure).
        if let Err(e) = store.rebuild_fts_index().await {
            eprintln!("warning: failed to rebuild FTS index: {e}");
        }

        // Update ingestion timestamp and cursor.
        if let Some(entry) = manifest.taxonomies.get_mut(taxonomy_name) {
            entry.last_ingested_at = Some(now);
            if new_cursor.is_some() {
                entry.cursor = new_cursor;
            }
        }

        // Generate description if empty.
        if let Some(entry) = manifest.taxonomies.get_mut(taxonomy_name)
            && entry.description.is_empty()
        {
            set_status(
                tx,
                &format!("Generating description for {taxonomy_name}..."),
            );
            let description =
                router::generate_description(&store, taxonomy_name, &mut models.chat, 10).await;
            entry.description = description;
            entry.description_updated_at = Some(now);

            // Pre-embed the description for taxonomy affinity scoring.
            if !entry.description.is_empty() {
                match models.embed.embed(&[entry.description.as_str()]) {
                    Ok(mut embeddings) => entry.description_embedding = embeddings.pop(),
                    Err(e) => {
                        eprintln!(
                            "warning: failed to embed description for {taxonomy_name}: {e:?}"
                        );
                    }
                }
            }
        }

        let artifact_word = if count == 1 { "artifact" } else { "artifacts" };
        let mode = if is_incremental {
            "incremental"
        } else {
            "full"
        };
        summary_lines.push(format!(
            "- **{taxonomy_name}**: {count} new {artifact_word} ({mode})"
        ));
    }

    if let Err(e) = manifest.save(&manifest_path) {
        tx.send(Event::Error(format!("failed to save manifest: {e}")))
            .ok();
        return;
    }

    let summary = format!("## Read complete\n\n{}", summary_lines.join("\n"));
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

/// Fetches artifacts from a source folder, optionally resuming from a cursor.
async fn fetch_artifacts(
    source: &SourceConfig,
    folder_name: &str,
    cursor: Option<&str>,
) -> Result<(Vec<Artifact>, Option<String>), String> {
    match source {
        SourceConfig::Imap(base_config) => {
            let folder_config = imap::ImapConfig {
                host: base_config.host.clone(),
                port: base_config.port,
                username: base_config.username.clone(),
                password: base_config.password.clone(),
                mailbox: Some(folder_name.to_string()),
                starttls: base_config.starttls,
                accept_invalid_certs: base_config.accept_invalid_certs,
            };

            let result = imap::fetch_emails(&folder_config, cursor)
                .await
                .map_err(|e| e.to_string())?;

            let artifacts = result
                .items
                .into_iter()
                .map(|item| Artifact {
                    id: item.message_id.as_str().into(),
                    timestamp: item.timestamp as u64,
                    author: item.from.as_str().into(),
                    contents: item.to_markdown().into(),
                    embedding: Vec::new(),
                })
                .collect();

            Ok((artifacts, result.cursor))
        }
    }
}
