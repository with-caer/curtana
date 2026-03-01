use std::path::Path;

use curtana_knows::manifest::{Manifest, TaxonomyEntry};
use curtana_reads::imap;
use tokio::sync::mpsc;

use crate::config::{Config, SourceConfig};
use crate::event::{CommandResult, DiscoverFolder, Event};

/// State retained between the two discovery phases.
pub(crate) struct DiscoverState {
    pub entries: Vec<DiscoverStateEntry>,
}

pub(crate) struct DiscoverStateEntry {
    pub folder_name: String,
    pub source_host: String,
    pub source_username: String,
}

/// Phase 1: connect to sources, discover folders, send listing to UI.
///
/// Returns state to hold until the user makes a selection.
pub(crate) async fn run(
    config: &Config,
    tx: &mpsc::UnboundedSender<Event>,
) -> Option<DiscoverState> {
    let data_dir = Path::new(config.data_dir());
    let manifest_path = data_dir.join("manifest.toml");

    let manifest = match Manifest::load(&manifest_path) {
        Ok(m) => m,
        Err(e) => {
            tx.send(Event::Error(format!("failed to load manifest: {e}")))
                .ok();
            return None;
        }
    };

    let mut entries = Vec::new();
    let mut folders = Vec::new();

    for source in &config.source {
        match source {
            SourceConfig::Imap(imap_config) => {
                let discovered = match imap::discover_folders(imap_config).await {
                    Ok(folders) => folders,
                    Err(e) => {
                        tx.send(Event::Error(format!("{e}"))).ok();
                        return None;
                    }
                };
                let selectable: Vec<_> =
                    discovered.into_iter().filter(|f| f.is_selectable).collect();

                for folder in selectable {
                    let already_tracked = manifest.taxonomies.values().any(|t| {
                        t.source_type == "imap"
                            && t.source_id == folder.name
                            && t.source_host == imap_config.host
                            && t.source_username == imap_config.username
                    });

                    let index = entries.len() + 1;
                    folders.push(DiscoverFolder {
                        index,
                        name: folder.name.clone(),
                        source_host: imap_config.host.clone(),
                        source_username: imap_config.username.clone(),
                        already_tracked,
                    });
                    entries.push(DiscoverStateEntry {
                        folder_name: folder.name,
                        source_host: imap_config.host.clone(),
                        source_username: imap_config.username.clone(),
                    });
                }
            }
        }
    }

    if entries.is_empty() {
        tx.send(Event::CommandDone(CommandResult::Message(
            "No selectable folders found.".into(),
        )))
        .ok();
        return None;
    }

    tx.send(Event::CommandDone(CommandResult::DiscoverFolders {
        folders,
    }))
    .ok();

    Some(DiscoverState { entries })
}

/// Phase 2: parse the user's selection and update the manifest.
pub(crate) fn select(
    config: &Config,
    input: &str,
    state: DiscoverState,
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

    let input = input.trim();

    let indices: Vec<usize> = if input.eq_ignore_ascii_case("all") {
        (0..state.entries.len()).collect()
    } else {
        input
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .map(|n| n.saturating_sub(1))
            .filter(|&i| i < state.entries.len())
            .collect()
    };

    if indices.is_empty() {
        tx.send(Event::CommandDone(CommandResult::Message(
            "No valid folders selected.".into(),
        )))
        .ok();
        return;
    }

    let mut added = Vec::new();
    let mut skipped = Vec::new();

    for i in indices {
        let entry = &state.entries[i];
        let taxonomy_name = format!(
            "imap-{}-{}",
            sanitize_source_label(&entry.source_username, &entry.source_host),
            sanitize_name(&entry.folder_name)
        );

        if manifest.taxonomies.contains_key(&taxonomy_name) {
            skipped.push(taxonomy_name);
            continue;
        }

        manifest.taxonomies.insert(
            taxonomy_name.clone(),
            TaxonomyEntry {
                name: taxonomy_name.clone(),
                description: String::new(),
                source_type: "imap".to_string(),
                source_id: entry.folder_name.clone(),
                source_host: entry.source_host.clone(),
                source_username: entry.source_username.clone(),
                last_ingested_at: None,
                description_updated_at: None,
            },
        );
        added.push(taxonomy_name);
    }

    if let Err(e) = manifest.save(&manifest_path) {
        tx.send(Event::Error(format!("failed to save manifest: {e}")))
            .ok();
        return;
    }

    let mut msg = String::new();
    if !added.is_empty() {
        msg.push_str(&format!(
            "Added {} taxonomies: {}",
            added.len(),
            added.join(", ")
        ));
    }
    if !skipped.is_empty() {
        if !msg.is_empty() {
            msg.push('\n');
        }
        msg.push_str(&format!(
            "Skipped {} (already tracked): {}",
            skipped.len(),
            skipped.join(", ")
        ));
    }

    tx.send(Event::CommandDone(CommandResult::Message(msg)))
        .ok();
}

/// Converts a folder name into a safe taxonomy name component.
/// Returns `"unnamed"` if the result would be empty after sanitization.
fn sanitize_name(name: &str) -> String {
    let sanitized = name
        .to_lowercase()
        .replace(|c: char| !c.is_alphanumeric() && c != '-', "-")
        .trim_matches('-')
        .to_string();
    if sanitized.is_empty() {
        "unnamed".to_string()
    } else {
        sanitized
    }
}

/// Builds a sanitized source label from username and host.
///
/// - `john` + `mail.example.com` → `john-mail-example-com`
/// - `me@caer.cc` + `127.0.0.1` → `me-caer-cc-via-127-0-0-1`
/// - `me@gmail.com` + `gmail.com` → `me-gmail-com`
fn sanitize_source_label(username: &str, host: &str) -> String {
    if let Some(domain) = username.rsplit_once('@').map(|(_, d)| d) {
        if domain == host {
            sanitize_name(username)
        } else {
            format!("{}-via-{}", sanitize_name(username), sanitize_name(host))
        }
    } else {
        format!("{}-{}", sanitize_name(username), sanitize_name(host))
    }
}
