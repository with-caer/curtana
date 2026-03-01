use std::io::{self, BufRead, Write};
use std::path::Path;

use curtana_knows::manifest::{Manifest, TaxonomyEntry};
use curtana_reads::imap;

use tracing::{info, warn};

use crate::config::{Config, SourceConfig};

/// Connects to each configured source, lists available folders,
/// prompts the user to select which to track, and writes manifest entries.
pub async fn run(config: &Config) {
    let data_dir = Path::new(config.data_dir());
    let manifest_path = data_dir.join("manifest.toml");
    let mut manifest = Manifest::load(&manifest_path).unwrap();

    for source in &config.source {
        match source {
            SourceConfig::Imap(imap_config) => {
                discover_imap(imap_config, &mut manifest).await;
            }
        }
    }

    manifest.save(&manifest_path).unwrap();
    info!("manifest saved to {}", manifest_path.display());
}

async fn discover_imap(config: &imap::ImapConfig, manifest: &mut Manifest) {
    info!("discovering folders on {}...", config.host);

    let folders = imap::discover_folders(config).await;

    let selectable: Vec<_> = folders.iter().filter(|f| f.is_selectable).collect();

    if selectable.is_empty() {
        warn!("no selectable folders found");
        return;
    }

    eprintln!("\navailable folders:");
    for (i, folder) in selectable.iter().enumerate() {
        let already = manifest
            .taxonomies
            .values()
            .any(|t| t.source_type == "imap" && t.source_id == folder.name);
        let marker = if already { " (already tracked)" } else { "" };
        eprintln!("  [{}] {}{}", i + 1, folder.name, marker);
    }

    eprint!("\nenter folder numbers to add (comma-separated), or 'all': ");
    io::stderr().flush().unwrap();

    let mut input = String::new();
    io::stdin().lock().read_line(&mut input).unwrap();
    let input = input.trim();

    let indices: Vec<usize> = if input.eq_ignore_ascii_case("all") {
        (0..selectable.len()).collect()
    } else {
        input
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .map(|n| n.saturating_sub(1))
            .filter(|&i| i < selectable.len())
            .collect()
    };

    for i in indices {
        let folder = &selectable[i];
        let taxonomy_name = format!("imap-{}", sanitize_name(&folder.name));

        if manifest.taxonomies.contains_key(&taxonomy_name) {
            warn!("skipping {taxonomy_name} (already tracked)");
            continue;
        }

        manifest.taxonomies.insert(
            taxonomy_name.clone(),
            TaxonomyEntry {
                name: taxonomy_name.clone(),
                description: String::new(),
                source_type: "imap".to_string(),
                source_id: folder.name.clone(),
                source_host: Some(config.host.clone()),
                last_ingested_at: None,
                description_updated_at: None,
            },
        );
        info!("added taxonomy: {taxonomy_name}");
    }
}

/// Converts a folder name into a safe taxonomy name component.
fn sanitize_name(name: &str) -> String {
    name.to_lowercase()
        .replace(|c: char| !c.is_alphanumeric() && c != '-', "-")
        .trim_matches('-')
        .to_string()
}
