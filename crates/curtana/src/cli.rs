use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use curtana_infers::{ChatModel, ModelRegistry, TextEmbeddingModel};
use curtana_knows::manifest::Manifest;
use curtana_knows::router::Router;
use curtana_knows::{Artifact, open_taxonomy_store};
use curtana_reads::ToMarkdown;
use curtana_reads::imap;
use tracing::info;

use crate::config::{Config, SourceConfig};

struct Models {
    chat: ChatModel,
    embed: TextEmbeddingModel,
}

fn load_models(config: &Config) -> Models {
    let registry = ModelRegistry::new().expect("failed to init model registry");
    let chat = registry
        .load_chat_model(&config.chat_model, "You are a helpful assistant.")
        .expect("failed to load chat model");
    let embed = registry
        .load_text_embedding_model(&config.embed_model)
        .expect("failed to load embedding model");
    Models { chat, embed }
}

/// Runs the full ingest pipeline headlessly, logging progress via tracing.
pub async fn ingest(config: &Config) {
    let data_dir = Path::new(config.data_dir());
    let manifest_path = data_dir.join("manifest.toml");

    let mut manifest = Manifest::load(&manifest_path).expect("failed to load manifest");

    if manifest.taxonomies.is_empty() {
        eprintln!("No taxonomies in manifest — run `curtana discover` first.");
        return;
    }

    let mut models = load_models(config);

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let taxonomy_names: Vec<String> = manifest.taxonomies.keys().cloned().collect();

    for taxonomy_name in &taxonomy_names {
        let entry = manifest.taxonomies.get(taxonomy_name).unwrap().clone();

        info!("Ingesting {taxonomy_name}...");

        let artifacts = match find_source(config, &entry.source_type) {
            Some(source) => fetch_artifacts(source, &entry.source_id).await,
            None => {
                info!("  No matching source config, skipping.");
                continue;
            }
        };

        info!("  Fetched {} artifacts.", artifacts.len());

        let store = open_taxonomy_store(data_dir, taxonomy_name).await;
        for artifact in &artifacts {
            store.upsert(artifact.id.clone(), artifact).await;
        }

        info!("  Embedding...");
        store.embed_pending(&mut models.embed).await;

        if let Some(entry) = manifest.taxonomies.get_mut(taxonomy_name) {
            entry.last_ingested_at = Some(now);
        }

        if let Some(entry) = manifest.taxonomies.get_mut(taxonomy_name)
            && entry.description.is_empty()
        {
            info!("  Generating description...");
            let description =
                curtana_knows::router::generate_description(&store, &mut models.chat, 10).await;
            entry.description = description;
            entry.description_updated_at = Some(now);
        }

        info!("  Done.");
    }

    manifest
        .save(&manifest_path)
        .expect("failed to save manifest");

    info!("Ingestion complete.");
}

/// Runs a query headlessly, printing the answer to stdout and sources to stderr.
pub async fn query(config: &Config, query: &str) {
    let data_dir = Path::new(config.data_dir());
    let manifest_path = data_dir.join("manifest.toml");

    let manifest = Manifest::load(&manifest_path).expect("failed to load manifest");

    if manifest.taxonomies.is_empty() {
        eprintln!("No taxonomies found — run `curtana discover` first.");
        return;
    }

    let mut models = load_models(config);

    let router = Router::new(manifest, data_dir.to_path_buf());
    let results = router.search(&mut models.embed, query, 15).await;

    if results.is_empty() {
        eprintln!("No results found.");
        return;
    }

    // Build context block from retrieved candidates.
    const MAX_SOURCES_CHARS: usize = 24_000;
    let mut sources_block = String::new();
    for (i, result) in results.iter().enumerate() {
        let text = format!("{}", result.artifact.contents);
        let content = curtana_knows::truncate_text(&text, 2000);
        let entry = format!(
            "[Source {}] (taxonomy: {}, author: {})\n{}\n\n",
            i + 1,
            result.taxonomy,
            result.artifact.author,
            content,
        );
        if sources_block.len() + entry.len() > MAX_SOURCES_CHARS {
            break;
        }
        sources_block.push_str(&entry);
    }

    let synthesis_prompt = format!(
        "Based on the following sources, answer this query: \"{query}\"\n\n\
         {sources_block}\
         Synthesize a clear, concise answer. Cite the sources you draw from."
    );

    // Stream the synthesized answer to stdout.
    let mut stdout = std::io::stdout();
    if let Err(e) = models.chat.infer(&synthesis_prompt, &mut stdout) {
        eprintln!("Inference error: {e:?}");
        return;
    }
    stdout.write_all(b"\n").ok();

    // Print sources to stderr.
    for (i, result) in results.iter().enumerate() {
        let text = format!("{}", result.artifact.contents);
        let title = text
            .lines()
            .next()
            .unwrap_or("")
            .trim_start_matches('#')
            .trim();
        let title = if title.is_empty() {
            format!("{}", result.artifact.id)
        } else {
            title.to_string()
        };
        eprintln!(
            "[{}] {:.4} | {} | {}",
            i + 1,
            result.score,
            result.taxonomy,
            title,
        );
    }
}

fn find_source<'a>(config: &'a Config, source_type: &str) -> Option<&'a SourceConfig> {
    config
        .source
        .iter()
        .find(|s| matches!((source_type, s), ("imap", SourceConfig::Imap(_))))
}

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
