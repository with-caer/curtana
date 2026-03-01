use std::io::Write;
use std::path::Path;

use curtana_infers::ModelRegistry;
use curtana_knows::manifest::Manifest;
use curtana_knows::router::Router;

use tracing::{debug, warn};

use crate::config::Config;

/// Conservative character budget for the sources block fed to the
/// synthesis model. Assumes ~2 chars/token (worst case for URL-heavy
/// content) and reserves headroom for the prompt template and response
/// within the 16 384-token context window.
const MAX_SOURCES_CHARS: usize = 24_000;

/// Loads the manifest and models, routes to relevant taxonomies,
/// searches them, synthesizes results into a single answer, and prints
/// compact source references.
pub async fn run(config: &Config, query: &str) {
    let data_dir = Path::new(config.data_dir());
    let manifest_path = data_dir.join("manifest.toml");
    let manifest = Manifest::load(&manifest_path).unwrap();

    if manifest.taxonomies.is_empty() {
        warn!("no taxonomies in manifest — run `curtana discover` first");
        return;
    }

    let registry = ModelRegistry::new().unwrap();
    let mut chat_model = registry
        .load_chat_model(&config.chat_model, "You are a helpful assistant.")
        .unwrap();
    let mut embed_model = registry
        .load_text_embedding_model(&config.embed_model)
        .unwrap();

    let router = Router::new(manifest, data_dir.to_path_buf());

    // Route to relevant taxonomies and search.
    let results = router.search(&mut embed_model, query, 15).await;

    if results.is_empty() {
        warn!("no results found.");
        return;
    }

    // Build context block from retrieved candidates, stopping when the
    // character budget is exhausted so the synthesis prompt fits in context.
    let mut sources_block = String::new();
    for (i, result) in results.iter().enumerate() {
        let text = format!("{}", result.artifact.contents);
        let content = if text.len() > 2000 {
            &text[..text.floor_char_boundary(2000)]
        } else {
            &text
        };
        let entry = format!(
            "[Source {}] (taxonomy: {}, author: {})\n{}\n\n",
            i + 1,
            result.taxonomy,
            result.artifact.author,
            content,
        );
        if sources_block.len() + entry.len() > MAX_SOURCES_CHARS {
            debug!(
                "context budget reached after {} of {} sources",
                i,
                results.len()
            );
            break;
        }
        sources_block.push_str(&entry);
    }

    let synthesis_prompt = format!(
        "Based on the following sources, answer this query: \"{query}\"\n\n\
         {sources_block}\
         Synthesize a clear, concise answer. Cite the sources you draw from."
    );

    // Stream synthesized answer to stdout.
    debug!("synthesizing answer...");
    let mut stdout = std::io::stdout();
    chat_model.infer(&synthesis_prompt, &mut stdout).unwrap();
    let _ = stdout.write_all(b"\n\n");

    // Print compact source references.
    println!("--- Sources ---");
    for (i, result) in results.iter().enumerate() {
        let text = format!("{}", result.artifact.contents);
        let title = text
            .lines()
            .next()
            .unwrap_or("")
            .trim_start_matches('#')
            .trim();
        let title = if title.is_empty() {
            result.artifact.id.as_str()
        } else {
            title
        };
        println!(
            "[{}] score: {:.4} | {} | {}",
            i + 1,
            result.score,
            result.taxonomy,
            title,
        );
    }
}
