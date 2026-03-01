use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use curtana_infers::{ChatModel, ModelRegistry, TextEmbeddingModel};
use curtana_knows::manifest::Manifest;
use curtana_knows::router::{Router, ScoredArtifact};
use curtana_knows::tools::{self, ToolExecutor, ToolResult};
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
            store.upsert(artifact).await;
        }

        info!("  Embedding...");
        store.embed_pending(&mut models.embed, |_, _| {}).await;

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

/// Agent system prompt for the CLI gathering phase.
const AGENT_SYSTEM_PROMPT: &str = "\
You are a research assistant with access to a knowledge base organized into taxonomies.

Available tools:
- list_taxonomies() — List all available taxonomies with descriptions and artifact counts
- count({\"taxonomy\": \"name\"}) — Count artifacts in a taxonomy
- search({\"query\": \"text\", \"taxonomy\": \"name\", \"top_k\": 10}) — Semantic search for relevant artifacts. 'taxonomy' and 'top_k' are optional.
- browse({\"taxonomy\": \"name\", \"offset\": 0, \"limit\": 5, \"order\": \"desc\"}) — Browse artifacts chronologically. 'offset', 'limit', 'order' are optional.
- filter({\"taxonomy\": \"name\", \"author\": \"name\", \"after\": 1234567890, \"before\": 1234567890, \"limit\": 10}) — Filter artifacts by metadata. All fields except 'taxonomy' are optional.

To call a tool, write exactly: <tool>tool_name({\"arg\": \"value\"})</tool>
For tools with no arguments: <tool>list_taxonomies()</tool>
When you have gathered enough information, write: <done/>

Strategy:
1. Start by listing taxonomies if you are unsure which to query.
2. Use search for semantic/topic queries, browse for chronological queries, and filter for metadata queries.
3. Gather only what you need, then write <done/>.";

/// Maximum gathering turns for the CLI agent loop.
const MAX_TURNS: usize = 5;

/// Maximum accumulated bytes before stopping gathering.
const MAX_GATHERING_BYTES: usize = 40_000;

/// Maximum characters for the synthesis sources block.
const MAX_SOURCES_CHARS_CAP: usize = 24_000;

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

    let data_dir = Path::new(config.data_dir());

    if config.use_agent_mode() {
        query_agent(data_dir, query, &manifest, &mut models).await;
    } else {
        query_simple(data_dir, query, &manifest, &mut models).await;
    }
}

/// Agent-mode query: gathering loop then synthesis.
async fn query_agent(data_dir: &Path, query: &str, manifest: &Manifest, models: &mut Models) {
    let executor = ToolExecutor::new(manifest.clone(), data_dir.to_path_buf());

    // === Gathering Phase ===
    models
        .chat
        .replace_system_prompt(AGENT_SYSTEM_PROMPT.to_string())
        .expect("failed to set agent prompt");

    let opening_prompt = format!(
        "The user asked: \"{query}\". Use tools to gather information, then write <done/>."
    );

    let mut gathered_sources: Vec<ScoredArtifact> = Vec::new();
    let mut gathered_context: Vec<String> = Vec::new();
    let mut prompt = opening_prompt;
    let mut accumulated_bytes: usize = AGENT_SYSTEM_PROMPT.len();

    for turn in 0..MAX_TURNS {
        accumulated_bytes += prompt.len();
        if accumulated_bytes > MAX_GATHERING_BYTES {
            eprintln!("[budget reached, synthesizing]");
            break;
        }

        let output = match models.chat.infer_with_history_to_string(&prompt) {
            Ok(o) => o,
            Err(_) => {
                // Context overflow — proceed to synthesis with what we have.
                break;
            }
        };

        accumulated_bytes += output.len();

        match tools::parse_tool_response(&output) {
            tools::ParseResult::ToolCall(call) => {
                eprintln!("[tool: {}]", call.name);
                let ToolResult { text, sources } = executor.execute(&mut models.embed, &call).await;
                gathered_context.push(text.clone());
                gathered_sources.extend(sources);
                prompt = format!("Tool result:\n{text}");
            }
            tools::ParseResult::Done => break,
            tools::ParseResult::Answer(answer) => {
                if turn == 0 && gathered_context.is_empty() {
                    gathered_context.push(answer);
                }
                break;
            }
        }
    }

    // === Synthesis Phase ===
    models.chat.clear_history();
    models
        .chat
        .replace_system_prompt("You are a helpful assistant.".to_string())
        .expect("failed to restore system prompt");

    // If no tools were called, fall back to Router::search.
    let final_sources: Vec<ScoredArtifact>;
    let sources_block;

    if gathered_sources.is_empty() {
        let router = Router::new(manifest.clone(), data_dir.to_path_buf());
        let results = router.search(&mut models.embed, query, 15).await;
        if results.is_empty() {
            eprintln!("No results found.");
            return;
        }
        let mut block = String::new();
        for ctx in &gathered_context {
            if block.len() + ctx.len() + 2 > MAX_SOURCES_CHARS_CAP {
                break;
            }
            block.push_str(ctx);
            block.push_str("\n\n");
        }
        block.push_str(&build_sources_block(&results));
        sources_block = block;
        final_sources = results;
    } else {
        let mut block = String::new();
        for ctx in &gathered_context {
            if block.len() + ctx.len() + 2 > MAX_SOURCES_CHARS_CAP {
                break;
            }
            if !block.is_empty() {
                block.push_str("\n\n");
            }
            block.push_str(ctx);
        }
        sources_block = block;
        final_sources = gathered_sources;
    }

    let synthesis_prompt = format!(
        "Based on the following sources, answer this query: \"{query}\"\n\n\
         {sources_block}\
         Synthesize a clear, concise answer. Cite the sources you draw from."
    );

    let mut stdout = std::io::stdout();
    if let Err(e) = models.chat.infer(&synthesis_prompt, &mut stdout) {
        eprintln!("Inference error: {e:?}");
        return;
    }
    stdout.write_all(b"\n").ok();

    print_sources(&final_sources);
}

/// Simple (non-agent) query: single-pass search and synthesis.
async fn query_simple(data_dir: &Path, query: &str, manifest: &Manifest, models: &mut Models) {
    let router = Router::new(manifest.clone(), data_dir.to_path_buf());
    let results = router.search(&mut models.embed, query, 15).await;

    if results.is_empty() {
        eprintln!("No results found.");
        return;
    }

    let sources_block = build_sources_block(&results);

    let synthesis_prompt = format!(
        "Based on the following sources, answer this query: \"{query}\"\n\n\
         {sources_block}\
         Synthesize a clear, concise answer. Cite the sources you draw from."
    );

    let mut stdout = std::io::stdout();
    if let Err(e) = models.chat.infer(&synthesis_prompt, &mut stdout) {
        eprintln!("Inference error: {e:?}");
        return;
    }
    stdout.write_all(b"\n").ok();

    print_sources(&results);
}

/// Builds a formatted sources block from scored artifacts.
fn build_sources_block(results: &[ScoredArtifact]) -> String {
    const MAX_SOURCES_CHARS: usize = 24_000;
    let mut block = String::new();
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
        if block.len() + entry.len() > MAX_SOURCES_CHARS {
            break;
        }
        block.push_str(&entry);
    }
    block
}

/// Prints source references to stderr.
fn print_sources(results: &[ScoredArtifact]) {
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
