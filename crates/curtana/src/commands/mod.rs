pub mod agent;
pub mod discover;
pub mod ingest;
pub mod query;
pub mod status;

use std::io;
use std::sync::Arc;

use tokio::sync::mpsc;

use crate::config::Config;
use crate::event::Event;

use curtana_infers::{ChatModel, ModelRegistry, TextEmbeddingModel};

/// Metadata for a slash command (used by tab completion).
pub struct CommandInfo {
    pub name: &'static str,
    pub description: &'static str,
}

/// All slash commands, sorted alphabetically for the completion popup.
pub const COMMANDS: &[CommandInfo] = &[
    CommandInfo {
        name: "/discover",
        description: "Discover source folders",
    },
    CommandInfo {
        name: "/help",
        description: "Show help",
    },
    CommandInfo {
        name: "/ingest",
        description: "Ingest artifacts",
    },
    CommandInfo {
        name: "/quit",
        description: "Exit curtana",
    },
    CommandInfo {
        name: "/status",
        description: "Show tracked taxonomies",
    },
];

/// Parsed user command.
pub enum Command {
    Query(String),
    Discover,
    Ingest,
    Status,
    Help,
    Quit,
}

/// Request sent from the main loop to the command thread.
pub enum CommandRequest {
    Query(String),
    Status,
    Discover,
    DiscoverSelect(String),
    Ingest,
}

/// Holds loaded models for reuse across commands.
pub(crate) struct Models {
    chat: ChatModel,
    embed: TextEmbeddingModel,
    conversation_history: Vec<ConversationEntry>,
}

/// Maximum number of prior conversation entries to retain.
pub(crate) const MAX_HISTORY_ENTRIES: usize = 2;

/// Maximum number of tool-use turns before forcing synthesis.
pub(crate) const MAX_TURNS: usize = 5;

/// Maximum accumulated bytes (across all messages) before stopping
/// gathering early. At ~4 chars/token, 40K chars ≈ 10K tokens, leaving
/// 6K tokens of headroom in the 16K context window for the model's
/// response and chat-template overhead.
pub(crate) const MAX_GATHERING_BYTES: usize = 40_000;

/// Conservative character budget for the synthesis sources block.
pub(crate) const MAX_SOURCES_CHARS: usize = 24_000;

pub(crate) const AGENT_SYSTEM_PROMPT: &str = "\
You are a research assistant with access to a knowledge base organized into taxonomies.

Available tools:
- list_taxonomies() — List all available taxonomies with descriptions and artifact counts
- count({\"taxonomy\": \"name\"}) — Count artifacts in a taxonomy
- search({\"query\": \"text\", \"taxonomy\": \"name\", \"top_k\": 10}) — Semantic search for relevant artifacts. 'taxonomy' and 'top_k' are optional.
- browse({\"taxonomy\": \"name\", \"offset\": 0, \"limit\": 5, \"order\": \"desc\"}) — Browse artifacts chronologically. 'offset', 'limit', 'order' are optional.
- filter({\"taxonomy\": \"name\", \"author\": \"name\", \"after\": 1234567890, \"before\": 1234567890, \"limit\": 10}) — Filter artifacts by metadata. All fields except 'taxonomy' are optional.

To call a tool, write exactly: <tool>tool_name({\"arg\": \"value\"})</tool>
For tools with no arguments: <tool>list_taxonomies()</tool>
When you have gathered enough information, write: <curtana:done/>

Strategy:
1. Start by listing taxonomies if you are unsure which to query.
2. Use search for semantic/topic queries, browse for chronological queries, and filter for metadata queries.
3. Gather only what you need, then write <curtana:done/>.";

/// Maximum total characters for the prior-conversation context block.
const MAX_HISTORY_CHARS: usize = 12_000;

/// Maximum characters of source content stored per history entry.
const MAX_SOURCES_PER_ENTRY: usize = 6_000;

/// A single conversation turn stored for follow-up context.
pub(crate) struct ConversationEntry {
    pub query: String,
    pub response: String,
    /// Condensed source content from this turn, so follow-up questions
    /// can reference the actual artifacts without re-searching.
    pub sources: String,
}

/// Formats prior conversation history into a context block for prompts.
/// Includes both the synthesized answer and key source content so the
/// model can answer follow-ups without re-searching.
pub(crate) fn format_conversation_context(history: &[ConversationEntry]) -> String {
    if history.is_empty() {
        return String::new();
    }

    let mut block = String::from("Previous conversation:\n");
    let mut budget = MAX_HISTORY_CHARS - block.len();

    for entry in history {
        let entry_header = format!("User: {}\nAssistant: ", entry.query);
        let header_len = entry_header.len() + 2; // +2 for trailing newlines

        if budget < header_len + 100 {
            break;
        }

        block.push_str(&entry_header);

        // Include the response, truncated if needed.
        let max_response_len = budget - header_len;
        if entry.response.len() > max_response_len {
            block.push_str(&entry.response[..max_response_len]);
            block.push_str("...");
        } else {
            block.push_str(&entry.response);
        }
        block.push_str("\n\n");

        budget = budget.saturating_sub(header_len + entry.response.len().min(max_response_len));

        // Append condensed source content if available and budget allows.
        if !entry.sources.is_empty() && budget > 200 {
            let sources_header = "Key sources from that answer:\n";
            block.push_str(sources_header);
            budget = budget.saturating_sub(sources_header.len());

            let max_sources_len = budget.min(entry.sources.len());
            if entry.sources.len() > max_sources_len {
                block.push_str(&entry.sources[..max_sources_len]);
                block.push_str("...\n\n");
            } else {
                block.push_str(&entry.sources);
                block.push_str("\n\n");
            }
            budget = budget.saturating_sub(max_sources_len + 2);
        }
    }

    block
}

/// Builds a condensed sources summary for conversation history storage.
/// Takes the raw sources block and truncates it to fit the per-entry budget.
pub(crate) fn condense_sources(sources_block: &str) -> String {
    if sources_block.len() <= MAX_SOURCES_PER_ENTRY {
        sources_block.to_string()
    } else {
        let mut truncated =
            curtana_knows::truncate_text(sources_block, MAX_SOURCES_PER_ENTRY).to_string();
        truncated.push_str("...");
        truncated
    }
}

/// An `io::Write` adapter that forwards writes to an inner writer while
/// also appending the raw bytes to a buffer for later capture.
pub(crate) struct TeeWriter<'a, W> {
    inner: W,
    buffer: &'a mut Vec<u8>,
}

impl<'a, W> TeeWriter<'a, W> {
    pub(crate) fn new(inner: W, buffer: &'a mut Vec<u8>) -> Self {
        Self { inner, buffer }
    }
}

impl<W: io::Write> io::Write for TeeWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.buffer.extend_from_slice(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Parses raw user input into a `Command`.
pub fn parse(input: &str) -> Command {
    let trimmed = input.trim();
    if trimmed.starts_with('/') {
        match trimmed.split_whitespace().next().unwrap_or("") {
            "/discover" => Command::Discover,
            "/ingest" => Command::Ingest,
            "/status" => Command::Status,
            "/help" => Command::Help,
            "/quit" | "/exit" => Command::Quit,
            _ => Command::Help,
        }
    } else {
        Command::Query(trimmed.to_string())
    }
}

/// Spawns the command processing thread.
///
/// Returns a sender for submitting `CommandRequest`s. The thread owns
/// the models and a dedicated single-threaded tokio runtime so that
/// non-Send model types never cross thread boundaries.
pub fn spawn_command_thread(
    config: Arc<Config>,
    event_tx: mpsc::UnboundedSender<Event>,
) -> mpsc::UnboundedSender<CommandRequest> {
    let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<CommandRequest>();

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build command runtime");

        rt.block_on(async {
            let mut models: Option<Models> = None;
            let mut discover_state: Option<discover::DiscoverState> = None;

            while let Some(request) = cmd_rx.recv().await {
                match request {
                    CommandRequest::Query(q) => {
                        if models.is_none() {
                            match load_models(&config) {
                                Ok(m) => models = Some(m),
                                Err(e) => {
                                    event_tx.send(Event::Error(e)).ok();
                                    continue;
                                }
                            }
                        }
                        let m = models.as_mut().unwrap();
                        if config.use_agent_mode() {
                            agent::run(&config, &q, m, &event_tx).await;
                        } else {
                            query::run(&config, &q, m, &event_tx).await;
                        }
                    }
                    CommandRequest::Status => {
                        status::run(&config, &event_tx);
                    }
                    CommandRequest::Discover => {
                        discover_state = discover::run(&config, &event_tx).await;
                    }
                    CommandRequest::DiscoverSelect(input) => {
                        if let Some(state) = discover_state.take() {
                            discover::select(&config, &input, state, &event_tx);
                        } else {
                            event_tx
                                .send(Event::Error("No pending discovery session.".into()))
                                .ok();
                        }
                    }
                    CommandRequest::Ingest => {
                        if models.is_none() {
                            match load_models(&config) {
                                Ok(m) => models = Some(m),
                                Err(e) => {
                                    event_tx.send(Event::Error(e)).ok();
                                    continue;
                                }
                            }
                        }
                        let m = models.as_mut().unwrap();
                        ingest::run(&config, m, &event_tx).await;
                    }
                }
            }
        });
    });

    cmd_tx
}

fn load_models(config: &Config) -> Result<Models, String> {
    let registry =
        ModelRegistry::new().map_err(|e| format!("failed to init model registry: {e:?}"))?;
    let chat = registry
        .load_chat_model(&config.chat_model, "You are a helpful assistant.")
        .map_err(|e| format!("failed to load chat model: {e:?}"))?;
    let embed = registry
        .load_text_embedding_model(&config.embed_model)
        .map_err(|e| format!("failed to load embedding model: {e:?}"))?;
    Ok(Models {
        chat,
        embed,
        conversation_history: Vec::new(),
    })
}
