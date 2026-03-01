pub mod discover;
pub mod ingest;
pub mod query;
pub mod status;

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
    CommandInfo { name: "/discover", description: "Discover source folders" },
    CommandInfo { name: "/help",     description: "Show help" },
    CommandInfo { name: "/ingest",   description: "Ingest artifacts" },
    CommandInfo { name: "/quit",     description: "Exit curtana" },
    CommandInfo { name: "/status",   description: "Show tracked taxonomies" },
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
                        query::run(&config, &q, m, &event_tx).await;
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
                                .send(Event::Error(
                                    "No pending discovery session.".into(),
                                ))
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
    Ok(Models { chat, embed })
}
