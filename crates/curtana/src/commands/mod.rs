pub mod agent;
pub mod explore;
pub mod read;
pub mod setup;
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
    CommandInfo {
        name: "/explore",
        description: "Explore source folders",
    },
    CommandInfo {
        name: "/help",
        description: "Show help",
    },
    CommandInfo {
        name: "/read",
        description: "Read from sources",
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
    Explore,
    Read,
    Status,
    Help,
    Quit,
}

/// Request sent from the main loop to the command thread.
pub enum CommandRequest {
    Query(String),
    Status,
    Explore,
    ExploreSelect(String),
    Read,
}

/// Holds loaded models for reuse across commands.
pub(crate) struct Models {
    chat: ChatModel,
    embed: TextEmbeddingModel,
    conversation_history: Vec<agent::ConversationEntry>,
}

/// Parses raw user input into a `Command`.
pub fn parse(input: &str) -> Command {
    let trimmed = input.trim();
    if trimmed.starts_with('/') {
        match trimmed.split_whitespace().next().unwrap_or("") {
            "/explore" => Command::Explore,
            "/read" => Command::Read,
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
            let mut explore_state: Option<explore::ExploreState> = None;

            while let Some(request) = cmd_rx.recv().await {
                match request {
                    CommandRequest::Query(q) => {
                        let m = match ensure_models(&mut models, &config) {
                            Ok(m) => m,
                            Err(e) => {
                                event_tx.send(Event::Error(e)).ok();
                                continue;
                            }
                        };
                        agent::run(&config, &q, m, &event_tx).await;
                    }
                    CommandRequest::Status => {
                        status::run(&config, &event_tx);
                    }
                    CommandRequest::Explore => {
                        explore_state = explore::run(&config, &event_tx).await;
                    }
                    CommandRequest::ExploreSelect(input) => {
                        if let Some(state) = explore_state.take() {
                            explore::select(&config, &input, state, &event_tx);
                        } else {
                            event_tx
                                .send(Event::Error("No pending explore session.".into()))
                                .ok();
                        }
                    }
                    CommandRequest::Read => {
                        let m = match ensure_models(&mut models, &config) {
                            Ok(m) => m,
                            Err(e) => {
                                event_tx.send(Event::Error(e)).ok();
                                continue;
                            }
                        };
                        read::run(&config, m, &event_tx).await;
                    }
                }
            }
        });
    });

    cmd_tx
}

fn ensure_models<'a>(
    models: &'a mut Option<Models>,
    config: &Config,
) -> Result<&'a mut Models, String> {
    if models.is_none() {
        *models = Some(load_models(config)?);
    }
    Ok(models.as_mut().unwrap())
}

fn load_models(config: &Config) -> Result<Models, String> {
    let registry =
        ModelRegistry::new().map_err(|e| format!("failed to init model registry: {e:?}"))?;

    let chat_path = config.chat_model_path();
    let chat_path_str = chat_path.to_str().ok_or_else(|| {
        format!(
            "chat model path is not valid UTF-8: {}",
            chat_path.display()
        )
    })?;
    let chat = registry
        .load_chat_model(chat_path_str, "You are a helpful assistant.")
        .map_err(|e| format!("failed to load chat model '{}': {e:?}", chat_path.display()))?;

    let embed_path = config.embed_model_path();
    let embed_path_str = embed_path.to_str().ok_or_else(|| {
        format!(
            "embed model path is not valid UTF-8: {}",
            embed_path.display()
        )
    })?;
    let embed = registry
        .load_text_embedding_model(embed_path_str)
        .map_err(|e| {
            format!(
                "failed to load embedding model '{}': {e:?}",
                embed_path.display()
            )
        })?;

    Ok(Models {
        chat,
        embed,
        conversation_history: Vec::new(),
    })
}
