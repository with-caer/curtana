use std::path::Path;

use curtana_knows::manifest::Manifest;
use curtana_knows::router::{Router, ScoredArtifact};
use curtana_knows::tools::{self, ToolExecutor, ToolResult};
use tokio::sync::mpsc;

use crate::event::{ChannelWriter, CommandResult, Event};

use super::{
    AGENT_SYSTEM_PROMPT, ConversationEntry, MAX_GATHERING_BYTES, MAX_HISTORY_ENTRIES,
    MAX_SOURCES_CHARS, MAX_TURNS, Models, TeeWriter, condense_sources, format_conversation_context,
};

/// Runs the agent query pipeline: gathering phase (tool-use loop),
/// then synthesis phase (streaming).
pub(crate) async fn run(
    config: &crate::config::Config,
    query: &str,
    models: &mut Models,
    tx: &mpsc::UnboundedSender<Event>,
) {
    let data_dir = Path::new(config.data_dir());
    let manifest_path = data_dir.join("manifest.toml");

    let manifest = match Manifest::load(&manifest_path) {
        Ok(m) => m,
        Err(e) => {
            tx.send(Event::Error(format!("failed to load manifest: {e}")))
                .ok();
            return;
        }
    };

    if manifest.taxonomies.is_empty() {
        tx.send(Event::Error(
            "no taxonomies found \u{2014} run /discover first".into(),
        ))
        .ok();
        return;
    }

    let executor = ToolExecutor::new(manifest.clone(), data_dir.to_path_buf());

    // === Gathering Phase ===
    tx.send(Event::StatusText("Thinking...".into())).ok();

    // Set the agent system prompt.
    if let Err(e) = models
        .chat
        .replace_system_prompt(AGENT_SYSTEM_PROMPT.to_string())
    {
        tx.send(Event::Error(format!("failed to set agent prompt: {e:?}")))
            .ok();
        return;
    }

    let conv_context = format_conversation_context(&models.conversation_history);
    let opening_prompt = if conv_context.is_empty() {
        format!(
            "The user asked:\n<user-query>{query}</user-query>\n\nUse tools to gather information, then write <curtana:done/>."
        )
    } else {
        format!(
            "You have already answered a previous question from this user. \
             Here is the full prior conversation, including the source artifacts you used:\n\n\
             {conv_context}\n\
             The user now asks:\n<user-query>{query}</user-query>\n\n\
             IMPORTANT: Review the prior conversation and sources above carefully. \
             If the answer is already contained in those sources, write <curtana:done/> immediately \
             without calling any tools. Only use tools if you need information that is NOT \
             already present above."
        )
    };

    let mut gathered_sources: Vec<ScoredArtifact> = Vec::new();
    let mut gathered_context: Vec<String> = Vec::new();
    let mut prompt = opening_prompt;
    let mut accumulated_bytes: usize = AGENT_SYSTEM_PROMPT.len();

    for turn in 0..MAX_TURNS {
        // Context budget guard based on actual content bytes.
        accumulated_bytes += prompt.len();
        if accumulated_bytes > MAX_GATHERING_BYTES {
            tx.send(Event::StatusText(
                "Context budget reached, synthesizing...".into(),
            ))
            .ok();
            break;
        }

        tx.send(Event::StatusText("Thinking...".into())).ok();
        tx.send(Event::ActivityLine(format!(
            "[{}/{}] Thinking...",
            turn + 1,
            MAX_TURNS
        )))
        .ok();

        let output = match models.chat.infer_with_history_to_string(&prompt) {
            Ok(o) => o,
            Err(_) => {
                // Context overflow or model error — proceed to synthesis
                // with whatever was gathered so far.
                break;
            }
        };

        accumulated_bytes += output.len();

        match tools::parse_tool_response(&output) {
            tools::ParseResult::ToolCall(call) => {
                let call_summary = summarize_tool_call(&call);
                tx.send(Event::ActivityLine(format!(
                    "[{}/{}] {call_summary}",
                    turn + 1,
                    MAX_TURNS
                )))
                .ok();
                tx.send(Event::StatusText(format!("Calling {}...", call.name)))
                    .ok();

                let ToolResult { text, sources } = executor.execute(&mut models.embed, &call).await;

                let result_summary = summarize_result(&text);
                tx.send(Event::ActivityLine(format!(
                    "[{}/{}] {call_summary} → {result_summary}",
                    turn + 1,
                    MAX_TURNS
                )))
                .ok();

                gathered_context.push(text.clone());
                gathered_sources.extend(sources);

                // Inject tool result as the next user prompt.
                prompt = format!("Tool result:\n{text}");
            }
            tools::ParseResult::Done => {
                tx.send(Event::ActivityLine("Synthesizing...".into())).ok();
                tx.send(Event::StatusText("Synthesizing...".into())).ok();
                break;
            }
            tools::ParseResult::Answer(answer) => {
                // Model answered directly without tools on the first turn.
                if turn == 0 && gathered_context.is_empty() {
                    // The model skipped tools — this is the direct answer.
                    // We'll store it and fall through to the fallback path,
                    // which will do a Router::search for proper sourcing.
                    gathered_context.push(answer);
                }
                break;
            }
        }
    }

    // === Synthesis Phase ===
    models.chat.clear_history();
    if let Err(e) = models
        .chat
        .replace_system_prompt("You are a helpful assistant.".to_string())
    {
        tx.send(Event::Error(format!(
            "failed to restore system prompt: {e:?}"
        )))
        .ok();
        return;
    }

    // If no tools were called, fall back to Router::search.
    let sources_block;

    if gathered_sources.is_empty() && gathered_context.is_empty() {
        // Pure fallback: model never produced anything useful.
        let router = Router::new(manifest, data_dir.to_path_buf());
        let results = match router.search(&mut models.embed, query, 15).await {
            Ok(r) => r,
            Err(e) => {
                tx.send(Event::Error(format!("search failed: {e}"))).ok();
                return;
            }
        };

        if results.is_empty() {
            tx.send(Event::Error("no results found.".into())).ok();
            return;
        }

        sources_block = build_sources_block(&results);
    } else if gathered_sources.is_empty() {
        // Model answered directly or context was gathered but no ScoredArtifacts.
        // Do a Router::search for source attribution.
        let router = Router::new(manifest, data_dir.to_path_buf());
        let results = match router.search(&mut models.embed, query, 15).await {
            Ok(r) => r,
            Err(e) => {
                tx.send(Event::Error(format!("search failed: {e}"))).ok();
                return;
            }
        };

        // Use gathered context + search results.
        let mut block = String::new();
        for ctx in &gathered_context {
            block.push_str(ctx);
            block.push_str("\n\n");
        }
        block.push_str(&build_sources_block(&results));
        sources_block = block;
    } else {
        // Normal path: use gathered context, capped to fit in context.
        let mut block = String::new();
        for ctx in &gathered_context {
            if block.len() + ctx.len() + 2 > MAX_SOURCES_CHARS {
                break;
            }
            if !block.is_empty() {
                block.push_str("\n\n");
            }
            block.push_str(ctx);
        }
        sources_block = block;
    }

    let synthesis_prompt = if conv_context.is_empty() {
        format!(
            "Based on the following sources, answer this query:\n\
             <user-query>{query}</user-query>\n\n\
             {sources_block}\
             Synthesize a clear, concise answer. Cite the sources you draw from."
        )
    } else {
        format!(
            "{conv_context}\
             Based on the following sources, answer the follow-up query:\n\
             <user-query>{query}</user-query>\n\n\
             {sources_block}\
             Synthesize a clear, concise answer. Cite the sources you draw from."
        )
    };

    // Stream the synthesized answer, capturing a copy for conversation history.
    let mut response_buf: Vec<u8> = Vec::new();
    {
        let channel_writer = ChannelWriter::new(tx.clone());
        let mut writer = TeeWriter::new(channel_writer, &mut response_buf);
        if let Err(e) = models.chat.infer(&synthesis_prompt, &mut writer) {
            tx.send(Event::Error(format!("inference error: {e:?}")))
                .ok();
            return;
        }
    }

    // Store conversation history for follow-up questions.
    let response_text = String::from_utf8_lossy(&response_buf).into_owned();
    let sources_summary = condense_sources(&sources_block);
    models.conversation_history.push(ConversationEntry {
        query: query.to_string(),
        response: response_text,
        sources: sources_summary,
    });
    if models.conversation_history.len() > MAX_HISTORY_ENTRIES {
        models
            .conversation_history
            .drain(..models.conversation_history.len() - MAX_HISTORY_ENTRIES);
    }

    tx.send(Event::CommandDone(CommandResult::QueryDone)).ok();
}

/// Produces a short summary of a tool call, e.g. `browse(emails, limit=5)`.
fn summarize_tool_call(call: &tools::ToolCall) -> String {
    let args = &call.args;
    let mut parts: Vec<String> = Vec::new();

    // Pull out the most informative args in display order.
    // Taxonomy and author are shown in full; query is truncated.
    for key in &["taxonomy", "author"] {
        if let Some(v) = args.get(*key).and_then(|v| v.as_str()) {
            parts.push(v.to_string());
        }
    }
    if let Some(v) = args.get("query").and_then(|v| v.as_str()) {
        if v.len() > 40 {
            parts.push(format!("{}...", curtana_knows::truncate_text(v, 37)));
        } else {
            parts.push(v.to_string());
        }
    }
    for key in &["limit", "top_k", "offset"] {
        if let Some(v) = args.get(*key).and_then(|v| v.as_u64()) {
            parts.push(format!("{key}={v}"));
        }
    }

    if parts.is_empty() {
        format!("{}()", call.name)
    } else {
        format!("{}({})", call.name, parts.join(", "))
    }
}

/// Produces a brief summary of a tool result string.
fn summarize_result(text: &str) -> String {
    let line_count = text.lines().count();
    if line_count > 3 {
        format!("{line_count} lines")
    } else if text.len() <= 60 {
        text.replace('\n', " ")
    } else {
        format!(
            "{}...",
            curtana_knows::truncate_text(text, 57).replace('\n', " ")
        )
    }
}

/// Builds a formatted sources block from scored artifacts, respecting
/// the character budget.
fn build_sources_block(results: &[ScoredArtifact]) -> String {
    let mut block = String::new();
    for result in results {
        let text = format!("{}", result.artifact.contents);
        let content = curtana_knows::truncate_text(&text, 2000);
        let entry = format!(
            "<source taxonomy=\"{}\" author=\"{}\">\n{}\n</source>\n\n",
            result.taxonomy, result.artifact.author, content,
        );
        if block.len() + entry.len() > MAX_SOURCES_CHARS {
            break;
        }
        block.push_str(&entry);
    }
    block
}
