use std::path::Path;

use curtana_knows::manifest::Manifest;
use curtana_knows::router::Router;
use tokio::sync::mpsc;

use crate::event::{ChannelWriter, CommandResult, Event};

use super::{
    ConversationEntry, MAX_HISTORY_ENTRIES, Models, TeeWriter, condense_sources,
    format_conversation_context,
};

/// Conservative character budget for the sources block fed to the
/// synthesis model.
const MAX_SOURCES_CHARS: usize = 24_000;

/// Runs the full query pipeline: search, synthesize (streaming), and
/// report sources.
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

    // Build context block from retrieved candidates.
    let mut sources_block = String::new();
    for result in &results {
        let text = format!("{}", result.artifact.contents);
        let content = curtana_knows::truncate_text(&text, 2000);
        let entry = format!(
            "<source taxonomy=\"{}\" author=\"{}\">\n{}\n</source>\n\n",
            curtana_knows::escape_xml(&result.taxonomy),
            curtana_knows::escape_xml(&format!("{}", result.artifact.author)),
            curtana_knows::escape_xml(content),
        );
        if sources_block.len() + entry.len() > MAX_SOURCES_CHARS {
            break;
        }
        sources_block.push_str(&entry);
    }

    let conv_context = format_conversation_context(&models.conversation_history);
    let escaped_query = curtana_knows::escape_xml(query);
    let synthesis_prompt = if conv_context.is_empty() {
        format!(
            "Based on the following sources, answer this query:\n\
             <user-query>{escaped_query}</user-query>\n\n\
             {sources_block}\
             Synthesize a clear, concise answer. Cite the sources you draw from."
        )
    } else {
        format!(
            "{conv_context}\
             Based on the following sources, answer the follow-up query:\n\
             <user-query>{escaped_query}</user-query>\n\n\
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
