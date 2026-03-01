use std::path::Path;

use curtana_knows::manifest::Manifest;
use curtana_knows::router::Router;
use tokio::sync::mpsc;

use crate::event::{ChannelWriter, CommandResult, Event, SourceRef};

use super::Models;

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
    let results = router.search(&mut models.embed, query, 15).await;

    if results.is_empty() {
        tx.send(Event::Error("no results found.".into())).ok();
        return;
    }

    // Build context block from retrieved candidates.
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
            break;
        }
        sources_block.push_str(&entry);
    }

    let synthesis_prompt = format!(
        "Based on the following sources, answer this query: \"{query}\"\n\n\
         {sources_block}\
         Synthesize a clear, concise answer. Cite the sources you draw from."
    );

    // Stream the synthesized answer through the channel.
    let mut writer = ChannelWriter::new(tx.clone());
    if let Err(e) = models.chat.infer(&synthesis_prompt, &mut writer) {
        tx.send(Event::Error(format!("inference error: {e:?}")))
            .ok();
        return;
    }

    // Build compact source references.
    let sources: Vec<SourceRef> = results
        .iter()
        .enumerate()
        .map(|(i, result)| {
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
            SourceRef {
                index: i + 1,
                score: result.score,
                taxonomy: result.taxonomy.clone(),
                title,
            }
        })
        .collect();

    tx.send(Event::CommandDone(CommandResult::Query { sources }))
        .ok();
}
