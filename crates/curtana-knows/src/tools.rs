use std::path::PathBuf;

use curtana_infers::TextEmbeddingModel;

use crate::{
    manifest::Manifest,
    open_taxonomy_store,
    router::{Router, ScoredArtifact},
    truncate_text, Artifact, BrowseOrder,
};

/// Maximum characters in a single tool result. Tool results are injected
/// into the gathering conversation, so they must fit within the model's
/// context window alongside the system prompt and prior turns.
const MAX_TOOL_RESULT_CHARS: usize = 3000;

/// Result of parsing model output for tool-call markers.
pub enum ParseResult {
    /// Found `<tool>name({...})</tool>`.
    ToolCall(ToolCall),
    /// Found `<done/>`.
    Done,
    /// No markers found — treat as direct answer.
    Answer(String),
}

/// A parsed tool invocation.
pub struct ToolCall {
    pub name: String,
    pub args: serde_json::Value,
}

/// Result returned by the tool executor.
pub struct ToolResult {
    /// Formatted text to feed back to the LLM.
    pub text: String,
    /// Source artifacts for the detail panel (search-only).
    pub sources: Vec<ScoredArtifact>,
}

/// Parses model output for `<tool>name({...})</tool>` or `<done/>` markers.
pub fn parse_tool_response(output: &str) -> ParseResult {
    let trimmed = output.trim();

    // Check for <done/>.
    if trimmed.contains("<done/>") {
        return ParseResult::Done;
    }

    // Check for <tool>...</tool>.
    if let Some(start) = trimmed.find("<tool>") {
        let after_tag = start + "<tool>".len();
        if let Some(end) = trimmed[after_tag..].find("</tool>") {
            let inner = trimmed[after_tag..after_tag + end].trim();
            return match parse_tool_call(inner) {
                Ok(call) => ParseResult::ToolCall(call),
                Err(e) => ParseResult::Answer(format!("Tool parse error: {e}")),
            };
        }
    }

    // No markers — treat as direct answer.
    ParseResult::Answer(trimmed.to_string())
}

/// Parses the inner content of a `<tool>...</tool>` block.
///
/// Expected format: `name({...})` or `name()`.
fn parse_tool_call(inner: &str) -> Result<ToolCall, String> {
    let paren_pos = inner
        .find('(')
        .ok_or_else(|| format!("missing '(' in tool call: {inner}"))?;

    let name = inner[..paren_pos].trim().to_string();
    if name.is_empty() {
        return Err("empty tool name".to_string());
    }

    let rest = inner[paren_pos + 1..].trim();
    let rest = rest
        .strip_suffix(')')
        .ok_or_else(|| format!("missing ')' in tool call: {inner}"))?
        .trim();

    let args = if rest.is_empty() {
        serde_json::Value::Object(Default::default())
    } else {
        serde_json::from_str(rest)
            .map_err(|e| format!("invalid JSON args in tool call: {e}"))?
    };

    Ok(ToolCall { name, args })
}

/// Executes tool calls against the knowledge store.
pub struct ToolExecutor {
    manifest: Manifest,
    data_dir: PathBuf,
}

impl ToolExecutor {
    pub fn new(manifest: Manifest, data_dir: PathBuf) -> Self {
        Self { manifest, data_dir }
    }

    pub async fn execute(
        &self,
        embed_model: &mut TextEmbeddingModel,
        call: &ToolCall,
    ) -> ToolResult {
        let mut result = match call.name.as_str() {
            "list_taxonomies" => self.list_taxonomies().await,
            "count" => self.tool_count(&call.args).await,
            "search" => self.tool_search(embed_model, &call.args).await,
            "browse" => self.tool_browse(&call.args).await,
            "filter" => self.tool_filter(&call.args).await,
            other => ToolResult {
                text: format!("Unknown tool: {other}"),
                sources: vec![],
            },
        };

        // Cap the text to prevent context window overflow during gathering.
        if result.text.len() > MAX_TOOL_RESULT_CHARS {
            result.text = truncate_text(&result.text, MAX_TOOL_RESULT_CHARS).to_string();
            result.text.push_str("\n... (truncated)");
        }

        result
    }

    async fn list_taxonomies(&self) -> ToolResult {
        let mut lines = Vec::new();
        for (name, entry) in &self.manifest.taxonomies {
            let store = open_taxonomy_store(&self.data_dir, name).await;
            let count = store.count().await;
            let desc = if entry.description.is_empty() {
                "(no description)"
            } else {
                &entry.description
            };
            lines.push(format!("{name}: {desc} ({count} items)"));
        }
        ToolResult {
            text: if lines.is_empty() {
                "No taxonomies available.".to_string()
            } else {
                lines.join("\n")
            },
            sources: vec![],
        }
    }

    async fn tool_count(&self, args: &serde_json::Value) -> ToolResult {
        let taxonomy = args["taxonomy"].as_str().unwrap_or("");
        if taxonomy.is_empty() || !self.manifest.taxonomies.contains_key(taxonomy) {
            return ToolResult {
                text: format!("Unknown taxonomy: {taxonomy:?}"),
                sources: vec![],
            };
        }
        let store = open_taxonomy_store(&self.data_dir, taxonomy).await;
        let count = store.count().await;
        ToolResult {
            text: format!("{taxonomy}: {count} artifacts"),
            sources: vec![],
        }
    }

    async fn tool_search(
        &self,
        embed_model: &mut TextEmbeddingModel,
        args: &serde_json::Value,
    ) -> ToolResult {
        let query = args["query"].as_str().unwrap_or("");
        let top_k = args["top_k"].as_u64().unwrap_or(10) as usize;
        let taxonomy = args["taxonomy"].as_str();

        if query.is_empty() {
            return ToolResult {
                text: "search requires a 'query' argument".to_string(),
                sources: vec![],
            };
        }

        let results = if let Some(taxonomy) = taxonomy {
            if !self.manifest.taxonomies.contains_key(taxonomy) {
                return ToolResult {
                    text: format!("Unknown taxonomy: {taxonomy:?}"),
                    sources: vec![],
                };
            }
            let store = open_taxonomy_store(&self.data_dir, taxonomy).await;
            let artifacts = store.search(embed_model, query, top_k).await;
            artifacts
                .into_iter()
                .map(|a| ScoredArtifact {
                    taxonomy: taxonomy.to_string(),
                    score: 0.0,
                    artifact: a,
                })
                .collect()
        } else {
            let router = Router::new(self.manifest.clone(), self.data_dir.clone());
            router.search(embed_model, query, top_k).await
        };

        let text = format_artifacts(&results);
        ToolResult {
            text,
            sources: results,
        }
    }

    async fn tool_browse(&self, args: &serde_json::Value) -> ToolResult {
        let taxonomy = args["taxonomy"].as_str().unwrap_or("");
        let offset = args["offset"].as_u64().unwrap_or(0) as usize;
        let limit = args["limit"].as_u64().unwrap_or(5) as usize;
        let order = match args["order"].as_str().unwrap_or("desc") {
            "asc" => BrowseOrder::Asc,
            _ => BrowseOrder::Desc,
        };

        if taxonomy.is_empty() || !self.manifest.taxonomies.contains_key(taxonomy) {
            return ToolResult {
                text: format!("Unknown taxonomy: {taxonomy:?}"),
                sources: vec![],
            };
        }

        let store = open_taxonomy_store(&self.data_dir, taxonomy).await;
        let artifacts = store.browse(offset, limit, order).await;

        let text = artifacts
            .iter()
            .enumerate()
            .map(|(i, a)| format_single_artifact(i + offset + 1, a, taxonomy))
            .collect::<Vec<_>>()
            .join("\n");

        ToolResult {
            text: if text.is_empty() {
                "No artifacts found.".to_string()
            } else {
                text
            },
            sources: vec![],
        }
    }

    async fn tool_filter(&self, args: &serde_json::Value) -> ToolResult {
        let taxonomy = args["taxonomy"].as_str().unwrap_or("");
        let author = args["author"].as_str().map(String::from);
        let after = args["after"].as_u64();
        let before = args["before"].as_u64();
        let limit = args["limit"].as_u64().unwrap_or(10) as usize;

        if taxonomy.is_empty() || !self.manifest.taxonomies.contains_key(taxonomy) {
            return ToolResult {
                text: format!("Unknown taxonomy: {taxonomy:?}"),
                sources: vec![],
            };
        }

        let store = open_taxonomy_store(&self.data_dir, taxonomy).await;
        let artifacts = store.filter(author, after, before, limit).await;

        let text = artifacts
            .iter()
            .enumerate()
            .map(|(i, a)| format_single_artifact(i + 1, a, taxonomy))
            .collect::<Vec<_>>()
            .join("\n");

        ToolResult {
            text: if text.is_empty() {
                "No artifacts match the filter criteria.".to_string()
            } else {
                text
            },
            sources: vec![],
        }
    }
}

/// Formats a list of scored artifacts as compact entries.
fn format_artifacts(results: &[ScoredArtifact]) -> String {
    if results.is_empty() {
        return "No results found.".to_string();
    }
    results
        .iter()
        .enumerate()
        .map(|(i, r)| format_single_artifact(i + 1, &r.artifact, &r.taxonomy))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Formats a single artifact as a compact entry for the LLM context.
fn format_single_artifact(index: usize, artifact: &Artifact, taxonomy: &str) -> String {
    let content = format!("{}", artifact.contents);
    let content = truncate_text(&content, 500);
    format!(
        "[{index}] Author: {} | Date: {} | Taxonomy: {taxonomy}\n{content}",
        artifact.author, artifact.timestamp,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tool_call_search() {
        let output = r#"I'll search for that. <tool>search({"query": "emails from alice", "taxonomy": "inbox"})</tool>"#;
        match parse_tool_response(output) {
            ParseResult::ToolCall(call) => {
                assert_eq!(call.name, "search");
                assert_eq!(call.args["query"], "emails from alice");
                assert_eq!(call.args["taxonomy"], "inbox");
            }
            other => panic!("expected ToolCall, got {:?}", result_name(&other)),
        }
    }

    #[test]
    fn parse_tool_call_no_args() {
        let output = "<tool>list_taxonomies()</tool>";
        match parse_tool_response(output) {
            ParseResult::ToolCall(call) => {
                assert_eq!(call.name, "list_taxonomies");
                assert!(call.args.is_object());
                assert!(call.args.as_object().unwrap().is_empty());
            }
            other => panic!("expected ToolCall, got {:?}", result_name(&other)),
        }
    }

    #[test]
    fn parse_done() {
        let output = "I have enough information. <done/>";
        assert!(matches!(parse_tool_response(output), ParseResult::Done));
    }

    #[test]
    fn parse_answer_fallback() {
        let output = "Here is the answer to your question.";
        match parse_tool_response(output) {
            ParseResult::Answer(text) => {
                assert_eq!(text, "Here is the answer to your question.");
            }
            other => panic!("expected Answer, got {:?}", result_name(&other)),
        }
    }

    #[test]
    fn parse_malformed_json() {
        let output = "<tool>search({bad json})</tool>";
        match parse_tool_response(output) {
            ParseResult::Answer(text) => {
                assert!(text.contains("Tool parse error"));
            }
            other => panic!("expected Answer with error, got {:?}", result_name(&other)),
        }
    }

    #[test]
    fn parse_with_whitespace() {
        let output = "  <tool>  browse( {\"taxonomy\": \"inbox\"} )  </tool>  ";
        match parse_tool_response(output) {
            ParseResult::ToolCall(call) => {
                assert_eq!(call.name, "browse");
                assert_eq!(call.args["taxonomy"], "inbox");
            }
            other => panic!("expected ToolCall, got {:?}", result_name(&other)),
        }
    }

    fn result_name(r: &ParseResult) -> &'static str {
        match r {
            ParseResult::ToolCall(_) => "ToolCall",
            ParseResult::Done => "Done",
            ParseResult::Answer(_) => "Answer",
        }
    }
}
