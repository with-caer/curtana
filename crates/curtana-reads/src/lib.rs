pub mod imap;

/// Convert a domain object into a Markdown document suitable for
/// chunking, embedding, or presentation to an LLM.
pub trait ToMarkdown {
    fn to_markdown(&self) -> String;
}
