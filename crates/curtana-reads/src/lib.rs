pub mod imap;

/// Convert a domain object into a Markdown document suitable for
/// chunking, embedding, or presentation to an LLM.
pub trait ToMarkdown {
    fn to_markdown(&self) -> String;
}

/// A single item produced by a source integration.
pub trait SourceItem: ToMarkdown {
    fn id(&self) -> &str;
    fn timestamp(&self) -> i64;
    fn author(&self) -> &str;
}

/// Result of an incremental source fetch: the fetched items plus an
/// opaque cursor for resuming from where we left off.
pub struct ReadResult<T> {
    pub items: Vec<T>,
    pub cursor: Option<String>,
}
