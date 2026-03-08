pub mod imap;

/// Result of an incremental source fetch: the fetched items plus an
/// opaque cursor for resuming from where we left off.
pub struct ReadResult<T> {
    pub items: Vec<T>,
    pub cursor: Option<String>,
}
