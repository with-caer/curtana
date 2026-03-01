use std::io::Write;

use crossterm::event::Event as CrosstermEvent;
use curtana_knows::manifest::TaxonomyEntry;
use tokio::sync::mpsc;

/// Events produced by the terminal and background tasks.
pub enum Event {
    /// A terminal event (key press, resize, etc.).
    Terminal(CrosstermEvent),
    /// A streaming inference token.
    Token(String),
    /// A background command completed.
    CommandDone(CommandResult),
    /// An error from a background task.
    Error(String),
}

/// Result of a completed command.
pub enum CommandResult {
    /// Query completed with source references.
    Query { sources: Vec<SourceRef> },
    /// Status listing of taxonomies.
    Status {
        entries: Vec<(String, TaxonomyEntry)>,
    },
    /// Discovery phase 1: folder list for user selection.
    DiscoverFolders {
        folders: Vec<DiscoverFolder>,
    },
    /// A simple message response.
    Message(String),
}

/// A discovered folder shown to the user for selection.
pub struct DiscoverFolder {
    pub index: usize,
    pub name: String,
    pub already_tracked: bool,
}

/// A compact reference to a source artifact used in query results.
pub struct SourceRef {
    pub index: usize,
    pub score: f32,
    pub taxonomy: String,
    pub title: String,
}

/// An `io::Write` implementation that sends tokens through a channel.
///
/// Buffers partial UTF-8 sequences so each `Event::Token` contains
/// valid text.
pub struct ChannelWriter {
    tx: mpsc::UnboundedSender<Event>,
    buf: Vec<u8>,
}

impl ChannelWriter {
    pub fn new(tx: mpsc::UnboundedSender<Event>) -> Self {
        Self {
            tx,
            buf: Vec::new(),
        }
    }
}

impl Write for ChannelWriter {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.buf.extend_from_slice(data);
        match std::str::from_utf8(&self.buf) {
            Ok(s) => {
                if !s.is_empty() {
                    self.tx.send(Event::Token(s.to_string())).map_err(|_| {
                        std::io::Error::new(std::io::ErrorKind::BrokenPipe, "channel closed")
                    })?;
                }
                self.buf.clear();
            }
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                if valid_up_to > 0 {
                    let s = std::str::from_utf8(&self.buf[..valid_up_to]).unwrap();
                    self.tx.send(Event::Token(s.to_string())).map_err(|_| {
                        std::io::Error::new(std::io::ErrorKind::BrokenPipe, "channel closed")
                    })?;
                    self.buf.drain(..valid_up_to);
                }
            }
        }
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Async event stream that merges terminal and app events into a
/// single `mpsc` channel.
pub struct EventStream {
    rx: mpsc::UnboundedReceiver<Event>,
    tx: mpsc::UnboundedSender<Event>,
}

impl EventStream {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::unbounded_channel();

        // Spawn a dedicated thread for crossterm terminal event polling.
        let term_tx = tx.clone();
        std::thread::spawn(move || loop {
            if crossterm::event::poll(std::time::Duration::from_millis(50)).unwrap_or(false) {
                if let Ok(evt) = crossterm::event::read() {
                    if term_tx.send(Event::Terminal(evt)).is_err() {
                        break;
                    }
                }
            }
        });

        Self { rx, tx }
    }

    /// Returns a sender that background tasks can use to push events.
    pub fn tx(&self) -> mpsc::UnboundedSender<Event> {
        self.tx.clone()
    }

    /// Waits for the next event.
    pub async fn next(&mut self) -> Option<Event> {
        self.rx.recv().await
    }
}
