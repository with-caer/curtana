use std::io::Write;

use crossterm::event::Event as CrosstermEvent;
use curtana_knows::manifest::TaxonomyEntry;
use tokio::sync::mpsc;

/// Structured progress for a long-running operation (e.g. embedding).
pub struct Progress {
    pub current: usize,
    pub total: usize,
    pub label: String,
}

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
    /// Timer tick for animations (spinner, progress bar).
    Tick,
    /// Progress update for a long-running operation.
    Progress(Progress),
    /// Update the loading status message in the header.
    StatusText(String),
    /// An activity log line shown during the agent gathering phase.
    ActivityLine(String),
}

/// Result of a completed command.
pub enum CommandResult {
    /// Query completed.
    QueryDone,
    /// Status listing of taxonomies.
    Status {
        entries: Vec<(String, TaxonomyEntry)>,
    },
    /// Explore phase 1: folder list for user selection.
    ExploreFolders { folders: Vec<ExploreFolder> },
    /// A simple message response.
    Message(String),
}

/// A discovered folder shown to the user for selection.
pub struct ExploreFolder {
    pub index: usize,
    pub name: String,
    pub source_host: String,
    pub source_username: String,
    pub already_tracked: bool,
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
        std::thread::spawn(move || {
            loop {
                if crossterm::event::poll(std::time::Duration::from_millis(50)).unwrap_or(false)
                    && let Ok(evt) = crossterm::event::read()
                    && term_tx.send(Event::Terminal(evt)).is_err()
                {
                    break;
                }
            }
        });

        // Spawn a tick thread for animations (~12.5fps).
        let tick_tx = tx.clone();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_millis(80));
                if tick_tx.send(Event::Tick).is_err() {
                    break;
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
