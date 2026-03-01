use crate::event::SourceRef;
use curtana_knows::manifest::TaxonomyEntry;

/// Application state for the TUI.
pub struct App {
    /// Current text in the input buffer.
    pub input: String,
    /// Byte offset of the cursor within `input`.
    pub cursor_position: usize,
    /// Chat message history.
    pub messages: Vec<Message>,
    /// Scroll offset from the bottom (0 = fully scrolled down).
    pub scroll_offset: usize,
    /// Optional right-side detail panel.
    pub detail_panel: Option<DetailView>,
    /// Whether the app loop should keep running.
    pub running: bool,
    /// Current status indicator.
    pub status: AppStatus,
}

pub enum AppStatus {
    Idle,
    Loading(String),
    AwaitingDiscoverSelection,
}

pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Role {
    User,
    Assistant,
    System,
}

pub enum DetailView {
    TaxonomyList(Vec<(String, TaxonomyEntry)>),
    Sources(Vec<SourceRef>),
}

impl Message {
    pub fn user(content: String) -> Self {
        Self {
            role: Role::User,
            content,
        }
    }

    pub fn assistant(content: String) -> Self {
        Self {
            role: Role::Assistant,
            content,
        }
    }

    pub fn system(content: String) -> Self {
        Self {
            role: Role::System,
            content,
        }
    }
}

impl App {
    pub fn new() -> Self {
        let welcome = Message::system(
            "Welcome to curtana. Type a question to search, or /help for commands.".to_string(),
        );
        Self {
            input: String::new(),
            cursor_position: 0,
            messages: vec![welcome],
            scroll_offset: 0,
            detail_panel: None,
            running: true,
            status: AppStatus::Idle,
        }
    }

    pub fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor_position, c);
        self.cursor_position += c.len_utf8();
    }

    pub fn delete_char(&mut self) {
        if self.cursor_position > 0 {
            let prev = self.input[..self.cursor_position]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.input.drain(prev..self.cursor_position);
            self.cursor_position = prev;
        }
    }

    pub fn move_cursor_left(&mut self) {
        if self.cursor_position > 0 {
            self.cursor_position = self.input[..self.cursor_position]
                .char_indices()
                .next_back()
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
    }

    pub fn move_cursor_right(&mut self) {
        if self.cursor_position < self.input.len() {
            self.cursor_position = self.input[self.cursor_position..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| self.cursor_position + i)
                .unwrap_or(self.input.len());
        }
    }

    pub fn move_cursor_home(&mut self) {
        self.cursor_position = 0;
    }

    pub fn move_cursor_end(&mut self) {
        self.cursor_position = self.input.len();
    }

    pub fn take_input(&mut self) -> String {
        self.cursor_position = 0;
        std::mem::take(&mut self.input)
    }

    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
        self.scroll_to_bottom();
    }

    pub fn append_to_last_message(&mut self, text: &str) {
        if let Some(msg) = self.messages.last_mut() {
            msg.content.push_str(text);
        }
        self.scroll_to_bottom();
    }

    pub fn scroll_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_add(1);
    }

    pub fn scroll_down(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(1);
    }

    pub fn scroll_page_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_add(10);
    }

    pub fn scroll_page_down(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(10);
    }

    pub fn scroll_to_bottom(&mut self) {
        self.scroll_offset = 0;
    }

    pub fn toggle_detail_panel(&mut self) {
        if self.detail_panel.is_some() {
            self.detail_panel = None;
        }
    }
}
