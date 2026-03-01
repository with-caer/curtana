use crate::commands::{COMMANDS, CommandInfo};
use crate::event::{Progress, SourceRef};
use curtana_knows::manifest::TaxonomyEntry;

/// Which pane currently has focus for scrolling.
#[derive(Clone, Copy, PartialEq)]
pub enum ActivePane {
    Chat,
    Detail,
}

/// Completion popup state for slash commands.
pub struct CompletionState {
    pub matches: Vec<&'static CommandInfo>,
    pub selected: usize,
}

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
    /// Scroll offset for the detail panel.
    pub detail_scroll_offset: usize,
    /// Optional right-side detail panel.
    pub detail_panel: Option<DetailView>,
    /// Which pane currently has keyboard focus.
    pub active_pane: ActivePane,
    /// Whether the app loop should keep running.
    pub running: bool,
    /// Current status indicator.
    pub status: AppStatus,
    /// Active command completion popup.
    pub completion: Option<CompletionState>,
    /// Current spinner animation frame.
    pub spinner_frame: usize,
    /// Progress for a long-running operation (shown in header).
    pub progress: Option<Progress>,
    /// Activity log lines shown during the agent gathering phase.
    pub activity_lines: Vec<String>,
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
            detail_scroll_offset: 0,
            detail_panel: None,
            active_pane: ActivePane::Chat,
            running: true,
            status: AppStatus::Idle,
            completion: None,
            spinner_frame: 0,
            progress: None,
            activity_lines: Vec::new(),
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

    /// Recompute the completion popup based on current input.
    pub fn update_completion(&mut self) {
        // Only complete when input starts with `/` and has no whitespace (still typing command name).
        let prefix = &self.input;
        if prefix.starts_with('/') && !prefix.contains(char::is_whitespace) {
            let matches: Vec<&'static CommandInfo> = COMMANDS
                .iter()
                .filter(|c| c.name.starts_with(prefix))
                .collect();
            if matches.is_empty() {
                self.completion = None;
            } else {
                let prev_selected = self
                    .completion
                    .as_ref()
                    .and_then(|cs| cs.matches.get(cs.selected).map(|m| m.name));
                let selected = prev_selected
                    .and_then(|name| matches.iter().position(|m| m.name == name))
                    .unwrap_or(0);
                self.completion = Some(CompletionState { matches, selected });
            }
        } else {
            self.completion = None;
        }
    }

    /// Accept the currently selected completion, replacing input with the command name.
    pub fn accept_completion(&mut self) {
        if let Some(cs) = self.completion.take()
            && let Some(cmd) = cs.matches.get(cs.selected)
        {
            self.input = cmd.name.to_string();
            self.cursor_position = self.input.len();
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
        match self.active_pane {
            ActivePane::Chat => {
                self.scroll_offset = self.scroll_offset.saturating_add(1);
            }
            ActivePane::Detail => {
                self.detail_scroll_offset = self.detail_scroll_offset.saturating_add(1);
            }
        }
    }

    pub fn scroll_down(&mut self) {
        match self.active_pane {
            ActivePane::Chat => {
                self.scroll_offset = self.scroll_offset.saturating_sub(1);
            }
            ActivePane::Detail => {
                self.detail_scroll_offset = self.detail_scroll_offset.saturating_sub(1);
            }
        }
    }

    pub fn scroll_page_up(&mut self) {
        match self.active_pane {
            ActivePane::Chat => {
                self.scroll_offset = self.scroll_offset.saturating_add(10);
            }
            ActivePane::Detail => {
                self.detail_scroll_offset = self.detail_scroll_offset.saturating_add(10);
            }
        }
    }

    pub fn scroll_page_down(&mut self) {
        match self.active_pane {
            ActivePane::Chat => {
                self.scroll_offset = self.scroll_offset.saturating_sub(10);
            }
            ActivePane::Detail => {
                self.detail_scroll_offset = self.detail_scroll_offset.saturating_sub(10);
            }
        }
    }

    pub fn scroll_to_bottom(&mut self) {
        self.scroll_offset = 0;
    }

    pub fn clear_activity(&mut self) {
        self.activity_lines.clear();
    }

    pub fn toggle_detail_panel(&mut self) {
        if self.detail_panel.is_some() {
            match self.active_pane {
                ActivePane::Chat => self.active_pane = ActivePane::Detail,
                ActivePane::Detail => {
                    self.active_pane = ActivePane::Chat;
                    self.detail_scroll_offset = 0;
                }
            }
        }
    }
}
