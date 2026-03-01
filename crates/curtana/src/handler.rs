use crossterm::event::{Event as CrosstermEvent, KeyCode, KeyEventKind, KeyModifiers};
use tokio::sync::mpsc;

use crate::app::{App, AppStatus, DetailView, Message};
use crate::commands::{self, Command, CommandRequest};
use crate::event::{CommandResult, Event};

const HELP_TEXT: &str = "\
Available commands:
  <text>       Query your knowledge base
  /status      Show tracked taxonomies
  /discover    Discover and select source folders
  /ingest      Ingest artifacts and generate embeddings
  /help        Show this help
  /quit        Exit curtana

Keyboard shortcuts:
  Enter        Submit input
  Tab          Toggle detail panel
  Up/Down      Scroll chat
  PgUp/PgDn    Scroll chat (fast)
  Ctrl+C       Exit";

/// Processes an event and mutates application state.
pub fn handle_event(
    app: &mut App,
    event: Event,
    cmd_tx: &mpsc::UnboundedSender<CommandRequest>,
) {
    match event {
        Event::Terminal(CrosstermEvent::Key(key)) if key.kind == KeyEventKind::Press => {
            handle_key(app, key.code, key.modifiers, cmd_tx);
        }
        Event::Token(token) => {
            app.append_to_last_message(&token);
        }
        Event::CommandDone(result) => {
            handle_command_done(app, result);
        }
        Event::Error(err) => {
            app.add_message(Message::system(format!("Error: {err}")));
            app.status = AppStatus::Idle;
        }
        _ => {}
    }
}

fn handle_key(
    app: &mut App,
    code: KeyCode,
    modifiers: KeyModifiers,
    cmd_tx: &mpsc::UnboundedSender<CommandRequest>,
) {
    match code {
        KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
            app.running = false;
        }
        KeyCode::Char(c) => {
            app.insert_char(c);
        }
        KeyCode::Backspace => {
            app.delete_char();
        }
        KeyCode::Enter => {
            if matches!(app.status, AppStatus::Loading(_)) {
                return;
            }
            let input = app.take_input();
            if input.is_empty() {
                return;
            }
            if matches!(app.status, AppStatus::AwaitingDiscoverSelection) {
                app.add_message(Message::user(input.clone()));
                app.status = AppStatus::Loading("Updating manifest...".into());
                cmd_tx.send(CommandRequest::DiscoverSelect(input)).ok();
            } else {
                submit(app, &input, cmd_tx);
            }
        }
        KeyCode::Left => app.move_cursor_left(),
        KeyCode::Right => app.move_cursor_right(),
        KeyCode::Home => app.move_cursor_home(),
        KeyCode::End => app.move_cursor_end(),
        KeyCode::Up => app.scroll_up(),
        KeyCode::Down => app.scroll_down(),
        KeyCode::PageUp => app.scroll_page_up(),
        KeyCode::PageDown => app.scroll_page_down(),
        KeyCode::Tab => app.toggle_detail_panel(),
        _ => {}
    }
}

/// Submits a command on startup without requiring user input text.
pub fn submit_auto_command(
    app: &mut App,
    cmd: CommandRequest,
    cmd_tx: &mpsc::UnboundedSender<CommandRequest>,
) {
    match &cmd {
        CommandRequest::Discover => {
            app.add_message(Message::user("/discover".into()));
            app.status = AppStatus::Loading("Discovering...".into());
        }
        CommandRequest::Ingest => {
            app.add_message(Message::user("/ingest".into()));
            app.add_message(Message::system(String::new()));
            app.status = AppStatus::Loading("Ingesting...".into());
        }
        CommandRequest::Query(q) => {
            app.add_message(Message::user(q.clone()));
            app.add_message(Message::assistant(String::new()));
            app.status = AppStatus::Loading("Querying...".into());
        }
        CommandRequest::Status => {
            app.add_message(Message::user("/status".into()));
            app.status = AppStatus::Loading("Loading status...".into());
        }
        CommandRequest::DiscoverSelect(_) => return,
    }
    cmd_tx.send(cmd).ok();
}

fn submit(app: &mut App, input: &str, cmd_tx: &mpsc::UnboundedSender<CommandRequest>) {
    let command = commands::parse(input);

    match command {
        Command::Query(query) => {
            app.add_message(Message::user(input.to_string()));
            app.add_message(Message::assistant(String::new()));
            app.status = AppStatus::Loading("Querying...".into());
            cmd_tx.send(CommandRequest::Query(query)).ok();
        }
        Command::Status => {
            app.add_message(Message::user(input.to_string()));
            app.status = AppStatus::Loading("Loading status...".into());
            cmd_tx.send(CommandRequest::Status).ok();
        }
        Command::Discover => {
            app.add_message(Message::user(input.to_string()));
            app.status = AppStatus::Loading("Discovering...".into());
            cmd_tx.send(CommandRequest::Discover).ok();
        }
        Command::Ingest => {
            app.add_message(Message::user(input.to_string()));
            app.add_message(Message::system(String::new()));
            app.status = AppStatus::Loading("Ingesting...".into());
            cmd_tx.send(CommandRequest::Ingest).ok();
        }
        Command::Help => {
            app.add_message(Message::user(input.to_string()));
            app.add_message(Message::system(HELP_TEXT.to_string()));
        }
        Command::Quit => {
            app.running = false;
        }
    }
}

fn handle_command_done(app: &mut App, result: CommandResult) {
    match result {
        CommandResult::Query { sources } => {
            app.status = AppStatus::Idle;
            let mut source_text = String::from("\n\n--- Sources ---");
            for s in &sources {
                source_text.push_str(&format!(
                    "\n[{}] {:.4} | {} | {}",
                    s.index, s.score, s.taxonomy, s.title,
                ));
            }
            app.append_to_last_message(&source_text);
            app.detail_panel = Some(DetailView::Sources(sources));
        }
        CommandResult::Status { entries } => {
            app.status = AppStatus::Idle;
            let mut text = String::from("Tracked taxonomies:\n");
            for (name, entry) in &entries {
                text.push_str(&format!("\n  {name}"));
                if !entry.description.is_empty() {
                    text.push_str(&format!(" \u{2014} {}", entry.description));
                }
            }
            app.add_message(Message::system(text));
            app.detail_panel = Some(DetailView::TaxonomyList(entries));
        }
        CommandResult::DiscoverFolders { folders } => {
            let mut text = String::from("Available folders:\n");
            for f in &folders {
                let marker = if f.already_tracked {
                    " (already tracked)"
                } else {
                    ""
                };
                text.push_str(&format!("\n  [{}] {}{}", f.index, f.name, marker));
            }
            text.push_str("\n\nEnter folder numbers (e.g. 1,3) or \"all\":");
            app.add_message(Message::system(text));
            app.status = AppStatus::AwaitingDiscoverSelection;
        }
        CommandResult::Message(msg) => {
            app.status = AppStatus::Idle;
            app.add_message(Message::system(msg));
        }
    }
}
