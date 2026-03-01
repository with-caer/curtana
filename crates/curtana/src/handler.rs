use crossterm::event::{Event as CrosstermEvent, KeyCode, KeyEventKind, KeyModifiers};
use tokio::sync::mpsc;

use crate::app::{App, AppStatus, DetailView, Message};
use crate::commands::{self, Command, CommandRequest};
use crate::event::{CommandResult, Event};

const HELP_TEXT: &str = "\
## Commands

- `<text>` \u{2014} Query your knowledge base
- `/status` \u{2014} Show tracked taxonomies
- `/discover` \u{2014} Discover and select source folders
- `/ingest` \u{2014} Ingest artifacts and generate embeddings
- `/help` \u{2014} Show this help
- `/quit` \u{2014} Exit curtana

## Keyboard shortcuts

- `Enter` \u{2014} Submit input
- `Tab` \u{2014} Switch focus between chat and detail panel
- `Up`/`Down` \u{2014} Scroll focused pane
- `PgUp`/`PgDn` \u{2014} Scroll focused pane (fast)
- `Ctrl+C` \u{2014} Exit";

/// Processes an event and mutates application state.
pub fn handle_event(app: &mut App, event: Event, cmd_tx: &mpsc::UnboundedSender<CommandRequest>) {
    match event {
        Event::Terminal(CrosstermEvent::Key(key)) if key.kind == KeyEventKind::Press => {
            handle_key(app, key.code, key.modifiers, cmd_tx);
        }
        Event::Token(token) => {
            if !app.activity_lines.is_empty() {
                app.clear_activity();
            }
            app.append_to_last_message(&token);
        }
        Event::CommandDone(result) => {
            app.clear_activity();
            handle_command_done(app, result);
        }
        Event::Error(err) => {
            app.clear_activity();
            app.add_message(Message::system(format!("Error: {err}")));
            app.status = AppStatus::Idle;
            app.progress = None;
        }
        Event::Tick => {
            if matches!(app.status, AppStatus::Loading(_)) {
                app.spinner_frame = app.spinner_frame.wrapping_add(1);
            }
        }
        Event::Progress(progress) => {
            app.progress = Some(progress);
        }
        Event::StatusText(msg) => {
            if matches!(app.status, AppStatus::Loading(_)) {
                app.status = AppStatus::Loading(msg);
            }
        }
        Event::ActivityLine(line) => {
            app.activity_lines.push(line);
            app.scroll_to_bottom();
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
    let completing = app.completion.is_some();

    match code {
        KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
            app.running = false;
        }
        KeyCode::Char(c) => {
            app.insert_char(c);
            app.update_completion();
        }
        KeyCode::Backspace => {
            app.delete_char();
            app.update_completion();
        }
        KeyCode::Tab if completing => {
            app.accept_completion();
        }
        KeyCode::Esc if completing => {
            app.completion = None;
        }
        KeyCode::Up if completing => {
            if let Some(cs) = &mut app.completion {
                if cs.selected == 0 {
                    cs.selected = cs.matches.len() - 1;
                } else {
                    cs.selected -= 1;
                }
            }
        }
        KeyCode::Down if completing => {
            if let Some(cs) = &mut app.completion {
                cs.selected = (cs.selected + 1) % cs.matches.len();
            }
        }
        KeyCode::Enter => {
            if matches!(app.status, AppStatus::Loading(_)) {
                return;
            }
            if completing {
                app.accept_completion();
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
    app.progress = None;
    match result {
        CommandResult::Query { sources } => {
            app.status = AppStatus::Idle;
            app.detail_panel = Some(DetailView::Sources(sources));
            app.detail_scroll_offset = 0;
        }
        CommandResult::Status { entries } => {
            app.status = AppStatus::Idle;
            let mut text = String::from("## Tracked taxonomies\n\n");
            for (name, entry) in &entries {
                if entry.description.is_empty() {
                    text.push_str(&format!("- **{name}**\n"));
                } else {
                    text.push_str(&format!("- **{name}** \u{2014} {}\n", entry.description));
                }
            }
            app.add_message(Message::system(text));
            app.detail_panel = Some(DetailView::TaxonomyList(entries));
            app.detail_scroll_offset = 0;
        }
        CommandResult::DiscoverFolders { folders } => {
            let mut text = String::from("## Available folders\n\n");

            // Group folders by source, preserving order of first appearance.
            let mut groups: Vec<(String, Vec<&crate::event::DiscoverFolder>)> = Vec::new();
            for f in &folders {
                let label = format_source_label(&f.source_username, &f.source_host);
                if let Some(group) = groups.iter_mut().find(|(l, _)| l == &label) {
                    group.1.push(f);
                } else {
                    groups.push((label, vec![f]));
                }
            }

            for (label, group_folders) in &groups {
                text.push_str(&format!("### {label}\n\n"));
                for f in group_folders {
                    let marker = if f.already_tracked {
                        " *(already tracked)*"
                    } else {
                        ""
                    };
                    text.push_str(&format!("- `[{}]` {}{}\n", f.index, f.name, marker));
                }
                text.push('\n');
            }

            text.push_str("Enter folder numbers (e.g. `1,3`) or `all`:");
            app.add_message(Message::system(text));
            app.status = AppStatus::AwaitingDiscoverSelection;
        }
        CommandResult::Message(msg) => {
            app.status = AppStatus::Idle;
            app.add_message(Message::system(msg));
        }
    }
}

/// Formats a human-readable label for an IMAP source.
///
/// - `john` + `mail.example.com` → `john@mail.example.com`
/// - `me@caer.cc` + `127.0.0.1` → `me@caer.cc via 127.0.0.1`
/// - `me@gmail.com` + `gmail.com` → `me@gmail.com`
fn format_source_label(username: &str, host: &str) -> String {
    if let Some(domain) = username.rsplit_once('@').map(|(_, d)| d) {
        if domain == host {
            username.to_string()
        } else {
            format!("{username} via {host}")
        }
    } else {
        format!("{username}@{host}")
    }
}
