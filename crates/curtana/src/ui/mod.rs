mod chat;
mod input;
mod markdown;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::app::{App, AppStatus};

const SPINNER_FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const PROGRESS_WIDTH: usize = 20;

/// Renders the entire TUI.
pub fn render(frame: &mut Frame, app: &App) {
    let area = frame.area();

    // Vertical layout: header | body | input.
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(3),
        ])
        .split(area);

    let header_area = chunks[0];
    let body_area = chunks[1];
    let input_area = chunks[2];

    // Header with status.
    let brand = Span::styled(
        " curtana",
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    let yellow_bold = Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::BOLD);

    let header_line = match &app.status {
        AppStatus::Idle => Line::from(brand),
        AppStatus::Loading(_) if app.progress.is_some() => {
            let p = app.progress.as_ref().unwrap();
            let filled = if p.total > 0 {
                (p.current * PROGRESS_WIDTH) / p.total
            } else {
                0
            };
            let empty = PROGRESS_WIDTH - filled;
            let bar = format!(
                "{}{} {}/{}",
                "\u{2501}".repeat(filled),
                "\u{254c}".repeat(empty),
                p.current,
                p.total,
            );
            Line::from(vec![
                brand,
                Span::styled(" \u{2014} ", yellow_bold),
                Span::styled(p.label.clone(), yellow_bold),
                Span::styled(" [", yellow_bold),
                Span::styled(bar, yellow_bold),
                Span::styled("]", yellow_bold),
            ])
        }
        AppStatus::Loading(msg) => {
            let spinner = SPINNER_FRAMES[app.spinner_frame % SPINNER_FRAMES.len()];
            Line::from(vec![
                brand,
                Span::styled(format!(" \u{2014} {spinner} {msg}"), yellow_bold),
            ])
        }
        AppStatus::AwaitingDiscoverSelection => Line::from(vec![
            brand,
            Span::styled(
                " \u{2014} select folders to track",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
    };
    frame.render_widget(Paragraph::new(header_line), header_area);

    // Body: chat area.
    chat::render(frame, app, body_area);

    // Input area.
    input::render(frame, app, input_area);
}
