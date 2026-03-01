mod chat;
mod detail;
mod input;

use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::app::{App, AppStatus};

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
    let title = match &app.status {
        AppStatus::Idle => Span::styled(
            " curtana",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        AppStatus::Loading(msg) => Span::styled(
            format!(" curtana \u{2014} {msg}"),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        AppStatus::AwaitingDiscoverSelection => Span::styled(
            " curtana \u{2014} select folders to track",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
    };
    frame.render_widget(Paragraph::new(title), header_area);

    // Body: chat area + optional detail panel.
    if app.detail_panel.is_some() {
        let body_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(body_area);

        chat::render(frame, app, body_chunks[0]);
        detail::render(frame, app, body_chunks[1]);
    } else {
        chat::render(frame, app, body_area);
    }

    // Input area.
    input::render(frame, app, input_area);
}
