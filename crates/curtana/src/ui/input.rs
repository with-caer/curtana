use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use crate::app::App;

/// Renders the text input area at the bottom of the screen.
pub fn render(frame: &mut Frame, app: &App, area: Rect) {
    let (display, style) = if app.input.is_empty() {
        (
            "> type a message or /command...".to_string(),
            Style::default().fg(Color::DarkGray),
        )
    } else {
        (format!("> {}", app.input), Style::default())
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let paragraph = Paragraph::new(display).style(style).block(block);
    frame.render_widget(paragraph, area);

    // Position the cursor when input is active.
    if !app.input.is_empty() || app.input.is_empty() {
        let prefix_width = 2u16; // "> "
        let cursor_offset = app.input[..app.cursor_position].chars().count() as u16;
        let cursor_x = area.x + 1 + prefix_width + cursor_offset;
        let cursor_y = area.y + 1;
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}
