use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use super::markdown;
use crate::app::{ActivePane, App, Role};

/// Renders the scrollable chat message history.
pub fn render(frame: &mut Frame, app: &App, area: Rect) {
    let mut lines: Vec<Line> = Vec::new();

    for message in &app.messages {
        // Add a blank line between messages (skip before the first).
        if !lines.is_empty() {
            lines.push(Line::from(""));
        }

        match message.role {
            Role::Assistant | Role::System => {
                if message.content.is_empty() {
                    // Streaming placeholder cursor.
                    lines.push(Line::from(Span::styled(
                        "\u{2588}",
                        Style::default().fg(Color::DarkGray),
                    )));
                } else {
                    lines.extend(markdown::to_lines(&message.content));
                }
            }
            Role::User => {
                let style = Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD);
                for (i, text_line) in message.content.lines().enumerate() {
                    let prefix = if i == 0 { "> " } else { "  " };
                    lines.push(Line::from(vec![
                        Span::styled(prefix, style),
                        Span::styled(text_line.to_string(), style),
                    ]));
                }
            }
        }
    }

    let focused = app.active_pane == ActivePane::Chat || app.detail_panel.is_none();
    let border_color = if focused {
        Color::Cyan
    } else {
        Color::DarkGray
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color));
    let inner_width = area.width.saturating_sub(2);
    let inner_height = area.height.saturating_sub(2) as usize;

    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });

    let line_count = paragraph.line_count(inner_width);
    let max_scroll = line_count.saturating_sub(inner_height);
    let scroll = max_scroll.saturating_sub(app.scroll_offset);

    let paragraph = paragraph.scroll((scroll as u16, 0));
    frame.render_widget(paragraph, area);
}
