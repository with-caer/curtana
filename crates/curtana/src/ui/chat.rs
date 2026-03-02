use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding, Paragraph, Wrap};

use super::markdown;
use crate::app::{App, Role};

/// Renders the scrollable chat message history.
pub fn render(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .padding(Padding::horizontal(1));
    let inner = block.inner(area);

    let mut lines: Vec<Line> = Vec::new();

    // Sword art + welcome text always at the top of chat history.
    lines.extend(welcome_screen(inner.width));

    // Skip the first message (welcome text is integrated into the art).
    for message in app.messages.iter().skip(1) {
        // Add a blank line between messages (skip before the first).
        if !lines.is_empty() {
            lines.push(Line::from(""));
        }

        match message.role {
            Role::Assistant | Role::System => {
                if message.content.is_empty() {
                    if !app.activity_lines.is_empty() {
                        let dim = Style::default().fg(Color::DarkGray);
                        for activity in &app.activity_lines {
                            lines.push(Line::from(Span::styled(format!("  {activity}"), dim)));
                        }
                    }
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

    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });

    let line_count = paragraph.line_count(inner.width);
    let max_scroll = line_count.saturating_sub(inner.height as usize);
    let scroll = max_scroll.saturating_sub(app.scroll_offset);

    let paragraph = paragraph.scroll((scroll as u16, 0));
    frame.render_widget(paragraph, area);
}

/// Welcome screen: sword ASCII art on the left, help text wrapping on the right.
fn welcome_screen(inner_width: u16) -> Vec<Line<'static>> {
    let sword = Style::default().fg(Color::Cyan);
    let text_style = Style::default().fg(Color::DarkGray);

    const ART_WIDTH: usize = 10;
    const GAP: usize = 3;
    const TEXT_OFFSET: usize = 2; // Start text at the crossguard line.

    let art: &[&str] = &[
        "     O    ",
        "    | |   ",
        " >===O===<",
        "    | |   ",
        "    | |   ",
        "    | |   ",
        "    | |   ",
        "    | |   ",
        "    | |   ",
        "    | |   ",
        "    | |   ",
        "    ===   ",
    ];

    let text_width = (inner_width as usize).saturating_sub(ART_WIDTH + GAP);
    let welcome = "Welcome to curtana. Type a question to search, or /help for commands.";
    let wrapped = if text_width > 10 {
        word_wrap(welcome, text_width)
    } else {
        Vec::new()
    };

    let mut lines = vec![Line::from("")];
    for (i, art_line) in art.iter().enumerate() {
        let mut spans = vec![Span::styled(*art_line, sword)];
        if i >= TEXT_OFFSET
            && let Some(text) = wrapped.get(i - TEXT_OFFSET)
        {
            spans.push(Span::raw(" ".repeat(GAP)));
            spans.push(Span::styled(text.clone(), text_style));
        }
        lines.push(Line::from(spans));
    }
    lines
}

/// Word-wrap text to fit within `max_width` columns.
fn word_wrap(text: &str, max_width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if current.is_empty() {
            current = word.to_string();
        } else if current.len() + 1 + word.len() <= max_width {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(std::mem::take(&mut current));
            current = word.to_string();
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}
