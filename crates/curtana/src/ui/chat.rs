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
                // 2-space indent to align with user "> " prefix.
                let indent = Span::raw("  ");
                let dim = Style::default().fg(Color::DarkGray);

                // Render persisted activity lines (from gathering phase).
                for activity in &message.activity {
                    lines.push(Line::from(Span::styled(format!("  {activity}"), dim)));
                }

                // Render live activity lines (still gathering).
                if message.content.is_empty() {
                    for activity in &app.activity_lines {
                        lines.push(Line::from(Span::styled(format!("  {activity}"), dim)));
                    }
                    // Streaming placeholder cursor.
                    lines.push(Line::from(vec![
                        indent,
                        Span::styled("\u{2588}", Style::default().fg(Color::DarkGray)),
                    ]));
                } else {
                    // Pre-wrap so every continuation line keeps the indent.
                    let content_width = (inner.width as usize).saturating_sub(2);
                    for line in markdown::to_lines(&message.content) {
                        for wrapped in wrap_spans(line.spans, content_width) {
                            let mut spans = vec![indent.clone()];
                            spans.extend(wrapped);
                            lines.push(Line::from(spans));
                        }
                    }
                }
            }
            Role::User => {
                let style = Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD);
                for (i, text_line) in message.content.lines().enumerate() {
                    // "> " on the first line, "  " on continuations to keep alignment.
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

/// Wrap styled spans to fit within `max_width` columns, breaking at word boundaries.
/// Returns one `Vec<Span>` per wrapped line, preserving span styles across breaks.
fn wrap_spans(spans: Vec<Span<'static>>, max_width: usize) -> Vec<Vec<Span<'static>>> {
    if max_width == 0 {
        return vec![spans];
    }

    // Flatten to (char, style) pairs.
    let flat: Vec<(char, Style)> = spans
        .iter()
        .flat_map(|s| s.content.chars().map(move |c| (c, s.style)))
        .collect();

    if flat.is_empty() {
        return vec![Vec::new()];
    }

    let mut result: Vec<Vec<Span<'static>>> = Vec::new();
    let mut start = 0;

    while start < flat.len() {
        let remaining = flat.len() - start;
        if remaining <= max_width {
            result.push(group_spans(&flat[start..]));
            break;
        }

        // Find the last space within max_width chars from start.
        let end = start + max_width;
        let break_at = flat[start..end]
            .iter()
            .rposition(|(c, _)| *c == ' ')
            .map(|p| start + p)
            .unwrap_or(end); // force break if no space found

        if break_at <= start {
            // No viable break point; force break at max_width.
            result.push(group_spans(&flat[start..end]));
            start = end;
            continue;
        }

        result.push(group_spans(&flat[start..break_at]));

        // Skip the space at the break point.
        start = break_at;
        if start < flat.len() && flat[start].0 == ' ' {
            start += 1;
        }
    }

    if result.is_empty() {
        result.push(Vec::new());
    }

    result
}

/// Group consecutive (char, style) pairs with the same style back into `Span`s.
fn group_spans(chars: &[(char, Style)]) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut text = String::new();
    let mut current_style: Option<Style> = None;

    for &(ch, style) in chars {
        if current_style == Some(style) {
            text.push(ch);
        } else {
            if let Some(s) = current_style {
                spans.push(Span::styled(std::mem::take(&mut text), s));
            }
            text.push(ch);
            current_style = Some(style);
        }
    }

    if let Some(s) = current_style
        && !text.is_empty()
    {
        spans.push(Span::styled(text, s));
    }

    spans
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
