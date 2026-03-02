use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Padding, Paragraph};

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
        .border_style(Style::default().fg(Color::DarkGray))
        .padding(Padding::horizontal(1));
    let inner = block.inner(area);

    let paragraph = Paragraph::new(display).style(style).block(block);
    frame.render_widget(paragraph, area);

    // Position the cursor inside the padded content area.
    let prefix_width = 2u16; // "> "
    let cursor_offset = app.input[..app.cursor_position].chars().count() as u16;
    frame.set_cursor_position((inner.x + prefix_width + cursor_offset, inner.y));

    // Render completion popup above the input area.
    if let Some(cs) = &app.completion {
        let item_count = cs.matches.len() as u16;
        // 2 for borders + 1 per item
        let popup_height = item_count + 2;
        // Don't overflow above the screen
        let popup_y = area.y.saturating_sub(popup_height);
        let actual_height = area.y - popup_y;

        let popup_width = area.width.min(40);
        let popup_area = Rect::new(area.x, popup_y, popup_width, actual_height);

        let items: Vec<ListItem> = cs
            .matches
            .iter()
            .enumerate()
            .map(|(i, cmd)| {
                let style = if i == cs.selected {
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                };
                let line = Line::from(vec![
                    Span::styled(format!("{:<12}", cmd.name), style),
                    Span::styled(
                        cmd.description,
                        style.fg(if i == cs.selected {
                            Color::Black
                        } else {
                            Color::DarkGray
                        }),
                    ),
                ]);
                ListItem::new(line)
            })
            .collect();

        let list = List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        );
        frame.render_widget(list, popup_area);
    }
}
