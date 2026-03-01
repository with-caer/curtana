use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{App, DetailView};

/// Renders the right-side detail panel.
pub fn render(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " Detail ",
            Style::default().add_modifier(Modifier::BOLD),
        ));

    let content = match &app.detail_panel {
        Some(DetailView::TaxonomyList(entries)) => {
            let mut lines = vec![Line::from(Span::styled(
                "Taxonomies",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ))];
            lines.push(Line::from(""));

            for (name, entry) in entries {
                lines.push(Line::from(Span::styled(
                    name.clone(),
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                )));
                if !entry.description.is_empty() {
                    lines.push(Line::from(format!("  {}", entry.description)));
                }
                lines.push(Line::from(format!(
                    "  source: {} ({})",
                    entry.source_type, entry.source_id
                )));
                if let Some(ts) = entry.last_ingested_at {
                    lines.push(Line::from(format!("  last ingested: {ts}")));
                }
                lines.push(Line::from(""));
            }
            lines
        }
        Some(DetailView::Sources(sources)) => {
            let mut lines = vec![Line::from(Span::styled(
                "Sources",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ))];
            lines.push(Line::from(""));

            for source in sources {
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("[{}] ", source.index),
                        Style::default().fg(Color::Yellow),
                    ),
                    Span::raw(format!("{:.4} | {} | {}", source.score, source.taxonomy, source.title)),
                ]));
            }
            lines
        }
        None => {
            vec![Line::from(Span::styled(
                "Press Tab to toggle",
                Style::default().fg(Color::DarkGray),
            ))]
        }
    };

    let paragraph = Paragraph::new(content).block(block).wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}
