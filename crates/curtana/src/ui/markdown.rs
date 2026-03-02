use markdown::{ParseOptions, to_mdast};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Converts a markdown string into styled ratatui `Line`s.
pub fn to_lines(input: &str) -> Vec<Line<'static>> {
    let tree = match to_mdast(input, &ParseOptions::gfm()) {
        Ok(tree) => tree,
        Err(_) => {
            // Markdown parse failed — return the input as plain text.
            return input
                .lines()
                .map(|line| Line::from(line.to_string()))
                .collect();
        }
    };
    let mut ctx = RenderContext::default();
    ctx.render_node(&tree);
    ctx.lines
}

#[derive(Default)]
struct RenderContext {
    lines: Vec<Line<'static>>,
    /// Style modifiers accumulated from ancestor inline nodes.
    style: Style,
    /// Current nesting depth inside lists (0 = top-level).
    list_depth: usize,
    /// Stack of list-ordering context: `Some(next_number)` for ordered, `None` for unordered.
    list_stack: Vec<Option<u32>>,
    /// Prefix spans to prepend to the first line of a block (used for list bullets, blockquote bars).
    line_prefix: Vec<Span<'static>>,
}

impl RenderContext {
    fn render_node(&mut self, node: &markdown::mdast::Node) {
        use markdown::mdast::Node;

        match node {
            Node::Root(root) => {
                for child in &root.children {
                    self.render_node(child);
                }
            }

            Node::Heading(heading) => {
                let heading_style = Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD);
                let prefix = "#".repeat(heading.depth as usize);
                let old_style = self.style;
                self.style = self.style.patch(heading_style);
                let mut spans = vec![Span::styled(format!("{prefix} "), self.style)];
                self.collect_inline_spans(&heading.children, &mut spans);
                self.lines.push(Line::from(spans));
                self.style = old_style;
            }

            Node::Paragraph(paragraph) => {
                let mut spans = Vec::new();
                // Prepend any accumulated prefix (list bullet, blockquote bar, etc.).
                spans.append(&mut self.line_prefix);
                self.collect_inline_spans(&paragraph.children, &mut spans);
                self.lines.push(Line::from(spans));
            }

            Node::Code(code) => {
                let code_style = Style::default().fg(Color::Green);
                let gutter = Span::styled("\u{2502} ", Style::default().fg(Color::DarkGray));
                for line in code.value.lines() {
                    self.lines.push(Line::from(vec![
                        gutter.clone(),
                        Span::styled(line.to_string(), code_style),
                    ]));
                }
                // Handle empty code blocks.
                if code.value.is_empty() {
                    self.lines.push(Line::from(gutter));
                }
            }

            Node::List(list) => {
                let start = if list.ordered { list.start } else { None };
                self.list_stack.push(start);
                self.list_depth += 1;
                for child in &list.children {
                    self.render_node(child);
                }
                self.list_depth -= 1;
                self.list_stack.pop();
            }

            Node::ListItem(item) => {
                let indent = "  ".repeat(self.list_depth);
                let bullet = if let Some(Some(num)) = self.list_stack.last_mut() {
                    let s = format!("{indent}{num}. ");
                    *num += 1;
                    s
                } else if self.list_depth > 1 {
                    format!("{indent}\u{25e6} ")
                } else {
                    format!("{indent}\u{2022} ")
                };
                self.line_prefix
                    .push(Span::styled(bullet, Style::default()));
                for child in &item.children {
                    self.render_node(child);
                }
                // Clear any unused prefix (shouldn't happen but be safe).
                self.line_prefix.clear();
            }

            Node::Blockquote(blockquote) => {
                let bar = Span::styled("\u{258e} ", Style::default().fg(Color::DarkGray));
                let old_style = self.style;
                self.style = self.style.patch(Style::default().fg(Color::DarkGray));
                // Render children, then prefix each produced line with the bar.
                let start = self.lines.len();
                for child in &blockquote.children {
                    self.render_node(child);
                }
                self.style = old_style;
                // Prefix every line that was just added.
                for line in &mut self.lines[start..] {
                    line.spans.insert(0, bar.clone());
                }
            }

            Node::ThematicBreak(_) => {
                // Full-width horizontal rule — we use 80 chars as a reasonable default;
                // ratatui will clip to the actual widget width.
                let rule = "\u{2500}".repeat(80);
                self.lines.push(Line::from(Span::styled(
                    rule,
                    Style::default().fg(Color::DarkGray),
                )));
            }

            // Inline-only nodes encountered at block level (shouldn't happen in well-formed
            // markdown, but handle gracefully).
            Node::Text(_)
            | Node::Strong(_)
            | Node::Emphasis(_)
            | Node::InlineCode(_)
            | Node::Link(_)
            | Node::Break(_) => {
                let mut spans = Vec::new();
                self.collect_inline_spans(std::slice::from_ref(node), &mut spans);
                self.lines.push(Line::from(spans));
            }

            Node::Table(table) => {
                let border_style = Style::default().fg(Color::DarkGray);
                let header_style = Style::default().add_modifier(Modifier::BOLD);

                // Collect each cell's spans and measure column widths.
                let mut grid: Vec<Vec<Vec<Span<'static>>>> = Vec::new();
                for row_node in &table.children {
                    let Node::TableRow(row) = row_node else {
                        continue;
                    };
                    let mut row_spans: Vec<Vec<Span<'static>>> = Vec::new();
                    for cell_node in &row.children {
                        let Node::TableCell(cell) = cell_node else {
                            continue;
                        };
                        let mut spans = Vec::new();
                        self.collect_inline_spans(&cell.children, &mut spans);
                        row_spans.push(spans);
                    }
                    grid.push(row_spans);
                }

                let num_cols = grid.iter().map(|r| r.len()).max().unwrap_or(0);
                if num_cols == 0 {
                    return;
                }

                // Compute max width per column.
                let mut col_widths = vec![0usize; num_cols];
                for row in &grid {
                    for (c, cell) in row.iter().enumerate() {
                        col_widths[c] = col_widths[c].max(spans_width(cell));
                    }
                }

                // Shrink columns to fit the available width.
                // The table is rendered inside a Block with Borders::ALL, so the
                // usable width is the terminal width minus 2 (left + right border).
                // Table chrome: "│ " + col0 + " │ " + col1 + … + " │"
                //             = 2 + sum(col_widths) + 3*(num_cols-1) + 2
                let term_width = crossterm::terminal::size()
                    .map(|(w, _)| w as usize)
                    .unwrap_or(80);
                let available = term_width.saturating_sub(2);
                let chrome = 2 + 3 * num_cols.saturating_sub(1) + 2;
                let content_budget = available.saturating_sub(chrome);
                let total_content: usize = col_widths.iter().sum();
                if total_content > content_budget {
                    shrink_columns(&mut col_widths, content_budget);
                }

                // Build border lines.
                let top = Line::from(Span::styled(
                    format!(
                        "\u{250c}{}\u{2510}",
                        col_widths
                            .iter()
                            .map(|w| "\u{2500}".repeat(w + 2))
                            .collect::<Vec<_>>()
                            .join("\u{252c}")
                    ),
                    border_style,
                ));
                let mid = Line::from(Span::styled(
                    format!(
                        "\u{251c}{}\u{2524}",
                        col_widths
                            .iter()
                            .map(|w| "\u{2500}".repeat(w + 2))
                            .collect::<Vec<_>>()
                            .join("\u{253c}")
                    ),
                    border_style,
                ));
                let bot = Line::from(Span::styled(
                    format!(
                        "\u{2514}{}\u{2518}",
                        col_widths
                            .iter()
                            .map(|w| "\u{2500}".repeat(w + 2))
                            .collect::<Vec<_>>()
                            .join("\u{2534}")
                    ),
                    border_style,
                ));

                let num_rows = grid.len();
                self.lines.push(top);
                for (r, row) in grid.into_iter().enumerate() {
                    let mut line_spans: Vec<Span<'static>> = Vec::new();
                    line_spans.push(Span::styled("\u{2502} ", border_style));
                    for (c, col_w) in col_widths.iter().enumerate() {
                        let cell = row.get(c).cloned().unwrap_or_default();
                        let align = table
                            .align
                            .get(c)
                            .copied()
                            .unwrap_or(markdown::mdast::AlignKind::None);
                        let mut padded = pad_spans(
                            cell,
                            *col_w,
                            align,
                            if r == 0 {
                                header_style
                            } else {
                                Style::default()
                            },
                        );
                        line_spans.append(&mut padded);
                        if c + 1 < num_cols {
                            line_spans.push(Span::styled(" \u{2502} ", border_style));
                        }
                    }
                    line_spans.push(Span::styled(" \u{2502}", border_style));
                    self.lines.push(Line::from(line_spans));
                    if r == 0 && num_rows > 1 {
                        self.lines.push(mid.clone());
                    }
                }
                self.lines.push(bot);
            }

            // Anything else we don't explicitly handle — recurse into children or ignore.
            other => {
                if let Some(children) = other.children() {
                    for child in children {
                        self.render_node(child);
                    }
                }
            }
        }
    }

    /// Recursively collects inline nodes into a flat `Vec<Span>`.
    fn collect_inline_spans(
        &mut self,
        nodes: &[markdown::mdast::Node],
        spans: &mut Vec<Span<'static>>,
    ) {
        use markdown::mdast::Node;

        for node in nodes {
            match node {
                Node::Text(text) => {
                    spans.push(Span::styled(text.value.clone(), self.style));
                }

                Node::Strong(strong) => {
                    let old = self.style;
                    self.style = self.style.add_modifier(Modifier::BOLD);
                    self.collect_inline_spans(&strong.children, spans);
                    self.style = old;
                }

                Node::Emphasis(emphasis) => {
                    let old = self.style;
                    self.style = self.style.add_modifier(Modifier::ITALIC);
                    self.collect_inline_spans(&emphasis.children, spans);
                    self.style = old;
                }

                Node::InlineCode(code) => {
                    spans.push(Span::styled(
                        code.value.clone(),
                        self.style.patch(Style::default().fg(Color::Magenta)),
                    ));
                }

                Node::Link(link) => {
                    let link_style = self.style.patch(
                        Style::default()
                            .fg(Color::Blue)
                            .add_modifier(Modifier::UNDERLINED),
                    );
                    let old = self.style;
                    self.style = link_style;
                    self.collect_inline_spans(&link.children, spans);
                    self.style = old;
                }

                Node::Break(_) => {
                    // Soft/hard line break inside a paragraph — we just add a space.
                    spans.push(Span::styled(" ", self.style));
                }

                // For any other node that might appear inline, try to recurse.
                other => {
                    if let Some(children) = other.children() {
                        self.collect_inline_spans(children, spans);
                    }
                }
            }
        }
    }
}

/// Sum the display widths of a span slice.
fn spans_width(spans: &[Span<'_>]) -> usize {
    spans.iter().map(|s| s.width()).sum()
}

/// Pad (or truncate) cell spans to `target_width` according to `align`, applying `style` to padding.
fn pad_spans(
    spans: Vec<Span<'static>>,
    target_width: usize,
    align: markdown::mdast::AlignKind,
    style: Style,
) -> Vec<Span<'static>> {
    use markdown::mdast::AlignKind;

    let spans = truncate_spans(spans, target_width);
    let current = spans_width(&spans);
    let pad = target_width.saturating_sub(current);
    match align {
        AlignKind::Right => {
            let mut out = vec![Span::styled(" ".repeat(pad), style)];
            out.extend(spans);
            out
        }
        AlignKind::Center => {
            let left = pad / 2;
            let right = pad - left;
            let mut out = vec![Span::styled(" ".repeat(left), style)];
            out.extend(spans);
            out.push(Span::styled(" ".repeat(right), style));
            out
        }
        // Left and None both left-align.
        _ => {
            let mut out = spans;
            out.push(Span::styled(" ".repeat(pad), style));
            out
        }
    }
}

/// Truncate a span list to fit within `max_width` display columns, appending "…" if truncated.
fn truncate_spans(spans: Vec<Span<'static>>, max_width: usize) -> Vec<Span<'static>> {
    let total = spans_width(&spans);
    if total <= max_width {
        return spans;
    }
    // Reserve 1 column for the ellipsis.
    let budget = max_width.saturating_sub(1);
    let mut out = Vec::new();
    let mut used = 0;
    for span in spans {
        let w = span.width();
        if used + w <= budget {
            out.push(span);
            used += w;
        } else {
            // Partially include this span.
            let remaining = budget - used;
            if remaining > 0 {
                let truncated: String = span.content.chars().take(remaining).collect();
                out.push(Span::styled(truncated, span.style));
            }
            break;
        }
    }
    // Append ellipsis using the style of the last span, or default.
    let ellipsis_style = out.last().map(|s| s.style).unwrap_or_default();
    out.push(Span::styled("\u{2026}", ellipsis_style));
    out
}

/// Shrink column widths so their sum fits within `budget`.
/// Repeatedly reduces the widest column by 1 until the total fits.
fn shrink_columns(widths: &mut [usize], budget: usize) {
    let mut total: usize = widths.iter().sum();
    while total > budget {
        // Find the widest column (first one wins ties).
        let (max_idx, &max_w) = widths.iter().enumerate().max_by_key(|(_, w)| **w).unwrap();
        if max_w == 0 {
            break;
        }
        // Shrink widest to at most the second-widest (or budget-share), whichever is larger.
        let second = widths
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != max_idx)
            .map(|(_, w)| *w)
            .max()
            .unwrap_or(0);
        let overshoot = total - budget;
        let shrink_to = max_w.saturating_sub(overshoot).max(second).max(1);
        let removed = max_w - shrink_to;
        widths[max_idx] = shrink_to;
        total -= removed;
    }
}
