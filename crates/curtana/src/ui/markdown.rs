use markdown::{to_mdast, ParseOptions};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Converts a markdown string into styled ratatui `Line`s.
pub fn to_lines(input: &str) -> Vec<Line<'static>> {
    let tree = to_mdast(input, &ParseOptions::default()).unwrap();
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
                let bar =
                    Span::styled("\u{258e} ", Style::default().fg(Color::DarkGray));
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
                self.collect_inline_spans(&[node.clone()], &mut spans);
                self.lines.push(Line::from(spans));
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
