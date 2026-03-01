mod app;
mod cli;
mod commands;
mod config;
mod event;
mod handler;
mod ui;

use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use app::App;
use commands::CommandRequest;
use config::Config;
use event::EventStream;

#[derive(Parser)]
#[command(name = "curtana", about = "Your AI concierge.")]
struct Cli {
    /// Path to the config file.
    #[arg(short, long, default_value = "Curtana.toml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Option<SubCmd>,
}

#[derive(Subcommand)]
enum SubCmd {
    /// Launch TUI and auto-start source discovery.
    Discover,
    /// Run the ingest pipeline headlessly.
    Ingest,
    /// Query your knowledge base headlessly.
    Query {
        /// The query string.
        query: String,
    },
}

#[tokio::main]
async fn main() -> io::Result<()> {
    let cli = Cli::parse();
    let config = Arc::new(Config::load(&cli.config));

    match cli.command {
        None => run_tui(config, None).await,
        Some(SubCmd::Discover) => run_tui(config, Some(CommandRequest::Discover)).await,
        Some(SubCmd::Ingest) => {
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .init();
            cli::ingest(&config).await;
            Ok(())
        }
        Some(SubCmd::Query { query }) => {
            cli::query(&config, &query).await;
            Ok(())
        }
    }
}

async fn run_tui(config: Arc<Config>, auto_command: Option<CommandRequest>) -> io::Result<()> {
    // Install a panic hook that restores the terminal before printing
    // the panic message, so it is readable.
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        original_hook(info);
    }));

    // Set up the terminal.
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create the event stream and command thread.
    let mut events = EventStream::new();
    let cmd_tx = commands::spawn_command_thread(config, events.tx());

    // Create the app.
    let mut app = App::new();

    // If an auto-command was requested, submit it immediately.
    if let Some(cmd) = auto_command {
        handler::submit_auto_command(&mut app, cmd, &cmd_tx);
    }

    // Main loop.
    while app.running {
        terminal.draw(|frame| ui::render(frame, &app))?;

        if let Some(event) = events.next().await {
            handler::handle_event(&mut app, event, &cmd_tx);
        } else {
            break;
        }
    }

    // Restore the terminal.
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}
