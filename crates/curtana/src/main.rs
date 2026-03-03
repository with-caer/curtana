mod app;
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
use config::Config;
use event::EventStream;

#[derive(Parser)]
#[command(name = "curtana", about = "Your AI concierge.")]
struct Cli {
    /// Path to the config file.
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Download default models and create ~/.curtana/ config.
    Setup,
}

#[tokio::main]
async fn main() -> io::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Setup) => commands::setup::run().await,
        None => {
            let config_path = match config::resolve_config_path(cli.config.as_deref()) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };

            let config = match Config::load(&config_path) {
                Ok(c) => Arc::new(c),
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };

            run_tui(config).await
        }
    }
}

async fn run_tui(config: Arc<Config>) -> io::Result<()> {
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

    // Ensure the terminal is restored on all exit paths (error, break, etc.),
    // not just clean exits. The panic hook covers panics; this covers `?`.
    let result = run_event_loop(&mut terminal, config).await;

    // Restore the terminal.
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

async fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    config: Arc<Config>,
) -> io::Result<()> {
    // Create the event stream and command thread.
    let mut events = EventStream::new();
    let cmd_tx = commands::spawn_command_thread(config, events.tx());

    // Create the app.
    let mut app = App::new();

    // Main loop.
    while app.running {
        terminal.draw(|frame| ui::render(frame, &app))?;

        if let Some(event) = events.next().await {
            handler::handle_event(&mut app, event, &cmd_tx);
        } else {
            break;
        }
    }

    Ok(())
}
