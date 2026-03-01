mod config;
mod discover;
mod ingest;
mod query;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use config::Config;

#[derive(Parser)]
#[command(name = "curtana", about = "Your AI concierge.")]
struct Cli {
    /// Path to config file.
    #[arg(short, long, default_value = "Curtana.toml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Discover available folders from configured sources.
    Discover,
    /// Ingest artifacts from discovered taxonomies.
    Ingest,
    /// Query across taxonomy stores.
    Query {
        /// The search query.
        query: String,
    },
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .without_time()
        .with_target(false)
        .init();

    let cli = Cli::parse();
    let config = Config::load(&cli.config);

    match cli.command {
        Command::Discover => discover::run(&config).await,
        Command::Ingest => ingest::run(&config).await,
        Command::Query { query } => query::run(&config, &query).await,
    }
}
