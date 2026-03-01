use std::path::Path;

use curtana_knows::manifest::Manifest;
use tokio::sync::mpsc;

use crate::config::Config;
use crate::event::{CommandResult, Event};

/// Loads the manifest and reports taxonomy listings.
pub fn run(config: &Config, tx: &mpsc::UnboundedSender<Event>) {
    let data_dir = Path::new(config.data_dir());
    let manifest_path = data_dir.join("manifest.toml");

    let manifest = match Manifest::load(&manifest_path) {
        Ok(m) => m,
        Err(e) => {
            tx.send(Event::Error(format!("failed to load manifest: {e}")))
                .ok();
            return;
        }
    };

    if manifest.taxonomies.is_empty() {
        tx.send(Event::CommandDone(CommandResult::Message(
            "No taxonomies discovered yet. Run /discover to get started.".into(),
        )))
        .ok();
        return;
    }

    let entries: Vec<(String, _)> = manifest.taxonomies.into_iter().collect();

    tx.send(Event::CommandDone(CommandResult::Status { entries }))
        .ok();
}
