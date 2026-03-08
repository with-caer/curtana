use curtana_knows::manifest::Manifest;
use tokio::sync::mpsc;

use crate::config::Config;
use crate::event::{CommandResult, Event};

/// Loads the manifest and reports taxonomy listings.
pub fn run(config: &Config, tx: &mpsc::UnboundedSender<Event>) {
    let data_dir = config.data_dir();
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
            "No taxonomies found yet. Run /explore to get started.".into(),
        )))
        .ok();
        return;
    }

    let mut text = String::from("## Tracked taxonomies\n\n");
    for (name, entry) in &manifest.taxonomies {
        let short_desc = entry
            .description
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("");
        if short_desc.is_empty() {
            text.push_str(&format!("- **{name}**\n"));
        } else {
            text.push_str(&format!("- **{name}** \u{2014} {short_desc}\n"));
        }
    }

    tx.send(Event::CommandDone(CommandResult::Message(text)))
        .ok();
}
