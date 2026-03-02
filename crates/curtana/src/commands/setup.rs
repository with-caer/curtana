use std::io;
use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use tokio::io::AsyncWriteExt;

pub const DEFAULT_CHAT_MODEL_FILENAME: &str = "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf";
pub const DEFAULT_EMBED_MODEL_FILENAME: &str = "all-MiniLM-L6-v2-ggml-model-f16.gguf";

const CHAT_MODEL_URL: &str = "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf";
const EMBED_MODEL_URL: &str = "https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-ggml-model-f16.gguf";

const DEFAULT_CONFIG_CONTENT: &str = r#"# Curtana configuration
# See https://github.com/withcaer/curtana for documentation.

# Uncomment and set these to override the default models:
# chat_model = "~/.curtana/models/your-chat-model.gguf"
# embed_model = "~/.curtana/models/your-embed-model.gguf"

# Add data sources below:
# [[source]]
# type = "imap"
# host = "imap.example.com"
# port = 993
# username = "you@example.com"
# password = "app-password"
## Change these to use different mailboxes or message ranges
# mailbox = "INBOX"
# sequence = "1:*"
## Uncomment these for local IMAP servers.
# accept_invalid_certs = true
# starttls = true
"#;

/// Maximum number of resume attempts per download.
const MAX_RETRIES: u32 = 3;

/// Returns the path to `~/.curtana/`.
pub fn home_curtana_dir() -> io::Result<PathBuf> {
    let home = home::home_dir().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "could not determine home directory",
        )
    })?;
    Ok(home.join(".curtana"))
}

/// Runs the setup command: creates `~/.curtana/`, writes a default config,
/// and downloads the default chat and embedding models.
pub async fn run() -> io::Result<()> {
    let base = home_curtana_dir()?;
    let models_dir = base.join("models");
    let config_path = base.join("Curtana.toml");

    // Create directories.
    std::fs::create_dir_all(&models_dir)?;

    // Write default config (skip if exists).
    if config_path.exists() {
        eprintln!("{} already exists, skipping.", config_path.display());
    } else {
        std::fs::write(&config_path, DEFAULT_CONFIG_CONTENT)?;
        eprintln!("Wrote {}.", config_path.display());
    }

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .read_timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| io::Error::other(format!("failed to create HTTP client: {e}")))?;

    // Download models (skip if exists).
    let chat_dest = models_dir.join(DEFAULT_CHAT_MODEL_FILENAME);
    if chat_dest.exists() {
        eprintln!("{} already exists, skipping.", chat_dest.display());
    } else {
        eprintln!("Downloading chat model...");
        download_model(&client, CHAT_MODEL_URL, &chat_dest).await?;
    }

    let embed_dest = models_dir.join(DEFAULT_EMBED_MODEL_FILENAME);
    if embed_dest.exists() {
        eprintln!("{} already exists, skipping.", embed_dest.display());
    } else {
        eprintln!("Downloading embedding model...");
        download_model(&client, EMBED_MODEL_URL, &embed_dest).await?;
    }

    eprintln!("\nSetup complete. Run `curtana` to start.");
    Ok(())
}

/// Downloads a file from `url` to `dest` with a progress bar on stderr.
/// Writes to a `.part` temp file and renames on completion. Resumes
/// partial downloads using HTTP Range requests, retrying up to
/// `MAX_RETRIES` times on transient errors.
async fn download_model(client: &reqwest::Client, url: &str, dest: &Path) -> io::Result<()> {
    let part_path = dest.with_extension("gguf.part");

    for attempt in 0..=MAX_RETRIES {
        match download_range(client, url, &part_path).await {
            Ok(()) => {
                tokio::fs::rename(&part_path, dest).await?;
                eprintln!("Saved {}.", dest.display());
                return Ok(());
            }
            Err(e) if attempt < MAX_RETRIES && e.kind() != io::ErrorKind::InvalidInput => {
                eprintln!("Download interrupted: {e}");
                eprintln!("Resuming (attempt {}/{MAX_RETRIES})...", attempt + 1);
            }
            Err(e) => {
                // Clean up partial file on final failure.
                let _ = tokio::fs::remove_file(&part_path).await;
                return Err(e);
            }
        }
    }

    unreachable!()
}

/// Performs a single download pass, resuming from any existing `.part` file.
async fn download_range(client: &reqwest::Client, url: &str, part_path: &Path) -> io::Result<()> {
    // Check how much we already have on disk.
    let existing_len = match tokio::fs::metadata(part_path).await {
        Ok(m) => m.len(),
        Err(_) => 0,
    };

    let mut request = client.get(url);
    if existing_len > 0 {
        request = request.header("Range", format!("bytes={existing_len}-"));
    }

    let response = request
        .send()
        .await
        .map_err(|e| io::Error::other(format!("download failed: {e}")))?
        .error_for_status()
        .map_err(|e| {
            // Mark client errors (4xx) as InvalidInput so the retry loop
            // can distinguish them from transient stream errors.
            let kind = if e.status().is_some_and(|s| s.is_client_error()) {
                io::ErrorKind::InvalidInput
            } else {
                io::ErrorKind::Other
            };
            io::Error::new(kind, format!("download failed: {e}"))
        })?;

    let status = response.status();

    // Determine total file size and whether the server accepted our range.
    let (total_size, resumed) = if status == reqwest::StatusCode::PARTIAL_CONTENT {
        // Server accepted the range — total size is existing + remaining.
        let remaining = response.content_length().unwrap_or(0);
        (existing_len + remaining, true)
    } else {
        // Full response (200) — server ignored or doesn't support Range.
        (response.content_length().unwrap_or(0), false)
    };

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    // Open file for append (resume) or create (fresh).
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(!resumed)
        .append(resumed)
        .open(part_path)
        .await?;

    if resumed {
        pb.set_position(existing_len);
    }

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| io::Error::other(format!("download error: {e}")))?;
        file.write_all(&chunk).await?;
        pb.inc(chunk.len() as u64);
    }

    file.flush().await?;
    drop(file);

    pb.finish_with_message("done");
    Ok(())
}
