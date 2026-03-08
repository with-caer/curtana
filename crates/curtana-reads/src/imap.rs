use std::fmt;

use async_native_tls::TlsConnector;
use chrono::DateTime;
use futures::TryStreamExt;
use mail_parser::MessageParser;
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;
use tracing::warn;

use crate::ReadResult;

pub struct EmailMessage {
    /// Value of the `Message-ID` header.
    pub message_id: String,
    /// Sender email address from the `From` header.
    pub from: String,
    /// Message date as Unix epoch seconds.
    pub timestamp: i64,
    /// Subject line.
    pub subject: String,
    /// Concatenated text body parts.
    pub body: String,
}

pub struct ImapFolder {
    /// Folder name as returned by the server.
    pub name: String,
    /// Whether this folder can be selected (i.e. contains messages).
    pub is_selectable: bool,
}

/// Errors that can occur during IMAP operations.
#[derive(Debug)]
pub enum ImapError {
    ConnectionFailed(String),
    TlsFailed(String),
    AuthFailed(String),
    SessionError(String),
    FetchFailed(String),
    ParseFailed(String),
}

impl fmt::Display for ImapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImapError::ConnectionFailed(e) => write!(f, "IMAP connection failed: {e}"),
            ImapError::TlsFailed(e) => write!(f, "IMAP TLS upgrade failed: {e}"),
            ImapError::AuthFailed(e) => write!(f, "IMAP authentication failed: {e}"),
            ImapError::SessionError(e) => write!(f, "IMAP session error: {e}"),
            ImapError::FetchFailed(e) => write!(f, "IMAP fetch failed: {e}"),
            ImapError::ParseFailed(e) => write!(f, "IMAP message parse failed: {e}"),
        }
    }
}

/// RAII wrapper around an IMAP session that logs out on drop.
struct SessionGuard {
    session: Option<async_imap::Session<async_native_tls::TlsStream<TcpStream>>>,
}

impl SessionGuard {
    fn new(session: async_imap::Session<async_native_tls::TlsStream<TcpStream>>) -> Self {
        Self {
            session: Some(session),
        }
    }

    fn inner_mut(&mut self) -> &mut async_imap::Session<async_native_tls::TlsStream<TcpStream>> {
        self.session.as_mut().expect("session already consumed")
    }

    /// Explicitly log out, consuming the guard.
    async fn logout(mut self) {
        if let Some(mut session) = self.session.take() {
            let _ = session.logout().await;
        }
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        // Best-effort logout if not already consumed.
        // Can't do async in Drop, but we ensure the session is dropped.
        self.session.take();
    }
}

/// Establishes an authenticated IMAP session.
///
/// By default, uses implicit TLS (connect over TLS immediately, typically
/// port 993). When `config.starttls` is `true`, uses the STARTTLS upgrade
/// flow (typically port 143).
async fn imap_session(config: &ImapConfig) -> Result<SessionGuard, ImapError> {
    let mut tls = TlsConnector::new();
    if config.accept_invalid_certs {
        tls = tls
            .danger_accept_invalid_hostnames(true)
            .danger_accept_invalid_certs(true);
    }

    // Establish a TCP connection.
    let tcp_stream = TcpStream::connect((config.host.clone(), config.port))
        .await
        .map_err(|e| ImapError::ConnectionFailed(e.to_string()))?;

    let client = if config.starttls {
        // STARTTLS: plaintext greeting, then upgrade to TLS.
        let mut client = async_imap::Client::new(tcp_stream);
        let _greeting = client.read_response().await.ok_or_else(|| {
            ImapError::ConnectionFailed("unexpected end of stream, expected greeting".into())
        })?;
        client
            .run_command_and_check_ok("STARTTLS", None)
            .await
            .map_err(|e| ImapError::TlsFailed(e.to_string()))?;
        let stream = client.into_inner();
        let tls_stream = tls
            .connect(config.host.clone(), stream)
            .await
            .map_err(|e| ImapError::TlsFailed(e.to_string()))?;
        async_imap::Client::new(tls_stream)
    } else {
        // Implicit TLS: wrap in TLS immediately, then read greeting.
        let tls_stream = tls
            .connect(config.host.clone(), tcp_stream)
            .await
            .map_err(|e| ImapError::TlsFailed(e.to_string()))?;
        let mut client = async_imap::Client::new(tls_stream);
        let _greeting = client.read_response().await.ok_or_else(|| {
            ImapError::ConnectionFailed("unexpected end of stream, expected greeting".into())
        })?;
        client
    };

    // Authenticate with the IMAP server.
    let session = client
        .login(config.username.clone(), config.password.clone())
        .await
        .map_err(|e| ImapError::AuthFailed(e.0.to_string()))?;

    Ok(SessionGuard::new(session))
}

/// Lists all folders on the IMAP server. Read-only — does not select
/// any mailbox or fetch any messages.
pub async fn discover_folders(config: &ImapConfig) -> Result<Vec<ImapFolder>, ImapError> {
    let mut guard = imap_session(config).await?;
    let session = guard.inner_mut();

    let names = session
        .list(None, Some("*"))
        .await
        .map_err(|e| ImapError::SessionError(e.to_string()))?;
    let folders: Vec<_> = names
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| ImapError::SessionError(e.to_string()))?
        .into_iter()
        .map(|name| {
            let is_selectable = !name
                .attributes()
                .iter()
                .any(|attr| matches!(attr, async_imap::types::NameAttribute::NoSelect));

            ImapFolder {
                name: name.name().to_owned(),
                is_selectable,
            }
        })
        .collect();

    guard.logout().await;

    Ok(folders)
}

/// Fetches emails from an IMAP mailbox, optionally resuming from a cursor.
///
/// When `cursor` is `None`, performs a full fetch. When a valid cursor is
/// provided and the mailbox's UIDVALIDITY still matches, only fetches
/// messages with UIDs higher than the cursor's `max_uid`.
pub async fn fetch_emails(
    config: &ImapConfig,
    cursor: Option<&str>,
) -> Result<ReadResult<EmailMessage>, ImapError> {
    let mut guard = imap_session(config).await?;
    let session = guard.inner_mut();

    let mailbox_name = config.mailbox.as_deref().unwrap_or("INBOX");

    // Select the mailbox and capture UIDVALIDITY.
    let mailbox = session
        .select(mailbox_name)
        .await
        .map_err(|e| ImapError::SessionError(e.to_string()))?;

    let server_uid_validity = mailbox.uid_validity;

    // Decode existing cursor, if any.
    let prev_cursor: Option<ImapCursor> = cursor.and_then(|c| match serde_json::from_str(c) {
        Ok(parsed) => Some(parsed),
        Err(e) => {
            warn!("ignoring malformed IMAP cursor: {e}");
            None
        }
    });

    // Determine UID fetch range.
    let (uid_set, min_uid_filter) = if let Some(ref prev) = prev_cursor
        && Some(prev.uid_validity) == server_uid_validity
    {
        // Incremental: fetch UIDs above max_uid.
        // IMAP "n:*" always returns at least the message at n if it exists,
        // so we filter it out below.
        let start = prev.max_uid.saturating_add(1);
        (format!("{start}:*"), Some(prev.max_uid))
    } else {
        if prev_cursor.is_some() {
            warn!("UIDVALIDITY changed for {mailbox_name} — performing full re-fetch");
        }
        // Full fetch: all UIDs.
        ("1:*".to_string(), None)
    };

    // Fetch messages by UID.
    let messages_stream = session
        .uid_fetch(&uid_set, "RFC822")
        .await
        .map_err(|e| ImapError::FetchFailed(e.to_string()))?;
    let raw_messages: Vec<_> = messages_stream
        .try_collect()
        .await
        .map_err(|e| ImapError::FetchFailed(e.to_string()))?;

    // Track the highest UID seen for the next cursor.
    let mut max_uid_seen: Option<u32> = prev_cursor.as_ref().map(|c| c.max_uid);

    // Parse messages, filtering out the IMAP n:* quirk.
    let mut messages: Vec<EmailMessage> = Vec::new();
    for raw in &raw_messages {
        let uid = match raw.uid {
            Some(uid) => uid,
            None => {
                warn!("skipping fetch result with no UID");
                continue;
            }
        };

        // Filter out the message at max_uid (IMAP "n:*" quirk).
        if let Some(min) = min_uid_filter
            && uid <= min
        {
            continue;
        }

        if let Some(email) = parse_message(raw) {
            messages.push(email);
            max_uid_seen = Some(max_uid_seen.map_or(uid, |prev| prev.max(uid)));
        }
    }

    // Sort messages in descending order by date.
    messages.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    // Close session.
    guard.logout().await;

    // Build new cursor.
    let new_cursor = match (server_uid_validity, max_uid_seen) {
        (Some(uid_validity), Some(max_uid)) => {
            let c = ImapCursor {
                uid_validity,
                max_uid,
            };
            Some(serde_json::to_string(&c).expect("ImapCursor serialization cannot fail"))
        }
        _ => cursor.map(|c| c.to_string()),
    };

    Ok(ReadResult {
        items: messages,
        cursor: new_cursor,
    })
}

/// Parses a single IMAP fetch result into an `EmailMessage`, or `None`
/// if the message lacks required fields (body, Message-ID).
fn parse_message(raw: &async_imap::types::Fetch) -> Option<EmailMessage> {
    let body = match raw.body() {
        Some(b) => b,
        None => {
            warn!("skipping message with no body");
            return None;
        }
    };
    let parsed = match MessageParser::default().parse(body) {
        Some(m) => m.into_owned(),
        None => {
            warn!("skipping unparseable message");
            return None;
        }
    };

    let message_id = parsed.message_id().unwrap_or_default().to_owned();
    if message_id.is_empty() {
        warn!("skipping message with no Message-ID");
        return None;
    }

    let from = parsed
        .from()
        .and_then(|a| a.first())
        .and_then(|a| a.address())
        .map(|a| a.to_string())
        .unwrap_or_default();
    let timestamp = parsed.date().map(|d| d.to_timestamp()).unwrap_or(0);
    let subject = parsed.subject().unwrap_or_default().to_owned();

    let body = if parsed.html_body_count() > 0 {
        let mut html = String::new();
        for i in 0..parsed.html_body_count() {
            if let Some(part) = parsed.body_html(i) {
                html.push_str(&part);
            }
        }
        htmd::HtmlToMarkdown::builder()
            .skip_tags(vec!["style", "script"])
            .build()
            .convert(&html)
            .unwrap_or(html)
    } else {
        let mut text = String::new();
        for i in 0..parsed.text_body_count() {
            if let Some(part) = parsed.body_text(i) {
                let trimmed = part.trim();
                if !trimmed.is_empty() {
                    text.push_str(trimmed);
                }
            }
        }
        text
    };

    Some(EmailMessage {
        message_id,
        from,
        timestamp,
        subject,
        body,
    })
}

impl EmailMessage {
    pub fn to_markdown(&self) -> String {
        let date = DateTime::from_timestamp(self.timestamp, 0)
            .map(|dt| dt.to_rfc3339())
            .unwrap_or_default();

        format!(
            "# {}\n\n**From:** {}\n**Date:** {}\n\n{}",
            self.subject, self.from, date, self.body
        )
    }
}

/// Opaque cursor for incremental IMAP fetching.
/// Encodes the mailbox's UIDVALIDITY and the highest UID we've seen,
/// so the next fetch can start from `max_uid + 1`.
#[derive(Serialize, Deserialize)]
struct ImapCursor {
    uid_validity: u32,
    max_uid: u32,
}

#[derive(Deserialize)]
pub struct ImapConfig {
    /// IMAP server hostname.
    pub host: String,
    /// IMAP server port (typically 993 for implicit TLS, 143 for STARTTLS).
    pub port: u16,
    /// Login username.
    pub username: String,
    /// Login password.
    pub password: String,
    /// Mailbox to select, e.g. `"INBOX"`, `"[Gmail]/All Mail"`.
    /// Defaults to `"INBOX"` when `None`.
    pub mailbox: Option<String>,
    /// Use STARTTLS upgrade instead of implicit TLS.
    /// Defaults to `false` (implicit TLS on port 993).
    #[serde(default)]
    pub starttls: bool,
    /// Accept invalid TLS certificates and hostnames.
    /// **Only** enable for local/dev IMAP servers.
    /// Defaults to `false`.
    #[serde(default)]
    pub accept_invalid_certs: bool,
}
