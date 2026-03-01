use std::fmt;

use async_native_tls::TlsConnector;
use chrono::DateTime;
use futures::TryStreamExt;
use mail_parser::MessageParser;
use serde::Deserialize;
use tokio::net::TcpStream;
use tracing::warn;

use crate::ToMarkdown;

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
    /// Hierarchy delimiter (e.g. `"/"` or `"."`).
    pub delimiter: Option<String>,
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

/// Establishes an authenticated IMAP session over STARTTLS.
async fn imap_session(config: &ImapConfig) -> Result<SessionGuard, ImapError> {
    // Establish a TCP connection.
    let tcp_stream = TcpStream::connect((config.host.clone(), config.port))
        .await
        .map_err(|e| ImapError::ConnectionFailed(e.to_string()))?;

    // Establish an IMAP connection.
    let mut client = async_imap::Client::new(tcp_stream);
    let _greeting = client.read_response().await.ok_or_else(|| {
        ImapError::ConnectionFailed("unexpected end of stream, expected greeting".into())
    })?;

    // Upgrade to a TLS connection.
    client
        .run_command_and_check_ok("STARTTLS", None)
        .await
        .map_err(|e| ImapError::TlsFailed(e.to_string()))?;
    let stream = client.into_inner();
    let mut tls = TlsConnector::new();
    if config.accept_invalid_certs {
        tls = tls
            .danger_accept_invalid_hostnames(true)
            .danger_accept_invalid_certs(true);
    }
    let tls_stream = tls
        .connect(config.host.clone(), stream)
        .await
        .map_err(|e| ImapError::TlsFailed(e.to_string()))?;
    let client = async_imap::Client::new(tls_stream);

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
                delimiter: name.delimiter().map(|d| d.to_string()),
                is_selectable,
            }
        })
        .collect();

    guard.logout().await;

    Ok(folders)
}

pub async fn fetch_emails(config: &ImapConfig) -> Result<Vec<EmailMessage>, ImapError> {
    let mut guard = imap_session(config).await?;
    let session = guard.inner_mut();

    let mailbox = config.mailbox.as_deref().unwrap_or("INBOX");
    let sequence = config.sequence.as_deref().unwrap_or("1:*");

    // Request access to the mailbox.
    session
        .select(mailbox)
        .await
        .map_err(|e| ImapError::SessionError(e.to_string()))?;

    // Fetch messages in this mailbox, along with their RFC822 field.
    let messages_stream = session
        .fetch(sequence, "RFC822")
        .await
        .map_err(|e| ImapError::FetchFailed(e.to_string()))?;
    let raw_messages: Vec<_> = messages_stream
        .try_collect()
        .await
        .map_err(|e| ImapError::FetchFailed(e.to_string()))?;

    // Parse messages, skipping any that fail to parse.
    let mut messages: Vec<_> = raw_messages
        .into_iter()
        .filter_map(|m| {
            let body = match m.body() {
                Some(b) => b,
                None => {
                    warn!("skipping message with no body");
                    return None;
                }
            };
            match MessageParser::default().parse(body) {
                Some(message) => Some(message.into_owned()),
                None => {
                    warn!("skipping unparseable message");
                    None
                }
            }
        })
        .collect();

    // Sort messages in descending order by date, similar to default inbox sorting.
    messages.sort_by(|a, b| {
        let a_date = a.date();
        let b_date = b.date();
        match (a_date, b_date) {
            (Some(a), Some(b)) => a.cmp(b).reverse(),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        }
    });

    // Extract message metadata and bodies, skipping messages without required fields.
    let messages: Vec<_> = messages
        .into_iter()
        .filter_map(|m| {
            let message_id = m.message_id().unwrap_or_default().to_owned();
            if message_id.is_empty() {
                warn!("skipping message with no Message-ID");
                return None;
            }
            let from = m
                .from()
                .and_then(|a| a.first())
                .and_then(|a| a.address())
                .map(|a| a.to_string())
                .unwrap_or_default();
            let timestamp = m.date().map(|d| d.to_timestamp()).unwrap_or(0);
            let subject = m.subject().unwrap_or_default().to_owned();

            let body = if m.html_body_count() > 0 {
                let mut html = String::new();
                for i in 0..m.html_body_count() {
                    if let Some(part) = m.body_html(i) {
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
                for i in 0..m.text_body_count() {
                    if let Some(part) = m.body_text(i) {
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
        })
        .collect();

    // Close session.
    guard.logout().await;

    Ok(messages)
}

impl ToMarkdown for EmailMessage {
    fn to_markdown(&self) -> String {
        let date = DateTime::from_timestamp(self.timestamp, 0)
            .map(|dt| dt.to_rfc3339())
            .unwrap_or_default();

        format!(
            "# {}\n\n**From:** {}\n**Date:** {}\n\n{}",
            self.subject, self.from, date, self.body
        )
    }
}

#[derive(Deserialize)]
pub struct ImapConfig {
    /// IMAP server hostname.
    pub host: String,
    /// IMAP server port (typically 143 for STARTTLS).
    pub port: u16,
    /// Login username.
    pub username: String,
    /// Login password.
    pub password: String,
    /// Mailbox to select, e.g. `"INBOX"`, `"[Gmail]/All Mail"`.
    /// Defaults to `"INBOX"` when `None`.
    pub mailbox: Option<String>,
    /// IMAP sequence set to fetch, e.g. `"1:*"`, `"1:50"`.
    /// Defaults to `"1:*"` when `None`.
    pub sequence: Option<String>,
    /// Accept invalid TLS certificates and hostnames.
    /// **Only** enable for local/dev IMAP servers.
    /// Defaults to `false`.
    #[serde(default)]
    pub accept_invalid_certs: bool,
}
