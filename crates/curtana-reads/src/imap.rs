use async_native_tls::TlsConnector;
use chrono::DateTime;
use futures::TryStreamExt;
use mail_parser::MessageParser;
use serde::Deserialize;
use tokio::net::TcpStream;

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

/// Establishes an authenticated IMAP session over STARTTLS.
async fn imap_session(
    config: &ImapConfig,
) -> async_imap::Session<async_native_tls::TlsStream<TcpStream>> {
    // Establish a TCP connection.
    let tcp_stream = TcpStream::connect((config.host.clone(), config.port))
        .await
        .unwrap();

    // Establish an IMAP connection.
    let mut client = async_imap::Client::new(tcp_stream);
    let _greeting = client
        .read_response()
        .await
        .expect("unexpected end of stream, expected greeting");

    // Upgrade to a TLS connection.
    client
        .run_command_and_check_ok("STARTTLS", None)
        .await
        .unwrap();
    let stream = client.into_inner();
    let tls = TlsConnector::new()
        .danger_accept_invalid_hostnames(true)
        .danger_accept_invalid_certs(true);
    let tls_stream = tls.connect(config.host.clone(), stream).await.unwrap();
    let client = async_imap::Client::new(tls_stream);

    // Authenticate with the IMAP server.
    client
        .login(config.username.clone(), config.password.clone())
        .await
        .unwrap()
}

/// Lists all folders on the IMAP server. Read-only — does not select
/// any mailbox or fetch any messages.
pub async fn discover_folders(config: &ImapConfig) -> Vec<ImapFolder> {
    let mut session = imap_session(config).await;

    let names = session.list(None, Some("*")).await.unwrap();
    let folders: Vec<_> = names
        .try_collect::<Vec<_>>()
        .await
        .unwrap()
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

    session.logout().await.unwrap();

    folders
}

pub async fn fetch_emails(config: &ImapConfig) -> Vec<EmailMessage> {
    let mut session = imap_session(config).await;

    let mailbox = config.mailbox.as_deref().unwrap_or("INBOX");
    let sequence = config.sequence.as_deref().unwrap_or("1:*");

    // Request access to the mailbox.
    session.select(mailbox).await.unwrap();

    // Fetch messages in this mailbox, along with their RFC822 field.
    // RFC 822 dictates the format of the body of e-mails
    let messages_stream = session.fetch(sequence, "RFC822").await.unwrap();
    let messages: Vec<_> = messages_stream.try_collect().await.unwrap();

    // Parse messages.
    let mut messages: Vec<_> = messages
        .into_iter()
        .map(|m| {
            let body = m.body().expect("message did not have a body!");
            let message = MessageParser::default().parse(body).unwrap();
            message.into_owned()
        })
        .collect();

    // Sort messages in descending order by date, similar to default inbox sorting.
    messages.sort_by(|a, b| {
        let a = a.date().unwrap();
        let b = b.date().unwrap();
        a.cmp(b).reverse()
    });

    // Extract message metadata and bodies.
    let messages: Vec<_> = messages
        .into_iter()
        .map(|m| {
            let message_id = m.message_id().unwrap().to_owned();
            let from = m
                .from()
                .and_then(|a| a.first())
                .and_then(|a| a.address())
                .map(|a| a.to_string())
                .unwrap_or_default();
            let timestamp = m.date().unwrap().to_timestamp();
            let subject = m.subject().unwrap().to_owned();

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

            EmailMessage {
                message_id,
                from,
                timestamp,
                subject,
                body,
            }
        })
        .collect();

    // Close session.
    session.logout().await.unwrap();

    messages
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
}
