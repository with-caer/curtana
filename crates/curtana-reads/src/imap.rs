use async_native_tls::TlsConnector;
use futures::TryStreamExt;
use mail_parser::MessageParser;
use serde::Deserialize;
use tokio::net::TcpStream;

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

pub async fn fetch_emails(config: &ImapConfig) -> Vec<EmailMessage> {
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
    let mut session = client
        .login(config.username.clone(), config.password.clone())
        .await
        .unwrap();

    // Request access to the inbox.
    session.select(&config.mailbox).await.unwrap();

    // Fetch messages in this mailbox, along with their RFC822 field.
    // RFC 822 dictates the format of the body of e-mails
    let messages_stream = session.fetch(&config.sequence, "RFC822").await.unwrap();
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

            let mut body = String::new();
            let text_bodies = m.text_body_count();
            for i in 0..text_bodies {
                let body_text = &m.body_text(i).unwrap();
                if !body_text.is_empty() {
                    body.push_str(body_text.trim());
                }
            }

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
    pub mailbox: String,
    /// IMAP sequence set to fetch, e.g. `"1:*"`, `"1:50"`.
    pub sequence: String,
}
