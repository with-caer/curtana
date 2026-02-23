use async_native_tls::TlsConnector;
use futures::TryStreamExt;
use mail_parser::MessageParser;
use tokio::net::TcpStream;

use crate::ImapConfig;

pub async fn fetch_emails(config: &ImapConfig) -> Vec<(String, String)> {
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
    session.select("INBOX").await.unwrap();

    // Fetch message number 1 in this mailbox, along with its RFC822 field.
    // RFC 822 dictates the format of the body of e-mails
    let messages_stream = session.fetch("1:*", "RFC822").await.unwrap();
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

    // Extract message headers and bodies.
    let messages: Vec<_> = messages
        .into_iter()
        .map(|m| {
            let subject = m.subject().unwrap();
            let mut body = String::new();

            let text_bodies = m.text_body_count();
            for i in 0..text_bodies {
                let body_text = &m.body_text(i).unwrap();
                if !body_text.is_empty() {
                    body.push_str(body_text.trim());
                }
            }

            (subject.to_owned(), body)
        })
        .collect();

    // Close session.
    session.logout().await.unwrap();

    messages
}
