[![FSL-1.1-MIT licensed](https://img.shields.io/badge/license-FSL--1.1--MIT-yellow.svg)](LICENSE.md)

Your AI concierge. Curtana reads your data (like emails) and answers questions about them using local LLMs. It _never_ writes data or takes actions without your opt-in consent.

## Quickstart

```sh
# Clone the repo.
git clone git@github.com:withcaer/curtana.git && cd curtana
git checkout caer/agent

# Install the `curtana` binary.
cargo install --path crates/curtana

# Run first-time setup.
curtana setup
```

`curtana setup` creates `~/.curtana/` and downloads the default chat (`Ministral-3-3B`) and embedding (`all-MiniLM`) models.

After setup runs, edit `~/.curtana/Curtana.toml` to add your data sources (e.g. IMAP email):

```toml
[[source]]
type = "imap"
host = "imap.example.com"
port = 993
username = "you@example.com"
password = "app-password"

# Change these to use different mailboxes or message ranges
# mailbox = "INBOX"
# sequence = "1:*"

# Uncomment these for local IMAP servers.
# accept_invalid_certs = true
# starttls = true
```

Run `curtana`. Within the UI, use `/explore` to scan and select sources to read from, `/read` to pull data from selected sources, then type any question.

## Crates

| Crate | Description |
|-------|-------------|
| [`curtana`](crates/curtana) | CLI and agent loop. |
| [`curtana-reads`](crates/curtana-reads) | Tools for reading from external data sources (iMessage, IMAP email). |
| [`curtana-knows`](crates/curtana-knows) | Persistent knowledge storage, embedding, and retrieval. |
| [`curtana-infers`](crates/curtana-infers) | Low-overhead local LLM inference via [`llama.cpp`](https://github.com/ggml-org/llama.cpp), supporting most `.gguf`-formatted chat and embedding models. |

## License

Copyright © 2025 - 2026 With Caer, LLC.

Licensed under the Functional Source License, Version 1.1, MIT Future License. Refer to [the license file](LICENSE.md) for more info.
