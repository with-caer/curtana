Curtana is your AI concierge.

The name comes from the Sword of Mercy: A ceremonial blade with a deliberately blunted tip, symbolizing power tempered by restraint. A metaphor for an agent that's deliberately limited for safety.

## Concepts

### Mitigates [the Lethal Trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)

The Lethal Trifecta is a dangerous combination of AI agent capabilities:

1. Access to private data.
2. Exposure to untrusted data.
3. Ability to take action.

Systems like OpenClaw have all three: They'll read email, process attacker-controlled content, and can send messages or run commands. 

This concierge breaks the trifecta by eliminating the ability to take action. It consists of three components:

1. A self-hostable, read-only agent that runs local LLM inference and RAG, but has no ability to take actions.
2. An external gateway that can receive action proposals from the agent, presenting them for human review.
3. An execution service that can execute approved actions, but has no agentic functionality.

### Normalizes Artifacts to Markdown

The read-only agent normalizes all artifacts (e.g., emails and PDFs) to Markdown text, which can then be content-aware chunked for embedding into a local vector index.

### Learns Taxonomies of Artifacts

The read-only agent retrieves artifacts and stores them in local vector databases organized by **taxonomy**. A taxonomy is a named, described collection of artifacts backed by its own database. Each taxonomy has:

- A **name** (e.g., `knowledge`, `historical`)
- A **description** that characterizes the kind of content it holds (e.g., "Reference material, how-tos, technical knowledge")
- One or more **sources** that feed artifacts into it (e.g., an IMAP folder)

#### Ingestion

Each unique data source is ingested into a dedicated taxonomy database. The system automatically prefixes the taxonomy name with the source type. For example, emails in a `knowledge` IMAP folder are ingested into an `imap-knowledge` database, and emails in a `historical` folder are ingested into an `imap-historical` database.

A future enhancement could enable hybrid assignment, where the agent uses LLM inference to reclassify or cross-file artifacts after ingestion.

#### Query Routing

When the agent receives a query, it uses the local LLM to classify the query's intent against the available taxonomy descriptions. The agent's context includes the name and description of each taxonomy, giving it enough information to route the query to the most relevant collection(s). A query may be routed to multiple taxonomies when it spans categories, with results merged and reranked.

For example:

- `"Tell me about the latest trends in AI."` routes to taxonomies related to historical knowledge.
- `"Tell me about encryption algorithms."` routes to taxonomies related to current knowledge.

#### Configuration

Taxonomies are declared in configuration, providing a name, description, and source for each:

```toml
[[taxonomy]]
name = "knowledge"
description = "Reference material, how-tos, technical knowledge"

[[taxonomy.source]]
type = "imap"
folder = "knowledge"

[[taxonomy]]
name = "historical"
description = "Time-sensitive news, trends, historical context"

[[taxonomy.source]]
type = "imap"
folder = "historical"
```

A future enhancement could enable the agent to automatically determine a name and description for a taxonomy by summarizing an arbitrary source, removing the need for manual description authoring.