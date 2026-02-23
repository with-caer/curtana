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
