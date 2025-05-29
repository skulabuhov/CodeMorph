# CodeMorph

Simple Telegram bot using OpenAI and FAISS for context retrieval.

## Configuration

The bot is configured via the following environment variables:

- `OPENAI_API_KEY` - API key for OpenAI.
- `TELEGRAM_TOKEN` - Telegram bot token.
- `ALLOWED_USERNAMES` - comma-separated list of Telegram usernames allowed to use the bot. Defaults to `skulabuhov` when unset.
