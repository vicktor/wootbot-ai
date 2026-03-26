# WootBot AI - Open Source AI Agent for Chatwoot Community Edition

An open-source AI-powered support bot for Chatwoot that provides RAG-based automatic responses using your knowledge base, documents, and past conversations. Works with Chatwoot Community Edition — no Enterprise license required.

## Features

- **RAG-powered responses** from documents, URLs, and past tickets
- **LLM-agnostic**: Gemini, Claude, OpenAI, or any OpenAI-compatible endpoint
- **Auto language detection** — responds in whatever language the customer writes (es, ca, en, pt, fr, de...)
- **Smart handoff** to human agents when confidence is low (handoff message in customer's language)
- **CSAT tracking** with response quality feedback
- **Document ingestion** from PDFs, URLs, and plain text
- **Conversation memory** for context-aware responses
- **pgvector** for semantic search (same DB as Chatwoot)

## Architecture

```
Customer → Chatwoot Widget → Chatwoot Server
                                    ↓ webhook
                              WootBot AI (FastAPI)
                              ├── RAG Search (pgvector)
                              ├── LLM Generation (Gemini/Claude)
                              └── Chatwoot API → Response to customer
```

## Quick Start

```bash
cp .env.example .env
# Edit .env with your config
pip install -r requirements.txt
python -m app.ingest  # Ingest your docs
uvicorn app.main:app --host 0.0.0.0 --port 8200
```

## Register the Bot in Chatwoot

### Option A: From the Chatwoot UI (v4+)

1. Go to **Settings → Integrations → Bots**
2. Click **Add Bot**
3. Fill in:
   - **Name**: WootBot AI
   - **Webhook URL**: `http://127.0.0.1:8200/webhook` (or your server IP/domain)
4. Save → copy the **Bot Token** into your `.env` as `CHATWOOT_BOT_TOKEN`
5. Go to the **Inbox** where you want the bot → **Bot Configuration** → select **WootBot AI**

### Option B: Via Rails Console (if UI doesn't show Bots)

```bash
# Connect to Chatwoot rails console
sudo -i -u chatwoot bash -l -c 'cd /home/chatwoot/chatwoot && RAILS_ENV=production bundle exec rails console'
```

Then run:
```ruby
# Create the bot (change the URL to your WootBot address)
bot = AgentBot.create!(
  name: "WootBot AI",
  outgoing_url: "http://127.0.0.1:8200/webhook",
  account_id: 1
)

# Get the token — save this for your .env
puts "BOT TOKEN: #{bot.access_token.token}"

# Connect bot to your inbox (replace Inbox.first with your target inbox)
AgentBotInbox.create!(inbox: Inbox.first, agent_bot: bot)

puts "Bot connected to inbox: #{Inbox.first.name}"
```

Copy the token and put it in your `.env`:
```
CHATWOOT_BOT_TOKEN=your_token_here
```

Then restart WootBot:
```bash
sudo systemctl restart wootbot
```

### Verify it works

1. Open your Chatwoot widget and send a message
2. The bot should respond automatically
3. Check logs: `sudo journalctl -u wootbot -f`
4. Check stats: `curl http://127.0.0.1:8200/admin/stats`

## Ingest Knowledge Base

```bash
# Ingest a website/docs
curl -X POST http://127.0.0.1:8200/admin/ingest/url \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://listen.doctor/help", "title": "Help Center"}'

# Ingest plain text
curl -X POST http://127.0.0.1:8200/admin/ingest/text \
  -H 'Content-Type: application/json' \
  -d '{"content": "Your FAQ text here...", "source": "faq", "title": "FAQ"}'

# Ingest PDF (from CLI)
cd /home/chatwoot/wootbot
.venv/bin/python -m app.rag.ingest pdf /path/to/manual.pdf "Product Manual"
```

## License

MIT
