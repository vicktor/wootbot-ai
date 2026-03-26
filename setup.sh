#!/bin/bash
set -e

echo "=== WootBot AI - Setup ==="

# 1. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "→ Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Create database
echo "→ Creating database..."
sudo -u postgres psql -c "CREATE DATABASE wootbot;" 2>/dev/null || echo "  Database already exists"
sudo -u postgres psql -d wootbot -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || echo "  pgvector already enabled"

# 3. Copy project
echo "→ Installing WootBot..."
INSTALL_DIR="/home/chatwoot/wootbot"
sudo mkdir -p "$INSTALL_DIR"
sudo cp -r app/ requirements.txt .env.example "$INSTALL_DIR/"
sudo chown -R chatwoot:chatwoot "$INSTALL_DIR"

# 4. Create venv with uv and install deps
echo "→ Creating virtual environment with uv..."
sudo -u chatwoot bash -c "
export PATH=\"\$HOME/.local/bin:\$PATH\"
cd $INSTALL_DIR
uv venv .venv
uv pip install -r requirements.txt
"

# 5. Setup .env
if [ ! -f "$INSTALL_DIR/.env" ]; then
    sudo cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    sudo chown chatwoot:chatwoot "$INSTALL_DIR/.env"
    echo ""
    echo "⚠️  Edit the .env file with your config:"
    echo "   sudo nano $INSTALL_DIR/.env"
    echo ""
fi

# 6. Install systemd service
echo "→ Installing systemd service..."
sudo cp wootbot.service /etc/systemd/system/wootbot.service
sudo systemctl daemon-reload
sudo systemctl enable wootbot

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit config:     sudo nano $INSTALL_DIR/.env"
echo "  2. Start service:   sudo systemctl start wootbot"
echo "  3. Check status:    sudo systemctl status wootbot"
echo "  4. View logs:       sudo journalctl -u wootbot -f"
echo ""
echo "Register the bot in Chatwoot:"
echo "  → Settings → Integrations → Bots → Add Bot"
echo "  → Name: WootBot AI"
echo "  → Webhook URL: http://127.0.0.1:8200/webhook"
echo "  → Copy the bot token to .env as CHATWOOT_BOT_TOKEN"
echo ""
echo "Admin panel (add as Chatwoot Dashboard App):"
echo "  → Settings → Integrations → Dashboard Apps → Add"
echo "  → Name: Knowledge Base"
echo "  → URL: http://127.0.0.1:8200/admin/ui"
echo ""
