#!/bin/bash
set -e

echo "=== WootBot AI - Setup ==="

# 1. Create database
echo "→ Creating database..."
sudo -u postgres psql -c "CREATE DATABASE wootbot;" 2>/dev/null || echo "  Database already exists"
sudo -u postgres psql -d wootbot -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || echo "  pgvector already enabled"

# 2. Copy project
echo "→ Installing WootBot..."
INSTALL_DIR="/home/chatwoot/wootbot"
sudo mkdir -p "$INSTALL_DIR"
sudo cp -r app/ requirements.txt .env.example "$INSTALL_DIR/"
sudo chown -R chatwoot:chatwoot "$INSTALL_DIR"

# 3. Create virtualenv and install deps
echo "→ Installing Python dependencies..."
sudo -u chatwoot bash -c "
cd $INSTALL_DIR
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install gunicorn
.venv/bin/pip install -r requirements.txt
"

# 4. Setup .env
if [ ! -f "$INSTALL_DIR/.env" ]; then
    sudo cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    sudo chown chatwoot:chatwoot "$INSTALL_DIR/.env"
    echo ""
    echo "⚠️  Edit the .env file with your config:"
    echo "   sudo nano $INSTALL_DIR/.env"
    echo ""
fi

# 5. Install systemd service
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
echo "Then in Chatwoot:"
echo "  → Settings → Bots → Add Bot"
echo "  → Name: WootBot AI"
echo "  → Webhook URL: http://127.0.0.1:8200/webhook"
echo ""
echo "Ingest docs:"
echo "  curl -X POST http://127.0.0.1:8200/admin/ingest/url \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"url\": \"https://docs.listen.doctor\", \"title\": \"Listen.Doctor Docs\"}'"
echo ""
