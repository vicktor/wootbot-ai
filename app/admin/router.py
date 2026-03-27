import hmac
import os
import tempfile
import structlog
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from app.rag.ingest import (
    ingest_url, ingest_text, ingest_file, ingest_resolved_tickets,
    get_document_stats, delete_document_by_source,
    SUPPORTED_EXTENSIONS,
)
from app.database import get_session, ConversationLog, get_bot_setting, set_bot_setting
from app.config import get_settings
from app.rag.embeddings import EMBEDDING_MODELS, get_embedding_config
from app.rag.reembed import reembed_all, get_reembed_status

logger = structlog.get_logger()

MAX_FILE_SIZE = 50 * 1024 * 1024  # #14: 50MB limit


# ── Auth ────────────────────────────────────────────────────────────

def _get_admin_secret():
    return get_settings().admin_secret


async def verify_admin(request: Request):
    """Check auth via cookie or header."""
    secret = _get_admin_secret()
    if not secret:
        return True
    token = (
        request.cookies.get("wootbot_admin")
        or request.headers.get("X-Admin-Secret")
        or ""
    )
    # #5: Constant-time comparison to prevent timing attacks
    return hmac.compare_digest(token, secret)


async def require_admin(request: Request):
    if not await verify_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")


# Two routers: protected (API) and public (UI + login)
router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])
public_router = APIRouter(prefix="/admin")


class IngestURLRequest(BaseModel):
    url: str
    title: str | None = None


class IngestTextRequest(BaseModel):
    content: str
    source: str
    title: str | None = None


class DeleteDocRequest(BaseModel):
    source: str


# ── API Endpoints ────────────────────────────────────────────────────

@router.post("/ingest/url")
async def api_ingest_url(req: IngestURLRequest):
    count = await ingest_url(req.url, req.title)
    return {"status": "ok", "chunks_ingested": count}


@router.post("/ingest/text")
async def api_ingest_text(req: IngestTextRequest):
    count = await ingest_text(req.content, req.source, req.title)
    return {"status": "ok", "chunks_ingested": count}


@router.post("/ingest/file")
async def api_ingest_file(
    file: UploadFile = File(...),
    title: str = Form(None),
):
    """Upload and ingest a file (PDF, DOCX, MD, TXT, CSV)."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

    # #14: Enforce file size limit
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large. Maximum: {MAX_FILE_SIZE // (1024*1024)}MB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        count = await ingest_file(tmp_path, title=title or file.filename)
        return {"status": "ok", "filename": file.filename, "chunks_ingested": count}
    finally:
        os.unlink(tmp_path)


@router.post("/ingest/tickets")
async def api_ingest_tickets(max_pages: int = 10):
    """Ingest resolved Chatwoot tickets into the knowledge base."""
    # #22: Bound max_pages
    if max_pages < 1 or max_pages > 100:
        raise HTTPException(400, "max_pages must be between 1 and 100")
    count = await ingest_resolved_tickets(max_pages=max_pages)
    return {"status": "ok", "chunks_ingested": count}


@router.get("/documents")
async def api_list_documents():
    return get_document_stats()


@router.delete("/documents")
async def api_delete_document(req: DeleteDocRequest):
    count = delete_document_by_source(req.source)
    return {"status": "ok", "chunks_deleted": count}


@router.get("/stats")
async def api_stats():
    session = get_session()
    try:
        from sqlalchemy import func
        total_logs = session.query(ConversationLog).count()
        avg_confidence = session.query(func.avg(ConversationLog.confidence)).scalar()
        settings = get_settings()
        handoff_count = session.query(ConversationLog).filter(
            ConversationLog.confidence < settings.confidence_threshold
        ).count()
        doc_stats = get_document_stats()
        return {
            "total_conversations_handled": total_logs,
            "average_confidence": round(float(avg_confidence or 0), 3),
            "handoff_count": handoff_count,
            "knowledge_base_chunks": doc_stats["total_chunks"],
            "knowledge_base_sources": doc_stats["total_sources"],
        }
    finally:
        session.close()


SETTING_KEYS = {"email_greeting", "email_closing"}


@router.get("/settings")
async def api_get_settings():
    settings = get_settings()
    return {
        key: get_bot_setting(key, getattr(settings, key, ""))
        for key in SETTING_KEYS
    }


class UpdateSettingsRequest(BaseModel):
    email_greeting: str | None = None
    email_closing: str | None = None


@router.put("/settings")
async def api_update_settings(req: UpdateSettingsRequest):
    updated = []
    for key in SETTING_KEYS:
        value = getattr(req, key, None)
        if value is not None:
            set_bot_setting(key, value)
            updated.append(key)
    return {"status": "ok", "updated": updated}


# ── Embedding Config Endpoints ──────────────────────────────────────

class EmbeddingConfigRequest(BaseModel):
    provider: str
    model: str
    api_key: str = ""


@router.get("/embedding")
async def api_get_embedding_config():
    config = get_embedding_config()
    masked_key = config["api_key"]
    if masked_key and len(masked_key) > 12:
        masked_key = masked_key[:8] + "..." + masked_key[-4:]
    elif masked_key:
        masked_key = "***"
    return {
        "provider": config["provider"],
        "model": config["model"],
        "api_key_set": bool(config["api_key"]),
        "api_key_masked": masked_key,
        "models": EMBEDDING_MODELS,
    }


@router.put("/embedding")
async def api_update_embedding(req: EmbeddingConfigRequest, background_tasks: BackgroundTasks):
    if req.provider not in EMBEDDING_MODELS:
        raise HTTPException(400, f"Unknown provider: {req.provider}")
    if req.model not in EMBEDDING_MODELS[req.provider]:
        raise HTTPException(400, f"Unknown model: {req.model} for provider {req.provider}")
    background_tasks.add_task(reembed_all, req.provider, req.model, req.api_key)
    return {"status": "ok", "message": "Re-embedding started in background"}


@router.get("/embedding/status")
async def api_reembed_status():
    return get_reembed_status()


@public_router.post("/login")
async def admin_login(request: Request):
    """Validate secret — client stores it and sends as header on future requests."""
    body = await request.json()
    secret = body.get("secret", "")
    admin_secret = _get_admin_secret()
    if not admin_secret or hmac.compare_digest(secret, admin_secret):
        return {"status": "ok"}
    raise HTTPException(status_code=401, detail="Invalid secret")


# ── Admin UI (embeddable as Chatwoot Dashboard App) ──────────────────

def _frame_ancestors():
    """Build CSP frame-ancestors from allowed_origins setting."""
    settings = get_settings()
    origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
    return "frame-ancestors 'self' " + " ".join(origins) if origins else "frame-ancestors 'self'"


def _secure_response(html: str) -> HTMLResponse:
    """Return HTML response with iframe security headers."""
    response = HTMLResponse(html)
    response.headers["Content-Security-Policy"] = _frame_ancestors()
    return response


@public_router.get("/ui", response_class=HTMLResponse)
async def admin_ui(request: Request):
    """Serve admin UI with secret injected into JS."""
    admin_secret = _get_admin_secret()
    if not admin_secret:
        # No auth configured — serve without secret
        return _secure_response(ADMIN_HTML.replace("__ADMIN_SECRET__", ""))

    # If ?secret= is valid, inject it into the page
    query_secret = request.query_params.get("secret", "")
    if query_secret and hmac.compare_digest(query_secret, admin_secret):
        return _secure_response(ADMIN_HTML.replace("__ADMIN_SECRET__", query_secret))

    # Show login page
    return _secure_response(LOGIN_HTML)


# #4: Login page uses POST form — secret never in URL
LOGIN_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WootBot Admin - Login</title>
<style>
body{font-family:-apple-system,sans-serif;display:grid;place-content:center;height:100vh;background:#f8f9fa;margin:0}
.box{background:white;padding:32px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.08);text-align:center;max-width:320px}
h2{margin-bottom:16px;font-size:1.2rem}
input{width:100%;padding:10px;border:1px solid #d1d5db;border-radius:6px;margin-bottom:12px;font-size:0.9rem}
button{width:100%;padding:10px;background:#1a56db;color:white;border:none;border-radius:6px;cursor:pointer;font-weight:600}
button:hover{background:#1e40af}
.err{color:#dc2626;font-size:0.8rem;margin-top:8px;display:none}
</style></head><body>
<div class="box">
<h2>WootBot Admin</h2>
<input type="password" id="secret" placeholder="Admin secret">
<button onclick="login()">Login</button>
<p class="err" id="err">Invalid secret</p>
</div>
<script>
const API = window.location.origin + '/wootbot';
async function login(){
  const s=document.getElementById('secret').value;
  if(!s)return;
  try {
    const r = await fetch(API + '/admin/login', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({secret: s})
    });
    if(r.ok) {
      window.location.href = window.location.pathname + '?secret=' + encodeURIComponent(s);
    }
    else { document.getElementById('err').style.display='block'; }
  } catch(e) { document.getElementById('err').style.display='block'; }
}
document.getElementById('secret').addEventListener('keypress',e=>{if(e.key==='Enter')login()});
</script></body></html>"""



ADMIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WootBot AI - Knowledge Base</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8f9fa; color: #1b2a4a; padding: 20px; }
  h1 { font-size: 1.4rem; margin-bottom: 4px; }
  .subtitle { color: #6b7280; font-size: 0.85rem; margin-bottom: 20px; }
  .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .card h2 { font-size: 1rem; margin-bottom: 12px; color: #1b2a4a; }
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
  .stat { text-align: center; padding: 12px; background: #f0f4ff; border-radius: 6px; }
  .stat-value { font-size: 1.8rem; font-weight: 700; color: #1a56db; }
  .stat-label { font-size: 0.75rem; color: #6b7280; margin-top: 2px; }
  .tabs { display: flex; gap: 0; margin-bottom: 16px; border-bottom: 2px solid #e5e7eb; }
  .tab { padding: 8px 16px; cursor: pointer; font-size: 0.85rem; border-bottom: 2px solid transparent; margin-bottom: -2px; color: #6b7280; }
  .tab.active { color: #1a56db; border-bottom-color: #1a56db; font-weight: 600; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  input[type="text"], input[type="url"], input[type="password"], select { width: 100%; padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 0.9rem; margin-bottom: 8px; }
  textarea { width: 100%; padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 0.9rem; margin-bottom: 8px; min-height: 80px; resize: vertical; }
  label { display: block; font-size: 0.8rem; font-weight: 600; margin-bottom: 4px; color: #374151; }
  .btn { padding: 8px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.85rem; font-weight: 600; }
  .btn-primary { background: #1a56db; color: white; }
  .btn-primary:hover { background: #1e40af; }
  .btn-danger { background: #dc2626; color: white; font-size: 0.75rem; padding: 4px 10px; }
  .btn-danger:hover { background: #b91c1c; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .file-drop { border: 2px dashed #d1d5db; border-radius: 8px; padding: 30px; text-align: center; cursor: pointer; transition: border-color 0.2s; margin-bottom: 8px; }
  .file-drop:hover, .file-drop.dragover { border-color: #1a56db; background: #f0f4ff; }
  .file-drop input { display: none; }
  .file-drop p { color: #6b7280; font-size: 0.85rem; }
  .file-drop .formats { font-size: 0.75rem; color: #9ca3af; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { text-align: left; padding: 8px; border-bottom: 2px solid #e5e7eb; color: #6b7280; font-size: 0.75rem; text-transform: uppercase; }
  td { padding: 8px; border-bottom: 1px solid #f3f4f6; }
  .toast { position: fixed; top: 20px; right: 20px; padding: 12px 20px; border-radius: 6px; color: white; font-size: 0.85rem; z-index: 999; animation: fadeIn 0.3s; }
  .toast.success { background: #059669; }
  .toast.error { background: #dc2626; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid white; border-top-color: transparent; border-radius: 50%; animation: spin 0.6s linear infinite; margin-right: 6px; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<h1>WootBot AI</h1>
<p class="subtitle">Knowledge Base Manager</p>

<!-- Stats -->
<div class="card">
  <div class="stats-grid" id="stats">
    <div class="stat"><div class="stat-value" id="s-sources">-</div><div class="stat-label">Sources</div></div>
    <div class="stat"><div class="stat-value" id="s-chunks">-</div><div class="stat-label">Chunks</div></div>
    <div class="stat"><div class="stat-value" id="s-convos">-</div><div class="stat-label">Conversations</div></div>
    <div class="stat"><div class="stat-value" id="s-confidence">-</div><div class="stat-label">Avg Confidence</div></div>
    <div class="stat"><div class="stat-value" id="s-handoffs">-</div><div class="stat-label">Handoffs</div></div>
  </div>
</div>

<!-- Ingest -->
<div class="card">
  <h2>Add Knowledge</h2>
  <div class="tabs">
    <div class="tab active" data-tab="tab-file">Upload File</div>
    <div class="tab" data-tab="tab-url">URL</div>
    <div class="tab" data-tab="tab-text">Text</div>
    <div class="tab" data-tab="tab-tickets">Chatwoot Tickets</div>
    <div class="tab" data-tab="tab-settings">Settings</div>
  </div>

  <div id="tab-file" class="tab-content active">
    <div class="file-drop" id="file-drop">
      <input type="file" id="file-input" accept=".pdf,.docx,.md,.txt,.csv">
      <p>Drop a file here or click to browse</p>
      <p class="formats">PDF, DOCX, MD, TXT, CSV (max 50MB)</p>
    </div>
    <label>Title (optional)</label>
    <input type="text" id="file-title" placeholder="Document title">
    <button class="btn btn-primary" id="btn-file" onclick="uploadFile()">Upload & Ingest</button>
  </div>

  <div id="tab-url" class="tab-content">
    <label>URL</label>
    <input type="url" id="url-input" placeholder="https://docs.example.com/help">
    <label>Title (optional)</label>
    <input type="text" id="url-title" placeholder="Help Center">
    <button class="btn btn-primary" id="btn-url" onclick="ingestURL()">Ingest URL</button>
  </div>

  <div id="tab-text" class="tab-content">
    <label>Source Name</label>
    <input type="text" id="text-source" placeholder="FAQ">
    <label>Title (optional)</label>
    <input type="text" id="text-title" placeholder="Frequently Asked Questions">
    <label>Content</label>
    <textarea id="text-content" placeholder="Paste your text here..."></textarea>
    <button class="btn btn-primary" id="btn-text" onclick="ingestText()">Ingest Text</button>
  </div>

  <div id="tab-tickets" class="tab-content">
    <p style="color:#6b7280;font-size:0.85rem;margin-bottom:12px">Import resolved Chatwoot conversations into the knowledge base. The bot will learn from successfully handled tickets.</p>
    <label>Max pages to fetch (25 tickets per page, max 100)</label>
    <input type="text" id="tickets-pages" placeholder="10" value="10">
    <button class="btn btn-primary" id="btn-tickets" onclick="ingestTickets()">Import Resolved Tickets</button>
  </div>

  <div id="tab-settings" class="tab-content">
    <h3 style="margin-bottom:8px;font-size:0.9rem">Embedding Provider</h3>
    <p style="color:#6b7280;font-size:0.85rem;margin-bottom:12px">Choose the embedding provider and model. Changing provider will re-generate all embeddings (may take a few minutes).</p>
    <label>Provider</label>
    <select id="set-emb-provider" onchange="updateModelOptions()">
      <option value="gemini">Google (Gemini)</option>
      <option value="openai">OpenAI</option>
    </select>
    <label>Model</label>
    <select id="set-emb-model"></select>
    <label>API Key (leave empty to use LLM key from .env)</label>
    <input type="password" id="set-emb-key" placeholder="API key for embedding provider">
    <div id="emb-status" style="display:none;padding:8px;background:#f0f4ff;border-radius:6px;margin-bottom:8px;font-size:0.85rem"></div>
    <button class="btn btn-primary" id="btn-embedding" onclick="saveEmbedding()">Save Embedding Config</button>
    <hr style="margin:20px 0;border:none;border-top:1px solid #e5e7eb">
    <h3 style="margin-bottom:8px;font-size:0.9rem">Email Formatting</h3>
    <p style="color:#6b7280;font-size:0.85rem;margin-bottom:12px">The greeting and closing are automatically translated by the AI to the customer's language.</p>
    <label>Email Greeting</label>
    <input type="text" id="set-email-greeting" placeholder="Hola, gracias por contactar con nosotros.">
    <label>Email Closing</label>
    <textarea id="set-email-closing" placeholder="Esperamos haber sido de ayuda..."></textarea>
    <button class="btn btn-primary" id="btn-settings" onclick="saveSettings()">Save Settings</button>
  </div>
</div>

<!-- Documents -->
<div class="card">
  <h2>Indexed Documents</h2>
  <table>
    <thead><tr><th>Title</th><th>Source</th><th>Chunks</th><th></th></tr></thead>
    <tbody id="doc-table"><tr><td colspan="4">Loading...</td></tr></tbody>
  </table>
</div>

<script>
// Secret injected by server — clean it from URL bar
const SECRET = '__ADMIN_SECRET__';
if (window.location.search.includes('secret=')) {
  window.history.replaceState({}, '', window.location.pathname);
}

const API = window.location.origin + '/wootbot';

// Authenticated fetch helper
function authFetch(url, opts = {}) {
  if (!opts.headers) opts.headers = {};
  opts.headers['X-Admin-Secret'] = SECRET;
  return fetch(url, opts);
}

// Tabs
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.tab).classList.add('active');
  });
});

// File drop zone
const dropZone = document.getElementById('file-drop');
const fileInput = document.getElementById('file-input');
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  fileInput.files = e.dataTransfer.files;
  dropZone.querySelector('p').textContent = fileInput.files[0].name;
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) dropZone.querySelector('p').textContent = fileInput.files[0].name;
});

function toast(msg, type = 'success') {
  const el = document.createElement('div');
  el.className = 'toast ' + type;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

function setLoading(btnId, loading) {
  const btn = document.getElementById(btnId);
  if (loading) {
    btn.disabled = true;
    btn.dataset.text = btn.textContent;
    btn.innerHTML = '<span class="spinner"></span>Processing...';
  } else {
    btn.disabled = false;
    btn.textContent = btn.dataset.text;
  }
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

async function uploadFile() {
  const file = fileInput.files[0];
  if (!file) return toast('Select a file first', 'error');
  const form = new FormData();
  form.append('file', file);
  const title = document.getElementById('file-title').value;
  if (title) form.append('title', title);
  setLoading('btn-file', true);
  try {
    const r = await authFetch(API + '/admin/ingest/file', { method: 'POST', body: form });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Upload failed');
    toast('Ingested ' + data.chunks_ingested + ' chunks from ' + file.name);
    fileInput.value = '';
    dropZone.querySelector('p').textContent = 'Drop a file here or click to browse';
    document.getElementById('file-title').value = '';
    loadDocs(); loadStats();
  } catch(e) { toast(e.message, 'error'); }
  setLoading('btn-file', false);
}

async function ingestURL() {
  const url = document.getElementById('url-input').value;
  if (!url) return toast('Enter a URL', 'error');
  const title = document.getElementById('url-title').value || null;
  setLoading('btn-url', true);
  try {
    const r = await authFetch(API + '/admin/ingest/url', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ url, title })
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Ingest failed');
    toast('Ingested ' + data.chunks_ingested + ' chunks from URL');
    document.getElementById('url-input').value = '';
    document.getElementById('url-title').value = '';
    loadDocs(); loadStats();
  } catch(e) { toast(e.message, 'error'); }
  setLoading('btn-url', false);
}

async function ingestText() {
  const content = document.getElementById('text-content').value;
  const source = document.getElementById('text-source').value || 'manual';
  const title = document.getElementById('text-title').value || null;
  if (!content) return toast('Enter some text', 'error');
  setLoading('btn-text', true);
  try {
    const r = await authFetch(API + '/admin/ingest/text', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ content, source, title })
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Ingest failed');
    toast('Ingested ' + data.chunks_ingested + ' chunks');
    document.getElementById('text-content').value = '';
    document.getElementById('text-source').value = '';
    document.getElementById('text-title').value = '';
    loadDocs(); loadStats();
  } catch(e) { toast(e.message, 'error'); }
  setLoading('btn-text', false);
}

async function ingestTickets() {
  const maxPages = parseInt(document.getElementById('tickets-pages').value) || 10;
  setLoading('btn-tickets', true);
  try {
    const r = await authFetch(API + '/admin/ingest/tickets?max_pages=' + maxPages, { method: 'POST' });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Ticket import failed');
    toast('Ingested ' + data.chunks_ingested + ' chunks from resolved tickets');
    loadDocs(); loadStats();
  } catch(e) { toast(e.message, 'error'); }
  setLoading('btn-tickets', false);
}

async function deleteDoc(source) {
  if (!confirm('Delete all chunks from this source?')) return;
  try {
    const r = await authFetch(API + '/admin/documents', {
      method: 'DELETE', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ source })
    });
    const data = await r.json();
    toast('Deleted ' + data.chunks_deleted + ' chunks');
    loadDocs(); loadStats();
  } catch(e) { toast(e.message, 'error'); }
}

async function loadDocs() {
  try {
    const r = await authFetch(API + '/admin/documents');
    const data = await r.json();
    const tbody = document.getElementById('doc-table');
    if (!data.sources || !data.sources.length) {
      tbody.innerHTML = '<tr><td colspan="4" style="color:#9ca3af">No documents ingested yet</td></tr>';
      return;
    }
    tbody.innerHTML = data.sources.map(s =>
      '<tr>' +
        '<td><strong>' + esc(s.title) + '</strong></td>' +
        '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#6b7280">' + esc(s.source) + '</td>' +
        '<td>' + s.chunks + '</td>' +
        '<td><button class="btn btn-danger" data-source="' + esc(s.source) + '">Delete</button></td>' +
      '</tr>'
    ).join('');
    tbody.querySelectorAll('[data-source]').forEach(btn => {
      btn.addEventListener('click', () => deleteDoc(btn.dataset.source));
    });
  } catch(e) { console.error(e); }
}

async function loadStats() {
  try {
    const r = await authFetch(API + '/admin/stats');
    const d = await r.json();
    document.getElementById('s-sources').textContent = d.knowledge_base_sources || 0;
    document.getElementById('s-chunks').textContent = d.knowledge_base_chunks || 0;
    document.getElementById('s-convos').textContent = d.total_conversations_handled || 0;
    document.getElementById('s-confidence').textContent = d.average_confidence ? (d.average_confidence * 100).toFixed(0) + '%' : '-';
    document.getElementById('s-handoffs').textContent = d.handoff_count || 0;
  } catch(e) { console.error(e); }
}

async function loadSettings() {
  try {
    const r = await authFetch(API + '/admin/settings');
    const d = await r.json();
    document.getElementById('set-email-greeting').value = d.email_greeting || '';
    document.getElementById('set-email-closing').value = d.email_closing || '';
  } catch(e) { console.error(e); }
}

async function saveSettings() {
  setLoading('btn-settings', true);
  try {
    const r = await authFetch(API + '/admin/settings', {
      method: 'PUT', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        email_greeting: document.getElementById('set-email-greeting').value,
        email_closing: document.getElementById('set-email-closing').value,
      })
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Save failed');
    toast('Settings saved');
  } catch(e) { toast(e.message, 'error'); }
  setLoading('btn-settings', false);
}

const EMB_MODELS = {};

function updateModelOptions() {
  const provider = document.getElementById('set-emb-provider').value;
  const select = document.getElementById('set-emb-model');
  const models = EMB_MODELS[provider] || {};
  select.innerHTML = Object.keys(models).map(m =>
    '<option value="' + m + '">' + m + ' (' + models[m] + ' dims)</option>'
  ).join('');
}

async function loadEmbeddingConfig() {
  try {
    const r = await authFetch(API + '/admin/embedding');
    const d = await r.json();
    Object.assign(EMB_MODELS, d.models || {});
    document.getElementById('set-emb-provider').value = d.provider;
    updateModelOptions();
    document.getElementById('set-emb-model').value = d.model;
    if (d.api_key_masked) {
      document.getElementById('set-emb-key').placeholder = 'Current: ' + d.api_key_masked;
    }
  } catch(e) { console.error(e); }
}

async function saveEmbedding() {
  const provider = document.getElementById('set-emb-provider').value;
  const model = document.getElementById('set-emb-model').value;
  const api_key = document.getElementById('set-emb-key').value;
  setLoading('btn-embedding', true);
  try {
    const r = await authFetch(API + '/admin/embedding', {
      method: 'PUT', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ provider, model, api_key })
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Save failed');
    toast('Embedding config saved. Re-embedding started...');
    pollReembedStatus();
  } catch(e) { toast(e.message, 'error'); }
  setLoading('btn-embedding', false);
}

function pollReembedStatus() {
  const el = document.getElementById('emb-status');
  el.style.display = 'block';
  el.textContent = 'Re-embedding in progress...';
  const interval = setInterval(async () => {
    try {
      const r = await authFetch(API + '/admin/embedding/status');
      const d = await r.json();
      if (d.running) {
        el.textContent = 'Re-embedding: ' + d.done + ' / ' + d.total + ' chunks...';
      } else {
        clearInterval(interval);
        if (d.error) {
          el.textContent = 'Error: ' + d.error;
          el.style.background = '#fef2f2';
        } else {
          el.textContent = 'Re-embedding complete! ' + d.done + ' chunks processed.';
          el.style.background = '#f0fdf4';
          setTimeout(() => { el.style.display = 'none'; }, 5000);
        }
        loadDocs(); loadStats();
      }
    } catch(e) { clearInterval(interval); }
  }, 2000);
}

loadDocs();
loadStats();
loadSettings();
loadEmbeddingConfig();
</script>
</body>
</html>"""
