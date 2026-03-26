import os
import tempfile
import structlog
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.rag.ingest import (
    ingest_url, ingest_text, ingest_file,
    get_document_stats, delete_document_by_source,
    SUPPORTED_EXTENSIONS,
)
from app.database import get_session, ConversationLog
from app.config import get_settings

logger = structlog.get_logger()
router = APIRouter(prefix="/admin")


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

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        count = await ingest_file(tmp_path, title=title or file.filename)
        return {"status": "ok", "filename": file.filename, "chunks_ingested": count}
    finally:
        os.unlink(tmp_path)


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


# ── Admin UI (embeddable as Chatwoot Dashboard App) ──────────────────

@router.get("/ui", response_class=HTMLResponse)
async def admin_ui():
    return ADMIN_HTML


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
  input[type="text"], input[type="url"] { width: 100%; padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 0.9rem; margin-bottom: 8px; }
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

<h1>🤖 WootBot AI</h1>
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
  </div>

  <div id="tab-file" class="tab-content active">
    <div class="file-drop" id="file-drop">
      <input type="file" id="file-input" accept=".pdf,.docx,.md,.txt,.csv">
      <p>📄 Drop a file here or click to browse</p>
      <p class="formats">PDF, DOCX, MD, TXT, CSV</p>
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
const API = window.location.origin;

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
  dropZone.querySelector('p').textContent = '📎 ' + fileInput.files[0].name;
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) dropZone.querySelector('p').textContent = '📎 ' + fileInput.files[0].name;
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

async function uploadFile() {
  const file = fileInput.files[0];
  if (!file) return toast('Select a file first', 'error');
  const form = new FormData();
  form.append('file', file);
  const title = document.getElementById('file-title').value;
  if (title) form.append('title', title);
  setLoading('btn-file', true);
  try {
    const r = await fetch(API + '/admin/ingest/file', { method: 'POST', body: form });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Upload failed');
    toast(`Ingested ${data.chunks_ingested} chunks from ${file.name}`);
    fileInput.value = '';
    dropZone.querySelector('p').textContent = '📄 Drop a file here or click to browse';
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
    const r = await fetch(API + '/admin/ingest/url', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ url, title })
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Ingest failed');
    toast(`Ingested ${data.chunks_ingested} chunks from URL`);
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
    const r = await fetch(API + '/admin/ingest/text', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ content, source, title })
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Ingest failed');
    toast(`Ingested ${data.chunks_ingested} chunks`);
    document.getElementById('text-content').value = '';
    document.getElementById('text-source').value = '';
    document.getElementById('text-title').value = '';
    loadDocs(); loadStats();
  } catch(e) { toast(e.message, 'error'); }
  setLoading('btn-text', false);
}

async function deleteDoc(source) {
  if (!confirm('Delete all chunks from this source?')) return;
  try {
    const r = await fetch(API + '/admin/documents', {
      method: 'DELETE', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ source })
    });
    const data = await r.json();
    toast(`Deleted ${data.chunks_deleted} chunks`);
    loadDocs(); loadStats();
  } catch(e) { toast(e.message, 'error'); }
}

async function loadDocs() {
  try {
    const r = await fetch(API + '/admin/documents');
    const data = await r.json();
    const tbody = document.getElementById('doc-table');
    if (!data.sources || !data.sources.length) {
      tbody.innerHTML = '<tr><td colspan="4" style="color:#9ca3af">No documents ingested yet</td></tr>';
      return;
    }
    tbody.innerHTML = data.sources.map(s =>
      `<tr>
        <td><strong>${esc(s.title)}</strong></td>
        <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#6b7280">${esc(s.source)}</td>
        <td>${s.chunks}</td>
        <td><button class="btn btn-danger" onclick="deleteDoc('${esc(s.source)}')">Delete</button></td>
      </tr>`
    ).join('');
  } catch(e) { console.error(e); }
}

async function loadStats() {
  try {
    const r = await fetch(API + '/admin/stats');
    const d = await r.json();
    document.getElementById('s-sources').textContent = d.knowledge_base_sources || 0;
    document.getElementById('s-chunks').textContent = d.knowledge_base_chunks || 0;
    document.getElementById('s-convos').textContent = d.total_conversations_handled || 0;
    document.getElementById('s-confidence').textContent = d.average_confidence ? (d.average_confidence * 100).toFixed(0) + '%' : '-';
    document.getElementById('s-handoffs').textContent = d.handoff_count || 0;
  } catch(e) { console.error(e); }
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

loadDocs();
loadStats();
</script>
</body>
</html>"""
