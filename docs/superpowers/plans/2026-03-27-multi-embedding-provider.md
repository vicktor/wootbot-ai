# Multi-Provider Embeddings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow users to choose between Google and OpenAI embedding providers from the admin UI, with automatic re-embedding when switching.

**Architecture:** Extract embedding logic from LLMProvider into a standalone `EmbeddingProvider` module. Config stored in `bot_settings` DB table. Admin UI extended with embedding config section. Re-embed runs as background task with progress tracking via a simple in-memory counter.

**Tech Stack:** Python, FastAPI, SQLAlchemy, pgvector, google-genai, openai, httpx

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `app/rag/embeddings.py` | **Create** | `EmbeddingProvider` classes (Gemini, OpenAI), `get_embedding_provider()` factory, model registry with dimensions |
| `app/rag/reembed.py` | **Create** | `reembed_all()` background task, progress tracking, vector column resize |
| `app/llm/provider.py` | **Modify** | Remove `get_embedding` from LLMProvider and all subclasses |
| `app/rag/search.py` | **Modify** | Import from `embeddings.py` instead of `provider.py` |
| `app/rag/ingest.py` | **Modify** | Import from `embeddings.py` instead of `provider.py` |
| `app/admin/router.py` | **Modify** | Add embedding config endpoints + re-embed trigger + progress endpoint + admin UI section |
| `app/database.py` | **Modify** | Add `resize_embedding_column()` helper |
| `app/main.py` | **Modify** | Import reembed for background task |

---

### Task 1: Create `app/rag/embeddings.py` — Embedding Provider Module

**Files:**
- Create: `app/rag/embeddings.py`

- [ ] **Step 1: Create the embedding provider module**

```python
import asyncio
import structlog
from abc import ABC, abstractmethod
from app.database import get_bot_setting
from app.config import get_settings

logger = structlog.get_logger()

# Registry of supported models with their dimensions
EMBEDDING_MODELS = {
    "gemini": {
        "gemini-embedding-001": {"dimensions": 3072},
    },
    "openai": {
        "text-embedding-3-small": {"dimensions": 1536},
        "text-embedding-3-large": {"dimensions": 3072},
    },
}

DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-embedding-001"


def get_embedding_config() -> dict:
    """Get current embedding config from DB, falling back to defaults."""
    return {
        "provider": get_bot_setting("embedding_provider", DEFAULT_PROVIDER),
        "model": get_bot_setting("embedding_model", DEFAULT_MODEL),
        "api_key": get_bot_setting("embedding_api_key", ""),
    }


def get_embedding_dimensions(provider: str, model: str) -> int:
    """Get vector dimensions for a provider/model combo."""
    return EMBEDDING_MODELS.get(provider, {}).get(model, {}).get("dimensions", 3072)


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        pass


class GeminiEmbedding(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    async def embed(self, text: str) -> list[float]:
        result = await asyncio.to_thread(
            self.client.models.embed_content,
            model=self.model, contents=text
        )
        return result.embeddings[0].values


class OpenAIEmbedding(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model=self.model, input=text
        )
        return response.data[0].embedding


_cached_provider: EmbeddingProvider | None = None
_cached_config_hash: str = ""


def _config_hash(config: dict) -> str:
    return f"{config['provider']}:{config['model']}:{config['api_key']}"


def get_embedding_provider() -> EmbeddingProvider:
    """Get or create the embedding provider based on DB config."""
    global _cached_provider, _cached_config_hash

    config = get_embedding_config()

    # Use fallback API key from .env if not configured in DB
    api_key = config["api_key"]
    if not api_key:
        settings = get_settings()
        api_key = settings.llm_api_key

    current_hash = _config_hash({**config, "api_key": api_key})
    if _cached_provider and current_hash == _cached_config_hash:
        return _cached_provider

    providers = {
        "gemini": GeminiEmbedding,
        "openai": OpenAIEmbedding,
    }
    provider_class = providers.get(config["provider"])
    if not provider_class:
        raise ValueError(f"Unknown embedding provider: {config['provider']}")

    _cached_provider = provider_class(api_key=api_key, model=config["model"])
    _cached_config_hash = current_hash
    logger.info("embedding_provider_created", provider=config["provider"], model=config["model"])
    return _cached_provider
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('app/rag/embeddings.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/rag/embeddings.py
git commit -m "feat: add standalone embedding provider module with Gemini and OpenAI support"
```

---

### Task 2: Create `app/rag/reembed.py` — Re-embedding Background Task

**Files:**
- Create: `app/rag/reembed.py`

- [ ] **Step 1: Create the re-embed module**

```python
import structlog
from sqlalchemy import text
from app.database import get_session, get_engine, Document, set_bot_setting
from app.rag.embeddings import get_embedding_provider, get_embedding_dimensions

logger = structlog.get_logger()

# Simple in-memory progress tracking
reembed_status = {
    "running": False,
    "total": 0,
    "done": 0,
    "error": None,
}


def get_reembed_status() -> dict:
    return dict(reembed_status)


async def reembed_all(provider: str, model: str, api_key: str):
    """Re-embed all documents with a new provider/model.

    1. Save new embedding config
    2. Resize the vector column if dimensions changed
    3. Re-generate embeddings for all documents
    """
    global reembed_status

    if reembed_status["running"]:
        raise RuntimeError("Re-embed already in progress")

    reembed_status.update(running=True, total=0, done=0, error=None)

    try:
        # 1. Get new dimensions
        new_dims = get_embedding_dimensions(provider, model)

        # 2. Save new config to DB
        set_bot_setting("embedding_provider", provider)
        set_bot_setting("embedding_model", model)
        if api_key:
            set_bot_setting("embedding_api_key", api_key)

        # 3. Resize vector column if needed
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text(f"ALTER TABLE documents ALTER COLUMN embedding TYPE vector({new_dims})"))
            conn.commit()

        # 4. Get all documents
        session = get_session()
        try:
            docs = session.query(Document).all()
            reembed_status["total"] = len(docs)
        finally:
            session.close()

        # 5. Re-embed each document
        embedder = get_embedding_provider()
        for doc in docs:
            try:
                embedding = await embedder.embed(doc.content)
                session = get_session()
                try:
                    session.execute(
                        text("UPDATE documents SET embedding = cast(:emb as vector) WHERE id = :id"),
                        {"emb": "[" + ",".join(str(x) for x in embedding) + "]", "id": doc.id}
                    )
                    session.commit()
                finally:
                    session.close()
                reembed_status["done"] += 1
            except Exception as e:
                logger.error("reembed_chunk_error", doc_id=doc.id, error=str(e))
                reembed_status["done"] += 1

        logger.info("reembed_complete", total=reembed_status["total"], provider=provider, model=model)

    except Exception as e:
        logger.error("reembed_error", error=str(e))
        reembed_status["error"] = str(e)
    finally:
        reembed_status["running"] = False
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('app/rag/reembed.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/rag/reembed.py
git commit -m "feat: add re-embed background task with progress tracking"
```

---

### Task 3: Remove `get_embedding` from LLMProvider — Clean Up provider.py

**Files:**
- Modify: `app/llm/provider.py`

- [ ] **Step 1: Remove the abstract method and all implementations**

Remove from `LLMProvider`:
```python
    @abstractmethod
    async def get_embedding(self, text: str) -> list[float]:
        pass
```

Remove from `GeminiProvider`:
- The line `self.embed_model = "gemini-embedding-001"` in `__init__`
- The entire `get_embedding` method

Remove from `OpenAIProvider`:
- The entire `get_embedding` method

Remove from `AnthropicProvider`:
- The lines `from google import genai` and `embedding_key = ...` and `self._genai_client = ...` in `__init__`
- The entire `get_embedding` method

Remove the `import asyncio` ONLY if `asyncio.to_thread` is still used in GeminiProvider.generate/translate (it is — keep the import).

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('app/llm/provider.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/llm/provider.py
git commit -m "refactor: remove embedding methods from LLM providers"
```

---

### Task 4: Update `search.py` and `ingest.py` to Use New Embedding Module

**Files:**
- Modify: `app/rag/search.py`
- Modify: `app/rag/ingest.py`

- [ ] **Step 1: Update search.py**

Change the import and usage:

Replace:
```python
from app.llm.provider import get_llm_provider
```
With:
```python
from app.rag.embeddings import get_embedding_provider
```

Replace in `search_documents()`:
```python
    llm = get_llm_provider()

    try:
        query_embedding = await llm.get_embedding(query)
```
With:
```python
    embedder = get_embedding_provider()

    try:
        query_embedding = await embedder.embed(query)
```

- [ ] **Step 2: Update ingest.py**

Replace:
```python
from app.llm.provider import get_llm_provider
```
With:
```python
from app.rag.embeddings import get_embedding_provider
```

In `ingest_text()`, replace:
```python
    llm = get_llm_provider()
```
With:
```python
    embedder = get_embedding_provider()
```

And replace:
```python
            embedding = await llm.get_embedding(chunk)
```
With:
```python
            embedding = await embedder.embed(chunk)
```

- [ ] **Step 3: Verify syntax of both files**

Run: `python -c "import ast; ast.parse(open('app/rag/search.py').read()); ast.parse(open('app/rag/ingest.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add app/rag/search.py app/rag/ingest.py
git commit -m "refactor: use standalone embedding provider in search and ingest"
```

---

### Task 5: Add Admin API Endpoints for Embedding Config

**Files:**
- Modify: `app/admin/router.py`

- [ ] **Step 1: Add imports at top of router.py**

Add to the existing imports:
```python
from app.rag.embeddings import EMBEDDING_MODELS, get_embedding_config
from app.rag.reembed import reembed_all, get_reembed_status
```

- [ ] **Step 2: Add the Pydantic model and endpoints after the existing settings endpoints**

After the `api_update_settings` endpoint, add:

```python
class EmbeddingConfigRequest(BaseModel):
    provider: str
    model: str
    api_key: str = ""


@router.get("/embedding")
async def api_get_embedding_config():
    config = get_embedding_config()
    # Don't expose the full API key
    masked_key = config["api_key"]
    if masked_key:
        masked_key = masked_key[:8] + "..." + masked_key[-4:] if len(masked_key) > 12 else "***"
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
```

- [ ] **Step 3: Add BackgroundTasks import**

Add `BackgroundTasks` to the FastAPI import at the top of the file if not already present:
```python
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends, BackgroundTasks
```

- [ ] **Step 4: Verify syntax**

Run: `python -c "import ast; ast.parse(open('app/admin/router.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add app/admin/router.py
git commit -m "feat: add embedding config API endpoints with re-embed trigger"
```

---

### Task 6: Add Embedding Config Section to Admin UI HTML

**Files:**
- Modify: `app/admin/router.py` (the ADMIN_HTML string)

- [ ] **Step 1: Add the Embedding Config section in the Settings tab**

In the `tab-settings` div, add the embedding section BEFORE the email section:

```html
    <h3 style="margin-bottom:8px;font-size:0.9rem">Embedding Provider</h3>
    <p style="color:#6b7280;font-size:0.85rem;margin-bottom:12px">Choose the embedding provider and model. Changing provider will re-generate all embeddings (may take a few minutes).</p>
    <label>Provider</label>
    <select id="set-emb-provider" style="width:100%;padding:8px 12px;border:1px solid #d1d5db;border-radius:6px;font-size:0.9rem;margin-bottom:8px" onchange="updateModelOptions()">
      <option value="gemini">Google (Gemini)</option>
      <option value="openai">OpenAI</option>
    </select>
    <label>Model</label>
    <select id="set-emb-model" style="width:100%;padding:8px 12px;border:1px solid #d1d5db;border-radius:6px;font-size:0.9rem;margin-bottom:8px"></select>
    <label>API Key (leave empty to use LLM key from .env)</label>
    <input type="password" id="set-emb-key" placeholder="API key for embedding provider" style="width:100%;padding:8px 12px;border:1px solid #d1d5db;border-radius:6px;font-size:0.9rem;margin-bottom:8px">
    <div id="emb-status" style="display:none;padding:8px;background:#f0f4ff;border-radius:6px;margin-bottom:8px;font-size:0.85rem"></div>
    <button class="btn btn-primary" id="btn-embedding" onclick="saveEmbedding()">Save Embedding Config</button>
    <hr style="margin:20px 0;border:none;border-top:1px solid #e5e7eb">
    <h3 style="margin-bottom:8px;font-size:0.9rem">Email Formatting</h3>
```

- [ ] **Step 2: Add the JavaScript functions**

Add before the `loadDocs()` call at the end of the script:

```javascript
const EMB_MODELS = {};

function updateModelOptions() {
  const provider = document.getElementById('set-emb-provider').value;
  const select = document.getElementById('set-emb-model');
  const models = EMB_MODELS[provider] || {};
  select.innerHTML = Object.keys(models).map(m =>
    '<option value="' + m + '">' + m + ' (' + models[m].dimensions + ' dims)</option>'
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
      }
    } catch(e) { clearInterval(interval); }
  }, 2000);
}
```

- [ ] **Step 3: Add `loadEmbeddingConfig()` to the initialization calls**

At the end of the script, where `loadDocs(); loadStats(); loadSettings();` are called, add:
```javascript
loadEmbeddingConfig();
```

- [ ] **Step 4: Update the existing Settings tab HTML structure**

Wrap the existing email settings with the `<h3>Email Formatting</h3>` header (the new embedding section comes first, then a divider, then email settings).

- [ ] **Step 5: Verify syntax**

Run: `python -c "import ast; ast.parse(open('app/admin/router.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add app/admin/router.py
git commit -m "feat: add embedding provider config UI with re-embed progress"
```

---

### Task 7: Add `input[type=password]` and `select` Styles to Admin CSS

**Files:**
- Modify: `app/admin/router.py` (CSS in ADMIN_HTML)

- [ ] **Step 1: Add missing CSS rules**

In the `<style>` section of ADMIN_HTML, add `select` to the existing input styles:

Change:
```css
input[type="text"], input[type="url"] { width: 100%; padding: 8px 12px; ...
```
To:
```css
input[type="text"], input[type="url"], input[type="password"], select { width: 100%; padding: 8px 12px; ...
```

- [ ] **Step 2: Commit**

```bash
git add app/admin/router.py
git commit -m "style: add password input and select to admin CSS"
```

---

### Task 8: Final Integration — Verify All Modules Work Together

**Files:**
- All modified files

- [ ] **Step 1: Verify all Python files parse correctly**

Run:
```bash
python -c "
import ast
for f in ['app/config.py', 'app/database.py', 'app/main.py', 'app/llm/provider.py', 'app/rag/embeddings.py', 'app/rag/reembed.py', 'app/rag/search.py', 'app/rag/ingest.py', 'app/admin/router.py']:
    ast.parse(open(f).read())
    print(f'  OK: {f}')
print('All files OK')
"
```
Expected: All files OK

- [ ] **Step 2: Verify JS syntax in admin HTML**

Extract the JS from the ADMIN_HTML and validate with node:
```bash
python3 -c "
import re
src = open('app/admin/router.py').read()
start = src.index('ADMIN_HTML = ')
html_start = src.index('\"\"\"', start) + 3
html_end = src.index('\"\"\"', html_start)
html = src[html_start:html_end]
match = re.search(r'<script>(.*?)</script>', html, re.DOTALL)
if match:
    js = match.group(1).replace('__ADMIN_SECRET__', 'test123')
    with open('/tmp/admin_js.js', 'w') as f:
        f.write(js)
    print(f'Extracted {len(js)} chars')
" && node --check /tmp/admin_js.js
```
Expected: No errors

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: multi-provider embeddings with admin UI configuration and auto re-embed"
```
