#!/usr/bin/env python3
"""
C2KV Playground Server

A lightweight interactive playground for the C2KV extract and chat endpoints.
Lets you freely compose multi-message requests, extract individual messages
into the KV cache, and send chat completions that reuse cached KV segments.

Usage:
    python c2kv/playground.py [--port 8080] [--sglang-url http://localhost:30000]
"""

import argparse
import sys

try:
    from flask import Flask, jsonify, render_template_string, request
    import requests as _req
except ImportError:
    print("Missing dependencies. Install with:\n  pip install flask requests")
    sys.exit(1)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# HTML / JS frontend
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>C2KV Playground</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e1e4e8; height: 100vh; display: flex; flex-direction: column; }

  /* ---- top bar ---- */
  #topbar {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 16px; background: #161b22; border-bottom: 1px solid #30363d;
    flex-shrink: 0;
  }
  #topbar h1 { font-size: 16px; font-weight: 700; color: #58a6ff; white-space: nowrap; }
  #topbar label { font-size: 12px; color: #8b949e; white-space: nowrap; }
  #sglang-url { flex: 1; background: #21262d; border: 1px solid #30363d; color: #e1e4e8; border-radius: 6px; padding: 5px 10px; font-size: 13px; }
  #topbar button { padding: 5px 14px; font-size: 13px; border-radius: 6px; cursor: pointer; border: none; }
  #btn-ping { background: #21262d; color: #8b949e; border: 1px solid #30363d; }
  #btn-ping:hover { border-color: #58a6ff; color: #58a6ff; }

  /* ---- main layout ---- */
  #layout { display: flex; flex: 1; overflow: hidden; }

  /* sidebar */
  #sidebar {
    width: 300px; min-width: 240px; background: #161b22; border-right: 1px solid #30363d;
    display: flex; flex-direction: column; overflow: hidden;
  }
  #sidebar-body { flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 14px; }

  /* center panel */
  #center { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #msg-header { display: flex; align-items: center; justify-content: space-between; padding: 10px 16px; border-bottom: 1px solid #30363d; flex-shrink: 0; }
  #msg-header h2 { font-size: 14px; font-weight: 600; }
  #messages { flex: 1; overflow-y: auto; padding: 10px 16px; display: flex; flex-direction: column; gap: 10px; }
  #send-bar { padding: 10px 16px; border-top: 1px solid #30363d; display: flex; align-items: center; gap: 10px; flex-shrink: 0; flex-wrap: wrap; }

  /* output panel */
  #output { width: 380px; min-width: 260px; background: #0d1117; border-left: 1px solid #30363d; display: flex; flex-direction: column; overflow: hidden; }
  #output-tabs { display: flex; border-bottom: 1px solid #30363d; flex-shrink: 0; }
  .otab { padding: 8px 16px; font-size: 12px; cursor: pointer; border-bottom: 2px solid transparent; color: #8b949e; }
  .otab.active { color: #58a6ff; border-bottom-color: #58a6ff; }
  .opane { display: none; flex: 1; overflow: auto; padding: 10px; }
  .opane.active { display: block; }
  .opane pre { font-size: 11px; white-space: pre-wrap; word-break: break-all; color: #e1e4e8; }

  /* ---- section headings ---- */
  .sec-head { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #8b949e; margin-bottom: 4px; }

  /* ---- extract form ---- */
  .ext-form textarea {
    width: 100%; height: 90px; background: #0d1117; border: 1px solid #30363d;
    border-radius: 6px; color: #e1e4e8; font-size: 12px; padding: 6px 8px; resize: vertical;
  }
  .ext-form textarea:focus { outline: none; border-color: #58a6ff; }
  .ext-row { display: flex; gap: 6px; align-items: center; margin-top: 6px; flex-wrap: wrap; }
  .ext-row select, .ext-row input[type=number] {
    background: #21262d; border: 1px solid #30363d; color: #e1e4e8;
    border-radius: 5px; padding: 4px 6px; font-size: 12px;
  }
  .ext-row input[type=number] { width: 56px; }

  /* ---- kv list ---- */
  #kv-list { display: flex; flex-direction: column; gap: 6px; }
  .kv-item {
    background: #21262d; border: 1px solid #30363d; border-radius: 6px; padding: 7px 9px; font-size: 11px;
    cursor: pointer; transition: border-color 0.15s;
  }
  .kv-item:hover { border-color: #58a6ff; }
  .kv-item .kv-hash { font-family: monospace; color: #79c0ff; font-size: 10px; word-break: break-all; }
  .kv-item .kv-meta { color: #8b949e; margin-top: 2px; }
  .kv-item .kv-preview { color: #c9d1d9; margin-top: 3px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
  .kv-item .kv-actions { display: flex; gap: 4px; margin-top: 5px; }

  /* ---- message row ---- */
  .msg-row {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px 12px;
    display: flex; flex-direction: column; gap: 7px;
  }
  .msg-row.has-c2kv { border-color: #388bfd; }
  .msg-top { display: flex; align-items: center; gap: 7px; }
  .msg-top select {
    background: #21262d; border: 1px solid #30363d; color: #e1e4e8;
    border-radius: 5px; padding: 4px 7px; font-size: 12px;
  }
  .msg-top .spacer { flex: 1; }
  .msg-body textarea {
    width: 100%; background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
    color: #e1e4e8; font-size: 12px; padding: 6px 8px; resize: vertical; min-height: 56px;
  }
  .msg-body textarea:focus { outline: none; border-color: #388bfd; }
  .c2kv-row { display: flex; align-items: center; gap: 6px; }
  .c2kv-row input {
    flex: 1; background: #0d1117; border: 1px solid #388bfd; border-radius: 5px;
    color: #79c0ff; font-size: 11px; padding: 4px 7px; font-family: monospace;
  }
  .c2kv-row input:focus { outline: none; }
  .c2kv-badge {
    font-size: 10px; background: #1f4068; border: 1px solid #388bfd; color: #79c0ff;
    border-radius: 4px; padding: 2px 6px; white-space: nowrap;
  }

  /* ---- buttons ---- */
  .btn { padding: 5px 12px; border-radius: 6px; border: none; cursor: pointer; font-size: 12px; font-weight: 500; transition: opacity 0.15s; }
  .btn:hover { opacity: 0.85; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary { background: #238636; color: #fff; }
  .btn-blue { background: #1f6feb; color: #fff; }
  .btn-sm { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 3px 8px; font-size: 11px; }
  .btn-danger { background: #21262d; color: #f85149; border: 1px solid #30363d; }
  .btn-extract-inline { background: #1f4068; color: #79c0ff; border: 1px solid #388bfd; padding: 3px 8px; font-size: 11px; border-radius: 5px; cursor: pointer; }
  .btn-extract-inline:hover { background: #1f6feb; }

  /* ---- param row ---- */
  .param-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .param-row label { font-size: 12px; color: #8b949e; }
  .param-row input[type=number] {
    width: 70px; background: #21262d; border: 1px solid #30363d; color: #e1e4e8;
    border-radius: 5px; padding: 4px 7px; font-size: 12px;
  }
  .param-row input[type=checkbox] { accent-color: #58a6ff; }

  /* ---- status ---- */
  #status { font-size: 11px; color: #8b949e; padding: 0 4px; }
  .status-ok { color: #3fb950; }
  .status-err { color: #f85149; }
  .status-busy { color: #d29922; }

  /* ---- empty state ---- */
  .empty { color: #484f58; font-size: 12px; text-align: center; padding: 14px 0; }

  /* scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
</head>
<body>

<!-- top bar -->
<div id="topbar">
  <h1>C2KV Playground</h1>
  <label for="sglang-url">SGLang URL</label>
  <input id="sglang-url" type="text" value="{{ default_url }}" placeholder="http://localhost:30000">
  <button id="btn-ping" onclick="ping()">Ping</button>
</div>

<!-- main layout -->
<div id="layout">

  <!-- sidebar: extract + kv cache -->
  <div id="sidebar">
    <div id="sidebar-body">

      <!-- extract form -->
      <div>
        <div class="sec-head">Extract to C2KV Cache</div>
        <div class="ext-form">
          <textarea id="ext-text" placeholder="Paste document text here…"></textarea>
          <div class="ext-row">
            <select id="ext-role">
              <option value="user">user</option>
              <option value="system">system</option>
              <option value="assistant">assistant</option>
            </select>
            <label style="font-size:11px;color:#8b949e">ratio</label>
            <input id="ext-ratio" type="number" value="4" min="1" max="32">
            <button class="btn btn-blue" style="flex:1" onclick="doExtract()">Extract</button>
          </div>
        </div>
      </div>

      <!-- kv cache list -->
      <div>
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
          <div class="sec-head" style="margin:0">Cached KVs</div>
          <button class="btn btn-sm" onclick="clearKVs()">Clear</button>
        </div>
        <div id="kv-list"><div class="empty">No cached KVs yet</div></div>
      </div>

    </div>
  </div>

  <!-- center: chat builder -->
  <div id="center">
    <div id="msg-header">
      <h2>Chat Messages</h2>
      <button class="btn btn-sm" onclick="addMessage()">+ Add Message</button>
    </div>
    <div id="messages"></div>
    <div id="send-bar">
      <div class="param-row">
        <label>max_tokens</label>
        <input id="max-tokens" type="number" value="256" min="1" max="4096">
        <label>temp</label>
        <input id="temperature" type="number" value="0.0" min="0" max="2" step="0.1" style="width:55px">
        <label><input id="enable-thinking" type="checkbox"> thinking</label>
      </div>
      <div style="flex:1"></div>
      <span id="status"></span>
      <button class="btn btn-sm" onclick="clearMessages()">Clear</button>
      <button class="btn btn-primary" onclick="sendChat()">Send Chat &#9654;</button>
    </div>
  </div>

  <!-- output panel -->
  <div id="output">
    <div id="output-tabs">
      <div class="otab active" onclick="switchTab('req')">Request</div>
      <div class="otab" onclick="switchTab('resp')">Response</div>
      <div class="otab" onclick="switchTab('ext')">Extract Log</div>
    </div>
    <div id="pane-req" class="opane active"><pre id="req-json">// request will appear here</pre></div>
    <div id="pane-resp" class="opane"><pre id="resp-json">// response will appear here</pre></div>
    <div id="pane-ext" class="opane"><pre id="ext-log">// extract logs will appear here</pre></div>
  </div>

</div>

<script>
// ---- state ----
const kvCache = []; // [{key_hash, role, text, gist_len, compression_ratio}]
let msgCounter = 0;

// ---- tab switching ----
function switchTab(name) {
  document.querySelectorAll('.otab').forEach((t, i) => {
    const names = ['req', 'resp', 'ext'];
    t.classList.toggle('active', names[i] === name);
  });
  document.querySelectorAll('.opane').forEach((p, i) => {
    const names = ['req', 'resp', 'ext'];
    p.classList.toggle('active', names[i] === name);
  });
}

// ---- status ----
function setStatus(msg, cls) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = cls || '';
}

// ---- ping ----
async function ping() {
  const url = document.getElementById('sglang-url').value.trim();
  setStatus('Pinging…', 'status-busy');
  try {
    const r = await fetch('/api/ping', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({sglang_url: url})
    });
    const d = await r.json();
    if (r.ok) setStatus('Server reachable ✓', 'status-ok');
    else setStatus('Server error: ' + d.error, 'status-err');
  } catch(e) { setStatus('Cannot reach server: ' + e.message, 'status-err'); }
}

// ---- KV cache ----
function renderKVList() {
  const el = document.getElementById('kv-list');
  if (kvCache.length === 0) { el.innerHTML = '<div class="empty">No cached KVs yet</div>'; return; }
  el.innerHTML = '';
  kvCache.forEach((kv, idx) => {
    const d = document.createElement('div');
    d.className = 'kv-item';
    d.innerHTML = `
      <div class="kv-hash">${kv.key_hash}</div>
      <div class="kv-meta">[${kv.role}] gist=${kv.gist_len} ratio=${kv.compression_ratio}</div>
      <div class="kv-preview">${escHtml(kv.text.slice(0, 120))}${kv.text.length > 120 ? '…' : ''}</div>
      <div class="kv-actions">
        <button class="btn btn-sm" onclick="copyHash(${idx})">Copy Hash</button>
        <button class="btn btn-sm" onclick="attachToLast(${idx})">Attach to Last Msg</button>
        <button class="btn btn-danger btn-sm" onclick="deleteKV(${idx})">✕</button>
      </div>`;
    el.appendChild(d);
  });
}

function copyHash(idx) {
  navigator.clipboard.writeText(kvCache[idx].key_hash);
  setStatus('Hash copied!', 'status-ok');
}

function attachToLast(idx) {
  const rows = document.querySelectorAll('.msg-row');
  if (rows.length === 0) { setStatus('No messages to attach to', 'status-err'); return; }
  const last = rows[rows.length - 1];
  enableC2KV(last, kvCache[idx].key_hash);
  setStatus('Attached hash to last message', 'status-ok');
}

function deleteKV(idx) {
  kvCache.splice(idx, 1);
  renderKVList();
}

function clearKVs() { kvCache.length = 0; renderKVList(); }

// ---- extract ----
async function doExtract(prefillText, prefillRole, targetRow) {
  const url = document.getElementById('sglang-url').value.trim();
  const text = prefillText !== undefined ? prefillText : document.getElementById('ext-text').value.trim();
  const role = prefillRole !== undefined ? prefillRole : document.getElementById('ext-role').value;
  const ratio = parseInt(document.getElementById('ext-ratio').value) || 4;
  if (!text) { setStatus('Enter text to extract', 'status-err'); return null; }

  setStatus('Extracting…', 'status-busy');
  switchTab('ext');
  try {
    const r = await fetch('/api/extract', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({sglang_url: url, text, role, compression_ratio: ratio})
    });
    const d = await r.json();
    const logEl = document.getElementById('ext-log');
    logEl.textContent = JSON.stringify(d, null, 2);
    if (!r.ok || !d.success) {
      setStatus('Extract failed: ' + (d.error || r.status), 'status-err');
      return null;
    }
    const entry = {key_hash: d.key_hash, gist_len: d.gist_len, compression_ratio: ratio, role, text};
    // dedup by key_hash
    if (!kvCache.find(k => k.key_hash === d.key_hash)) {
      kvCache.unshift(entry);
      renderKVList();
    }
    setStatus(`Extracted ✓  key=${d.key_hash.slice(0,12)}…`, 'status-ok');
    if (targetRow) enableC2KV(targetRow, d.key_hash);
    return d.key_hash;
  } catch(e) {
    setStatus('Extract error: ' + e.message, 'status-err');
    document.getElementById('ext-log').textContent = e.message;
    return null;
  }
}

// ---- messages ----
function addMessage(role, content, c2kvHash) {
  const id = ++msgCounter;
  const container = document.getElementById('messages');
  const row = document.createElement('div');
  row.className = 'msg-row';
  row.dataset.id = id;
  row.innerHTML = `
    <div class="msg-top">
      <select class="msg-role">
        <option value="user"${role==='user'?' selected':''}>user</option>
        <option value="system"${role==='system'?' selected':''}>system</option>
        <option value="assistant"${role==='assistant'?' selected':''}>assistant</option>
      </select>
      <span class="spacer"></span>
      <button class="btn btn-extract-inline" title="Extract this message into C2KV cache" onclick="extractRow(this)">⚡ Extract</button>
      <button class="btn btn-sm" title="Toggle C2KV reuse" onclick="toggleC2KV(this)">C2KV</button>
      <button class="btn btn-danger btn-sm" onclick="removeRow(this)">✕</button>
    </div>
    <div class="msg-body">
      <textarea class="msg-content" placeholder="Message content…" rows="3">${escHtml(content||'')}</textarea>
    </div>
    <div class="c2kv-row" style="display:none">
      <span class="c2kv-badge">c2kv_key_hash</span>
      <input class="msg-hash" type="text" placeholder="paste key_hash here or use ⚡ Extract above">
    </div>`;
  container.appendChild(row);
  if (c2kvHash) enableC2KV(row, c2kvHash);
  // auto-scroll
  container.scrollTop = container.scrollHeight;
  return row;
}

function enableC2KV(row, hash) {
  const c2kvRow = row.querySelector('.c2kv-row');
  const input = row.querySelector('.msg-hash');
  c2kvRow.style.display = 'flex';
  row.classList.add('has-c2kv');
  if (hash) input.value = hash;
}

function toggleC2KV(btn) {
  const row = btn.closest('.msg-row');
  const c2kvRow = row.querySelector('.c2kv-row');
  const visible = c2kvRow.style.display !== 'none';
  c2kvRow.style.display = visible ? 'none' : 'flex';
  row.classList.toggle('has-c2kv', !visible);
}

async function extractRow(btn) {
  const row = btn.closest('.msg-row');
  const text = row.querySelector('.msg-content').value.trim();
  const role = row.querySelector('.msg-role').value;
  if (!text) { setStatus('Message is empty', 'status-err'); return; }
  // put text in sidebar extract form too (for reference)
  document.getElementById('ext-text').value = text;
  document.getElementById('ext-role').value = role;
  await doExtract(text, role, row);
}

function removeRow(btn) {
  btn.closest('.msg-row').remove();
}

function clearMessages() {
  document.getElementById('messages').innerHTML = '';
  msgCounter = 0;
}

// ---- send chat ----
async function sendChat() {
  const url = document.getElementById('sglang-url').value.trim();
  const maxTokens = parseInt(document.getElementById('max-tokens').value) || 256;
  const temperature = parseFloat(document.getElementById('temperature').value) || 0.0;
  const enableThinking = document.getElementById('enable-thinking').checked;

  const rows = document.querySelectorAll('.msg-row');
  if (rows.length === 0) { setStatus('Add at least one message', 'status-err'); return; }

  const messages = [];
  for (const row of rows) {
    const role = row.querySelector('.msg-role').value;
    const content = row.querySelector('.msg-content').value;
    const c2kvRow = row.querySelector('.c2kv-row');
    const hashInput = row.querySelector('.msg-hash');
    const msg = {role, content};
    if (c2kvRow.style.display !== 'none' && hashInput.value.trim()) {
      msg.c2kv_key_hash = hashInput.value.trim();
    }
    messages.push(msg);
  }

  const payload = {
    sglang_url: url,
    model: 'default',
    messages,
    max_completion_tokens: maxTokens,
    temperature,
    chat_template_kwargs: {enable_thinking: enableThinking},
  };

  document.getElementById('req-json').textContent = JSON.stringify({
    url: url + '/v1/chat/completions',
    body: {model: payload.model, messages: payload.messages,
           max_completion_tokens: payload.max_completion_tokens,
           temperature: payload.temperature,
           chat_template_kwargs: payload.chat_template_kwargs}
  }, null, 2);
  switchTab('req');
  setStatus('Sending…', 'status-busy');

  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const d = await r.json();
    document.getElementById('resp-json').textContent = JSON.stringify(d, null, 2);
    switchTab('resp');
    if (r.ok && d.choices) {
      const content = d.choices[0]?.message?.content || '';
      setStatus(`Done ✓  ${d.usage?.total_tokens ?? '?'} tokens`, 'status-ok');
    } else {
      setStatus('Error: ' + (d.error?.message || r.status), 'status-err');
    }
  } catch(e) {
    setStatus('Request failed: ' + e.message, 'status-err');
    document.getElementById('resp-json').textContent = e.message;
    switchTab('resp');
  }
}

// ---- helpers ----
function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ---- preload sample messages on page load ----
window.addEventListener('load', () => {
  addMessage('user', 'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889 as the centerpiece of the 1889 World\'s Fair. '.repeat(10));
  addMessage('user', 'What is the Eiffel Tower? Give a brief summary.');
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template_string(_HTML, default_url=app.config["SGLANG_URL"])


@app.route("/api/ping", methods=["POST"])
def api_ping():
    data = request.json or {}
    sglang_url = data.get("sglang_url", app.config["SGLANG_URL"]).rstrip("/")
    try:
        r = _req.get(f"{sglang_url}/health", timeout=5)
        return jsonify({"ok": r.ok, "status": r.status_code})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/api/extract", methods=["POST"])
def api_extract():
    data = request.json or {}
    sglang_url = data.pop("sglang_url", app.config["SGLANG_URL"]).rstrip("/")
    try:
        r = _req.post(f"{sglang_url}/v1/c2kv/extract", json=data, timeout=60)
        return (r.text, r.status_code, {"Content-Type": "application/json"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 502


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    sglang_url = data.pop("sglang_url", app.config["SGLANG_URL"]).rstrip("/")
    # strip playground-only keys not meant for SGLang
    data.pop("model", None)
    payload = {
        "model": "default",
        "messages": data.get("messages", []),
        "max_completion_tokens": data.get("max_completion_tokens", 256),
        "temperature": data.get("temperature", 0.0),
        "chat_template_kwargs": data.get("chat_template_kwargs", {}),
    }
    try:
        r = _req.post(
            f"{sglang_url}/v1/chat/completions", json=payload, timeout=120
        )
        return (r.text, r.status_code, {"Content-Type": "application/json"})
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 502


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="C2KV Playground Server")
    parser.add_argument("--port", type=int, default=8080, help="Playground port (default: 8080)")
    parser.add_argument(
        "--sglang-url",
        default="http://localhost:30000",
        help="Default SGLang server URL (default: http://localhost:30000)",
    )
    args = parser.parse_args()

    app.config["SGLANG_URL"] = args.sglang_url.rstrip("/")

    print(f"C2KV Playground → http://localhost:{args.port}")
    print(f"Default SGLang   → {app.config['SGLANG_URL']}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
