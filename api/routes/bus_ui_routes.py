"""Einfache Web-UI für KEI-Bus (Explorer/Schema Registry).
Liefert statisches HTML mit minimalem JS, das die
bestehenden Admin-Endpunkte konsumiert. Dient als DX-Hilfe.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/api/v1/bus/ui", tags=["KEI-Bus-UI"])


@router.get("/", response_class=HTMLResponse)
async def bus_ui() -> HTMLResponse:
    """Liefert eine einfache HTML-Seite mit Formularen/Aktionen für KEI-Bus."""
    html = """
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>KEI-Bus Explorer</title>
  <style>
    body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }
    h1 { margin-bottom: 8px; }
    section { border: 1px solid #ddd; padding: 16px; margin-bottom: 16px; border-radius: 8px; }
    label { display:block; font-weight: 600; margin-top: 8px; }
    input, textarea { width: 100%; padding: 8px; font-family: monospace; }
    button { margin-top: 8px; padding: 8px 12px; }
    pre { background:#f7f7f7; padding:12px; border-radius:6px; overflow:auto; }
    .row { display:flex; gap:16px; }
    .col { flex:1; }
  </style>
  <script>
    async function doPublish() {
      const subject = document.getElementById('pub_subject').value;
      const type = document.getElementById('pub_type').value;
      const tenant = document.getElementById('pub_tenant').value || null;
      const key = document.getElementById('pub_key').value || null;
      const payload = JSON.parse(document.getElementById('pub_payload').value || '{}');
      const headers = JSON.parse(document.getElementById('pub_headers').value || '{}');
      const res = await fetch('/api/v1/bus/publish', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject, type, tenant, key, payload, headers })
      });
      const txt = await res.text();
      document.getElementById('pub_out').textContent = txt;
    }
    async function listSchemas() {
      const res = await fetch('/api/v1/bus/admin/schemas');
      document.getElementById('schema_out').textContent = await res.text();
    }
    async function registerSchema() {
      const uri = document.getElementById('schema_uri').value;
      const schema = JSON.parse(document.getElementById('schema_body').value || '{}');
      const type = document.getElementById('schema_type').value || 'json';
      const compat = document.getElementById('schema_compat').value || 'backward';
      const res = await fetch('/api/v1/bus/admin/schemas/register', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uri, schema, schema_type: type, compatibility: compat })
      });
      document.getElementById('schema_out').textContent = await res.text();
    }
    async function listDLQ() {
      const res = await fetch('/api/v1/bus/admin/dlq');
      document.getElementById('dlq_out').textContent = await res.text();
    }
    async function showLatency() {
      const subject = document.getElementById('lat_subject').value;
      const res = await fetch('/api/v1/bus/admin/latency/percentiles' + (subject ? ('?subject=' + encodeURIComponent(subject)) : ''));
      document.getElementById('lat_out').textContent = await res.text();
    }
  </script>
</head>
<body>
  <h1>KEI‑Bus Explorer</h1>

  <section>
    <h2>Publish</h2>
    <div class="row">
      <div class="col">
        <label>Subject</label>
        <input id="pub_subject" placeholder="kei.events.tenant.bc.aggregate.event.v1" />
        <label>Type</label>
        <input id="pub_type" placeholder="domain_event" />
        <label>Tenant</label>
        <input id="pub_tenant" placeholder="tenant-id" />
        <label>Ordering Key</label>
        <input id="pub_key" placeholder="key-123" />
      </div>
      <div class="col">
        <label>Payload (JSON)</label>
        <textarea id="pub_payload" rows="8">{"ok": true}</textarea>
        <label>Headers (JSON)</label>
        <textarea id="pub_headers" rows="4">{}</textarea>
        <button onclick="doPublish()">Senden</button>
      </div>
    </div>
    <pre id="pub_out"></pre>
  </section>

  <section>
    <h2>Schema Registry</h2>
    <button onclick="listSchemas()">Schemas auflisten</button>
    <div class="row">
      <div class="col">
        <label>Schema URI</label>
        <input id="schema_uri" placeholder="schema:tenant:resource:v1" />
        <label>Typ</label>
        <input id="schema_type" placeholder="json|avro|protobuf" value="json" />
        <label>Kompatibilität</label>
        <input id="schema_compat" placeholder="backward|forward" value="backward" />
      </div>
      <div class="col">
        <label>Schema (JSON)</label>
        <textarea id="schema_body" rows="8">{"type":"object","properties":{"ok":{"type":"boolean"}}}</textarea>
        <button onclick="registerSchema()">Schema registrieren</button>
      </div>
    </div>
    <pre id="schema_out"></pre>
  </section>

  <section>
    <h2>DLQ</h2>
    <button onclick="listDLQ()">DLQ anzeigen</button>
    <pre id="dlq_out"></pre>
  </section>

  <section>
    <h2>Latenz (p50/p95/p99)</h2>
    <label>Subject (optional)</label>
    <input id="lat_subject" placeholder="kei.events.tenant.bc.aggregate.event.v1" />
    <button onclick="showLatency()">Abrufen</button>
    <pre id="lat_out"></pre>
  </section>

</body>
</html>
    """
    return HTMLResponse(html)


__all__ = ["router"]
