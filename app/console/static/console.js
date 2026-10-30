/* HRD Intelligence AI console.
 * Talks to the AI service API with the console cookie (set by GET /console) and the acting user's
 * X-User-Id / X-User-Role headers, exactly like Spring would. No build step, no dependencies. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = {
  user: loadUser(),
  info: null,
  ready: null,
  docs: [],
  sessions: [],
  sessionId: null,
  allowed: new Map(), // sessionId -> Set of allowed document ids (what Spring would send)
  stats: null,
  traces: null,
  logKind: "chat",
  openTraceId: null,
  docStatus: "",
  docFilter: "",
  uploads: [],
  streaming: false,
  pollTimer: null,
};

const VIEWS = {
  overview: "Overview",
  assistant: "AI assistant",
  documents: "Data management",
  pipeline: "Pipeline log",
  models: "Model stack",
};

/* --------------------------------------------------------------------------- helpers */

/** Escape text for safe insertion into HTML. */
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

/** Milliseconds as "850 ms", "12.4 s" or "2m 05s"; "—" when unknown. */
function fmtMs(ms) {
  if (ms === null || ms === undefined || Number.isNaN(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)} s`;
  const s = Math.round(ms / 1000);
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

/** Thousands separators. */
function fmtNum(n) {
  return n === null || n === undefined ? "—" : Number(n).toLocaleString();
}

const rtf = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
/** "3 min ago" style time. */
function relTime(iso) {
  if (!iso) return "—";
  const diff = (new Date(iso).getTime() - Date.now()) / 1000;
  const steps = [[60, "second"], [3600, "minute"], [86400, "hour"], [604800, "day"], [Infinity, "week"]];
  const divisors = { second: 1, minute: 60, hour: 3600, day: 86400, week: 604800 };
  for (const [limit, unit] of steps) {
    if (Math.abs(diff) < limit) return rtf.format(Math.round(diff / divisors[unit]), unit);
  }
  return "";
}

/** Up to two initials for an avatar. */
function initials(name) {
  const parts = String(name || "?").replace(/[-_.]/g, " ").split(/\s+/).filter(Boolean);
  return ((parts[0]?.[0] || "?") + (parts[1]?.[0] || parts[0]?.[1] || "")).toUpperCase();
}

const AVATAR_COLORS = [["#fde7e7", "#b42318"], ["#e7f0fe", "#1d4ed8"], ["#e6f7ef", "#047857"], ["#fdf1dc", "#b45309"], ["#efe9ff", "#6d28d9"], ["#e0f7fa", "#0e7490"]];
/** Stable pastel colours for a name. */
function avatarColors(name) {
  let h = 0;
  for (const ch of String(name)) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return AVATAR_COLORS[h % AVATAR_COLORS.length];
}

const FILE_ICONS = { pdf: "i-file", docx: "i-file", txt: "i-file", md: "i-file", pptx: "i-slides", xlsx: "i-sheet", csv: "i-sheet", png: "i-image", jpg: "i-image", jpeg: "i-image" };
/** Coloured file-type tile. */
function fileTile(ext) {
  const e = String(ext || "").replace(".", "").toLowerCase();
  return `<span class="ftile ${esc(e)}"><svg class="icon" aria-hidden="true"><use href="#${FILE_ICONS[e] || "i-file"}"/></svg></span>`;
}

/** Inline SVG icon. */
function icon(id, cls = "icon") {
  return `<svg class="${cls}" aria-hidden="true"><use href="#${id}"/></svg>`;
}

/** Short-lived message at the bottom of the screen. */
function toast(message, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  $("#toasts").append(el);
  setTimeout(() => el.remove(), 4200);
}

/** The acting user (simulates the user Spring forwards), remembered in this browser. */
function loadUser() {
  try {
    const saved = JSON.parse(localStorage.getItem("console-user") || "null");
    if (saved?.id) return saved;
  } catch (e) { /* storage blocked: use the default */ }
  return { id: "admin-1", role: "ADMIN" };
}

/** Persist the acting user. */
function saveUser() {
  try { localStorage.setItem("console-user", JSON.stringify(state.user)); } catch (e) { /* not persisted */ }
}

/* --------------------------------------------------------------------------- API */

/** Call the AI service. Throws Error("<status>: <detail>") on failure. */
async function api(path, { method = "GET", json, body, raw = false } = {}) {
  const headers = { "X-User-Id": state.user.id, "X-User-Role": state.user.role };
  if (json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  }
  const res = await fetch(path, { method, headers, body, credentials: "same-origin" });
  if (res.status === 401) {
    sessionExpired();
    throw new Error("Console session expired");
  }
  if (raw) return res;
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch (e) { /* keep status text */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.status === 204 ? null : res.json();
}

/** The service restarted (new console secret): reload once to get a fresh cookie. */
function sessionExpired() {
  try {
    if (sessionStorage.getItem("console-reloaded")) return toast("Console session expired - reload the page", "bad");
    sessionStorage.setItem("console-reloaded", "1");
  } catch (e) { /* fall through to reload */ }
  location.reload();
}

/** Parse a Server-Sent Events response body (POST + fetch, so EventSource cannot be used). */
async function* readSSE(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let event = "message";
      const data = [];
      for (const line of block.split("\n")) {
        if (line.startsWith(":")) continue;
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (data.length) yield { event, data: JSON.parse(data.join("\n")) };
    }
  }
}

/* --------------------------------------------------------------------------- pipeline stages */

const CHAT_STAGES = [
  { label: "Prepare", color: "var(--st-prepare)", get: (t) => t.prepare_ms, note: (t) => `${t.allowed_documents ?? 0} allowed docs` },
  { label: "Rewrite query", color: "var(--st-rewrite)", get: (t) => t.rewrite_ms, note: (t) => (t.rewritten ? "follow-up rewritten" : "not needed") },
  { label: "Embed query", color: "var(--st-embed)", get: (t) => t.embed_ms, note: () => "bge-m3" },
  { label: "Vector search", color: "var(--st-dense)", get: (t) => t.dense_ms, note: (t) => `${t.dense_hits ?? 0} hits · HNSW` },
  { label: "Keyword search", color: "var(--st-lexical)", get: (t) => t.lexical_ms, note: (t) => `${t.lexical_hits ?? 0} hits · full-text` },
  { label: "Fuse (RRF)", color: "var(--st-fuse)", get: (t) => t.fuse_ms, note: (t) => `${t.candidates ?? 0} candidates` },
  { label: "Rerank", color: "var(--st-rerank)", get: (t) => t.rerank_ms, note: (t) => (t.reranked === false ? "skipped" : `${t.kept ?? (Array.isArray(t.sources) ? t.sources.length : 0)} kept`) },
  { label: "LLM first token", color: "var(--st-ttft)", get: (t) => t.llm_ttft_ms, note: (t) => `${fmtNum(t.input_tokens)} prompt tokens` },
  { label: "Generate answer", color: "var(--st-generate)", get: (t) => (t.generate_ms != null ? Math.max(0, t.generate_ms - (t.llm_ttft_ms || 0)) : null),
    note: (t) => (t.output_tokens ? `${t.output_tokens} tokens · ${t.tokens_per_s ?? "—"} tok/s` : "") },
];

const INGEST_STAGES = [
  { label: "Queue wait", color: "var(--st-queue)", get: (t) => t.queue_ms, note: (t) => (t.attempt > 1 ? `attempt ${t.attempt}` : "") },
  { label: "Parse & extract", color: "var(--st-extract)", get: (t) => t.extract_ms ?? t.parse_ms, note: (t) => `${t.blocks ?? 0} blocks` },
  { label: "OCR", color: "var(--st-ocr)", get: (t) => t.ocr?.ms, note: (t) => (t.ocr ? ocrSummary(t.ocr) : "") },
  { label: "Chunk", color: "var(--st-chunk)", get: (t) => t.chunk_ms, note: (t) => `${t.chunks ?? 0} chunks · ${fmtNum(t.tokens)} tokens` },
  { label: "Embed", color: "var(--st-embed)", get: (t) => t.embed_ms, note: (t) => (t.embed_chunks_per_s ? `${t.embed_chunks_per_s} chunks/s` : "") },
  { label: "Store", color: "var(--st-store)", get: (t) => t.store_ms, note: () => "pgvector" },
];

/** "2 pages · PaddleOCR ×1, Surya ×1". */
function ocrSummary(ocr) {
  const names = { paddle: "PaddleOCR", surya: "Surya", tesseract: "Tesseract", ppstructure: "PP-Structure" };
  const engines = Object.entries(ocr.engines || {}).map(([k, v]) => `${names[k] || k} ×${v}`).join(", ");
  return `${ocr.pages} page${ocr.pages === 1 ? "" : "s"} · ${engines}`;
}

/** Stage rows with values for one trace; adds the remainder as "Save & cite" for chat answers. */
function stageRows(stages, trace, total) {
  const rows = stages
    .map((s) => ({ label: s.label, color: s.color, ms: s.get(trace), note: s.note(trace) }))
    .filter((r) => typeof r.ms === "number");
  if (total && stages === CHAT_STAGES) {
    const rest = total - rows.reduce((a, r) => a + r.ms, 0);
    if (rest > 0) rows.push({ label: "Save & cite", color: "var(--st-finalize)", ms: rest, note: "database + citations" });
  }
  return rows;
}

/** Waterfall: one bar per stage, offset by the stages before it (the pipeline runs in sequence). */
function waterfallHTML(rows, caption) {
  const total = rows.reduce((a, r) => a + r.ms, 0) || 1;
  let start = 0;
  const body = rows.map((r) => {
    const left = (start / total) * 100;
    const width = Math.max((r.ms / total) * 100, 0.6);
    start += r.ms;
    return `<tr><th scope="row"><span class="swatch-dot" style="background:${r.color}"></span>${esc(r.label)}</th>
      <td><div class="wf-track"><span class="wf-bar" style="left:${left}%;width:${width}%;background:${r.color}"></span></div></td>
      <td class="wf-val">${fmtMs(r.ms)}${r.note ? `<span class="wf-note">${esc(r.note)}</span>` : ""}</td></tr>`;
  }).join("");
  return `<table class="waterfall"><caption class="visually-hidden">${esc(caption)}</caption><tbody>${body}</tbody></table>`;
}

/** Thin stacked bar showing the share of each stage. */
function stripHTML(rows) {
  const total = rows.reduce((a, r) => a + r.ms, 0) || 1;
  return `<div class="strip" aria-hidden="true">${rows.map((r) => `<span style="width:${(r.ms / total) * 100}%;background:${r.color}" title="${esc(r.label)}: ${fmtMs(r.ms)}"></span>`).join("")}</div>`;
}

/** CSS class for a timing cell. */
function msClass(ms, slow = 30000, mid = 15000) {
  if (ms == null) return "ms dim";
  return ms >= slow ? "ms slow" : ms >= mid ? "ms mid" : "ms";
}

/* --------------------------------------------------------------------------- charts */

/** Semicircle gauge with a percentage. */
function gaugeSVG(pct, color, label) {
  const p = Math.max(0, Math.min(1, pct || 0));
  const len = Math.PI * 26;
  return `<svg class="gauge" viewBox="0 0 64 38" role="img" aria-label="${esc(label)}: ${Math.round(p * 100)}%">
    <path d="M6 34 A26 26 0 0 1 58 34" fill="none" stroke="var(--track)" stroke-width="6" stroke-linecap="round"/>
    <path d="M6 34 A26 26 0 0 1 58 34" fill="none" stroke="${color}" stroke-width="6" stroke-linecap="round" stroke-dasharray="${len * p} ${len}"/>
    <text x="32" y="33" text-anchor="middle">${Math.round(p * 100)}%</text></svg>`;
}

/** Donut chart plus a legend that carries the same numbers as text (accessible alternative). */
function donutHTML(segments, centerValue, centerLabel, format = fmtMs) {
  const total = segments.reduce((a, s) => a + s.value, 0);
  if (!total) return `<p class="empty">No data yet.</p>`;
  const r = 60;
  const c = 2 * Math.PI * r;
  let offset = 0;
  const arcs = segments.map((s) => {
    const len = (s.value / total) * c;
    const arc = `<circle r="${r}" cx="84" cy="84" fill="none" stroke="${s.color}" stroke-width="18" stroke-dasharray="${Math.max(len - 2, 0.5)} ${c}" stroke-dashoffset="${-offset}" transform="rotate(-90 84 84)"/>`;
    offset += len;
    return arc;
  }).join("");
  const legend = segments.map((s) => `<li><span class="swatch" style="background:${s.color}"></span>${esc(s.label)}<span class="val">${format(s.value)}</span><span class="pct">${Math.round((s.value / total) * 100)}%</span></li>`).join("");
  return `<svg class="donut" viewBox="0 0 168 168" aria-hidden="true"><circle r="${r}" cx="84" cy="84" fill="none" stroke="var(--track)" stroke-width="18"/>${arcs}
    <text x="84" y="82" text-anchor="middle" font-size="22" font-weight="700">${esc(centerValue)}</text>
    <text x="84" y="102" text-anchor="middle" font-size="10" fill="var(--muted)" letter-spacing="1">${esc(centerLabel)}</text></svg>
    <ul class="legend" role="list">${legend}</ul>`;
}

/* --------------------------------------------------------------------------- routing */

/** Show the view named in the URL hash. */
function route(focus = true) {
  const name = VIEWS[location.hash.slice(1)] ? location.hash.slice(1) : "overview";
  for (const section of $$(".view")) section.hidden = section.dataset.view !== name;
  for (const link of $$("[data-nav]")) {
    if (link.dataset.nav === name) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  $("#crumb-current").textContent = VIEWS[name];
  document.title = `${VIEWS[name]} | HRD Intelligence AI Console`;
  if (focus) $("#content").focus({ preventScroll: true });
  const loaders = { overview: loadOverview, assistant: loadAssistant, documents: loadDocuments, pipeline: loadPipeline, models: loadModels };
  loaders[name]().catch((e) => toast(e.message, "bad"));
}

/* --------------------------------------------------------------------------- theme */

/** Two-state light/dark toggle: flips what is shown; returning to the system choice clears the override. */
function setupThemeToggle() {
  const meta = $('meta[name="color-scheme"]');
  const system = matchMedia("(prefers-color-scheme: dark)");
  $("#theme-toggle").addEventListener("click", () => {
    const rendered = meta.content === "light dark" ? (system.matches ? "dark" : "light") : meta.content;
    const target = rendered === "dark" ? "light" : "dark";
    const systemScheme = system.matches ? "dark" : "light";
    try {
      if (target === systemScheme) localStorage.removeItem("color-scheme");
      else localStorage.setItem("color-scheme", target);
    } catch (e) { /* applied without persisting */ }
    meta.content = target === systemScheme ? "light dark" : target;
  });
  addEventListener("storage", (e) => {
    if (e.key === "color-scheme") meta.content = e.newValue ?? "light dark";
  });
}

/* --------------------------------------------------------------------------- shared data */

/** Load documents and keep polling while something is indexing (updates the nav badge too). */
async function refreshDocs() {
  state.docs = await api("/api/v1/documents?limit=500");
  const indexing = state.docs.filter((d) => d.status === "pending" || d.status === "processing").length;
  const badge = $("#nav-indexing");
  badge.hidden = !indexing;
  badge.textContent = indexing;
  clearTimeout(state.pollTimer);
  if (indexing) {
    state.pollTimer = setTimeout(async () => {
      await refreshDocs().catch(() => {});
      const view = location.hash.slice(1) || "overview";
      if (view === "documents") renderDocuments();
      if (view === "overview") loadOverview().catch(() => {});
    }, 3000);
  }
  return state.docs;
}

/** Service readiness (database, LLM, embeddings, reranker) shown in the top bar. */
async function refreshHealth() {
  try {
    state.ready = await (await fetch("/ready")).json();
  } catch (e) {
    state.ready = { status: "down", checks: {} };
  }
  const checks = state.ready.checks || {};
  const down = Object.entries(checks).filter(([, v]) => v !== "ok" && v !== "disabled").map(([k]) => k);
  const pill = $("#health-pill");
  $(".dot", pill).className = `dot ${down.length ? (checks.database === "ok" ? "warn" : "bad") : "ok"}`;
  $(".health-text", pill).textContent = down.length ? `${down.join(", ")} unavailable` : "All services ready";
}

/** Readiness of one component ("ok", "disabled" or an error text). */
function checkState(key) {
  const v = state.ready?.checks?.[key];
  if (v === "ok") return ["ok", "Ready"];
  if (v === "disabled") return ["idle", "Disabled"];
  return v ? ["bad", "Unavailable"] : ["idle", "Unknown"];
}

/* --------------------------------------------------------------------------- overview */

/** Overview page: KPIs, answer-time breakdown, recent answers, model stack, indexing. */
async function loadOverview() {
  const [stats, traces, , info] = await Promise.all([
    api("/api/v1/stats"), api("/api/v1/traces?limit=6"), refreshDocs(), state.info ? state.info : api("/api/v1/info"),
  ]);
  state.stats = stats;
  state.info = info;
  const hour = new Date().getHours();
  $("#greeting").textContent = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
  const cs = stats.chat.stages;
  const is = stats.ingestion.stages;

  // banner
  $("#banner-title").textContent = stats.chat.answers ? "Pipeline summary is ready" : "No answers yet";
  $("#banner-text").textContent = stats.chat.answers
    ? `${info.llm.model} answered ${stats.chat.answers} questions · average ${fmtMs(cs.total_ms.avg)}, first token after ${fmtMs(cs.first_token_ms.avg)} · indexing takes ${fmtMs(is.total_ms.avg)} per document.`
    : "Upload documents in Data management, then ask the assistant - every stage will be timed here.";

  // KPIs
  const by = stats.documents.by_status;
  const ready = by.ready?.count || 0;
  const indexing = (by.pending?.count || 0) + (by.processing?.count || 0);
  const failed = by.failed?.count || 0;
  const total = stats.documents.total || 0;
  const pct = (n) => (total ? (n / total) * 100 : 0);
  const llmShare = cs.total_ms.avg ? ((cs.llm_ttft_ms.avg || 0) + Math.max(0, (cs.generate_ms.avg || 0) - (cs.llm_ttft_ms.avg || 0))) / cs.total_ms.avg : 0;
  const retrievalShare = cs.total_ms.avg ? ((cs.embed_ms.avg || 0) + (cs.dense_ms.avg || 0) + (cs.lexical_ms.avg || 0) + (cs.rerank_ms.avg || 0)) / cs.total_ms.avg : 0;
  const ocrShare = is.total_ms.avg ? (is.ocr_ms.avg || 0) / is.total_ms.avg : 0;
  const kpi = (tile, ic, title, sub, value, foot, gauge = "") => `<article class="card kpi">
      <div class="kpi-top"><span class="tile ${tile}">${icon(ic)}</span><div><h2>${title}</h2><p>${sub}</p></div></div>
      <div class="kpi-bottom"><div><div class="kpi-value">${value}</div><div class="kpi-sub">${foot}</div></div>${gauge}</div></article>`;
  const big = (ms) => {
    if (ms == null) return "—";
    return ms < 1000 ? `${Math.round(ms)}<small>ms</small>` : `${(ms / 1000).toFixed(1)}<small>s</small>`;
  };
  $("#kpis").innerHTML = `
    <article class="card kpi featured">
      <div class="kpi-top"><span class="tile">${icon("i-folder")}</span><div><h2>Documents</h2><p>Uploaded to the AI service</p></div></div>
      <div class="kpi-value">${fmtNum(total)}</div>
      <div class="stackbar" aria-hidden="true"><span style="width:${pct(ready)}%;background:#4ade80"></span><span style="width:${pct(indexing)}%;background:#7dd3fc"></span><span style="width:${pct(failed)}%;background:#fca5a5"></span></div>
      <div class="legend-inline"><span><i style="background:#4ade80"></i>${ready} ready</span><span><i style="background:#7dd3fc"></i>${indexing} indexing</span><span><i style="background:#fca5a5"></i>${failed} failed</span></div>
    </article>
    ${kpi("green", "i-layers", "Chunks indexed", "Searchable passages", fmtNum(stats.documents.chunks), `${info.embeddings.model} · ${info.embeddings.dim}-dim`, gaugeSVG(total ? ready / total : 0, "var(--ok)", "Documents ready"))}
    ${kpi("blue", "i-clock", "Avg answer time", "Question to full answer", big(cs.total_ms.avg), `median ${fmtMs(cs.total_ms.p50)} · ${stats.chat.answers} answers`, gaugeSVG(llmShare, "var(--info)", "Share spent in the LLM"))}
    ${kpi("amber", "i-zap", "First token", "Until text starts streaming", big(cs.first_token_ms.avg), `LLM alone ${fmtMs(cs.llm_ttft_ms.avg)}`, gaugeSVG(retrievalShare, "var(--warn)", "Share spent in retrieval"))}
    ${kpi("violet", "i-scan", "Avg indexing time", "Upload to searchable", big(is.total_ms.avg), `OCR ${fmtMs(is.ocr_ms.avg)} · embed ${fmtMs(is.embed_ms.avg)}`, gaugeSVG(ocrShare, "var(--violet-500)", "Share spent in OCR"))}`;

  // where answer time goes
  const seg = (label, color, v) => ({ label, color, value: Math.max(0, v || 0) });
  $("#answer-donut").innerHTML = donutHTML([
    seg("Prepare & rewrite", "var(--st-rewrite)", (cs.prepare_ms.avg || 0) + (cs.rewrite_ms.avg || 0)),
    seg("Embed query", "var(--st-embed)", cs.embed_ms.avg),
    seg("Search (vector + keyword)", "var(--st-dense)", (cs.dense_ms.avg || 0) + (cs.lexical_ms.avg || 0) + (cs.fuse_ms.avg || 0)),
    seg("Rerank", "var(--st-rerank)", cs.rerank_ms.avg),
    seg("LLM first token", "var(--st-ttft)", cs.llm_ttft_ms.avg),
    seg("Generate answer", "var(--st-generate)", (cs.generate_ms.avg || 0) - (cs.llm_ttft_ms.avg || 0)),
  ].filter((s) => s.value > 0), fmtMs(cs.total_ms.avg), "AVG TOTAL");

  // recent answers
  $("#recent-answers").innerHTML = traces.chat.slice(0, 5).map((t) => {
    const [bg, fg] = avatarColors(t.user_id);
    const ms = t.trace.total_ms;
    const badge = ms <= 10000 ? ["ok", "Fast"] : ms <= 30000 ? ["warn", "Normal"] : ["bad", "Slow"];
    return `<li><button type="button" class="row-btn" data-trace="${t.id}">
      <span class="mini-avatar" style="background:${bg};color:${fg}">${esc(initials(t.user_id))}</span>
      <span class="grow"><span class="title">${esc(t.question || t.title || "Question")}</span><br>
      <span class="meta">${fmtMs(ms)} total · first token ${fmtMs(t.trace.first_token_ms)} · ${(t.trace.sources || []).length} sources</span></span>
      <span class="badge ${badge[0]}">${badge[1]}</span>${icon("i-chevron-right", "icon icon-sm")}</button></li>`;
  }).join("") || `<li class="empty">No answers yet.</li>`;

  // model stack
  renderStackList();

  // indexing stages
  const ingest = [
    ["Queue wait", "var(--st-queue)", is.queue_ms.avg], ["Parse & extract", "var(--st-extract)", is.extract_ms.avg], ["OCR", "var(--st-ocr)", is.ocr_ms.avg],
    ["Chunk", "var(--st-chunk)", is.chunk_ms.avg], ["Embed", "var(--st-embed)", is.embed_ms.avg], ["Store", "var(--st-store)", is.store_ms.avg],
  ];
  const max = Math.max(...ingest.map((r) => r[2] || 0), 1);
  $("#ingest-bars").innerHTML = stats.ingestion.runs
    ? `<div class="hbars">${ingest.map(([l, c, v]) => `<div><div class="hbar-label"><span>${l}</span><span>${fmtMs(v)}</span></div>
        <div class="hbar-track" aria-hidden="true"><div class="hbar-fill" style="width:${((v || 0) / max) * 100}%;background:${c}"></div></div></div>`).join("")}</div>`
    : `<p class="empty">No indexing runs yet.</p>`;

  // documents by type
  const types = {};
  for (const d of state.docs) types[d.extension] = (types[d.extension] || 0) + 1;
  const typeColors = { ".pdf": "var(--red-600)", ".docx": "var(--blue-600)", ".pptx": "var(--amber-500)", ".xlsx": "var(--green-600)", ".csv": "var(--teal-600)",
    ".txt": "var(--slate-500)", ".md": "var(--violet-500)", ".png": "var(--pink-600)", ".jpg": "#be185d", ".jpeg": "#be185d" };
  $("#type-donut").innerHTML = donutHTML(
    Object.entries(types).sort((a, b) => b[1] - a[1]).map(([ext, n]) => ({ label: ext.slice(1).toUpperCase(), color: typeColors[ext] || "var(--slate-500)", value: n })),
    String(state.docs.length), "FILES", (v) => String(v));

  // recently indexed
  const recent = [...state.docs].sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at)).slice(0, 5);
  $("#recent-docs").innerHTML = recent.map((d) => `<li><button type="button" class="row-btn" data-doc="${esc(d.id)}">${fileTile(d.extension)}
      <span class="grow"><span class="title">${esc(d.filename)}</span><br><span class="meta">${esc(d.folder_id || "—")} · ${d.chunk_count} chunks · ${fmtMs(d.trace?.total_ms)}</span></span>
      ${statusBadge(d.status)}</button></li>`).join("") || `<li class="empty">No documents yet.</li>`;
}

/** Overview "Model stack" list. */
function renderStackList() {
  const info = state.info;
  if (!info) return;
  const rows = [
    ["i-sparkles", "LLM", info.llm.model, `Ollama/vLLM · thinking ${info.llm.thinking} · ${fmtNum(info.llm.context_window)} ctx`, checkState("llm")],
    ["i-layers", "Embeddings", info.embeddings.model, `${info.embeddings.dim}-dim vectors`, checkState("embeddings")],
    ["i-gauge", "Reranker", info.reranker.model, `${info.reranker.candidates} candidates → top ${info.retrieval.context_top_k}`, checkState("reranker")],
    ["i-scan", "OCR", info.ocr.mode === "auto" ? "PaddleOCR → Surya" : info.ocr.mode, info.ocr.surya_installed ? "Surya worker installed" : "Surya not installed", info.ocr.mode === "auto" && !info.ocr.surya_installed ? ["warn", "Partial"] : ["ok", "Ready"]],
    ["i-database", "Vector DB", "pgvector", "HNSW halfvec + GIN full-text", checkState("database")],
  ];
  $("#stack-list").innerHTML = rows.map(([ic, role, name, meta, [cls, text]], i) => `<li>
      <span class="rank">${i + 1}</span><span class="tile ${["violet", "green", "amber", "pink", "blue"][i]}">${icon(ic)}</span>
      <span><span class="title">${esc(name)}</span><br><span class="meta">${esc(role)} · ${esc(meta)}</span></span>
      <span class="badge ${cls}">${text}</span></li>`).join("");
}

/* --------------------------------------------------------------------------- documents */

/** Status pill for a document. */
function statusBadge(status) {
  const map = { ready: ["ok", "Ready"], processing: ["info", "Indexing"], pending: ["idle", "Queued"], failed: ["bad", "Failed"] };
  const [cls, text] = map[status] || ["idle", status];
  return `<span class="badge ${cls}">${text}</span>`;
}

/** Data management page. */
async function loadDocuments() {
  await refreshDocs();
  renderDocuments();
}

/** Render tiles, folder options, the table and the upload queue from state.docs. */
function renderDocuments() {
  const docs = state.docs;
  const count = (fn) => docs.filter(fn).length;
  const tiles = [
    ["i-check", "Ready", count((d) => d.status === "ready"), "var(--ok)"],
    ["i-loader", "Indexing", count((d) => d.status === "pending" || d.status === "processing"), "var(--info)"],
    ["i-alert", "Failed", count((d) => d.status === "failed"), "var(--bad)"],
    ["i-layers", "Chunks", docs.reduce((a, d) => a + d.chunk_count, 0), "var(--violet-500)"],
  ];
  $("#status-tiles").innerHTML = tiles.map(([ic, l, v, c]) => `<div class="status-tile"><span class="l" style="color:${c}">${icon(ic, "icon icon-sm")}<span style="color:var(--muted)">${l}</span></span><span class="v">${fmtNum(v)}</span></div>`).join("");

  const folders = [...new Set(docs.map((d) => d.folder_id).filter(Boolean))].sort();
  $("#folder-options").innerHTML = folders.map((f) => `<option value="${esc(f)}"></option>`).join("");

  const term = state.docFilter.toLowerCase();
  const rows = docs.filter((d) => {
    if (state.docStatus === "indexing" && !(d.status === "pending" || d.status === "processing")) return false;
    if (state.docStatus && state.docStatus !== "indexing" && d.status !== state.docStatus) return false;
    return !term || d.filename.toLowerCase().includes(term) || (d.folder_id || "").toLowerCase().includes(term);
  });
  $("#doc-table-title").textContent = `Documents (${rows.length})`;
  $("#doc-table tbody").innerHTML = rows.map((d) => {
    const ocr = d.trace?.ocr;
    const ocrCell = ocr ? Object.keys(ocr.engines || {}).map((e) => `<span class="chip ${e === "surya" ? "primary" : ""}">${e === "paddle" ? "PaddleOCR" : e === "surya" ? "Surya" : esc(e)}</span>`).join(" ") : `<span class="muted">—</span>`;
    return `<tr data-doc="${esc(d.id)}">
      <td><div class="doc-cell">${fileTile(d.extension)}<div class="grow"><button type="button" class="doc-name" data-doc="${esc(d.id)}">${esc(d.filename)}</button><span class="doc-id">${esc(d.id)}</span>
        ${d.status === "failed" && d.error ? `<span class="err" title="${esc(d.error)}">${esc(d.error)}</span>` : ""}</div></div></td>
      <td>${esc(d.folder_id || "—")}</td><td>${statusBadge(d.status)}</td>
      <td class="num">${d.page_count ?? "—"}</td><td class="num">${fmtNum(d.chunk_count)}</td><td>${ocrCell}</td>
      <td class="num"><span class="${msClass(d.trace?.total_ms, 120000, 30000)}">${fmtMs(d.trace?.total_ms)}</span></td>
      <td class="muted">${relTime(d.updated_at)}</td></tr>`;
  }).join("") || `<tr><td colspan="8" class="empty">No documents match.</td></tr>`;

  renderUploads();
}

/** Upload progress list (follows each file until it is indexed). */
function renderUploads() {
  $("#upload-queue").innerHTML = state.uploads.slice(-6).reverse().map((u) => {
    const doc = state.docs.find((d) => d.id === u.id);
    let status = u.error ? `<span class="badge bad">Failed</span>` : u.phase === "uploading" ? `<span class="badge info">Uploading</span>` : doc ? statusBadge(doc.status) : `<span class="badge idle">Queued</span>`;
    const detail = u.error || doc?.error || (doc?.status === "ready" ? `${doc.chunk_count} chunks · indexed in ${fmtMs(doc.trace?.total_ms)}` : doc?.status === "processing" ? "Parsing, OCR, chunking and embedding…" : u.note || "Waiting for the worker…");
    const spinner = !u.error && (!doc || doc.status === "pending" || doc.status === "processing") ? icon("i-loader", "icon icon-sm spin") : "";
    return `<li>${fileTile(u.ext)}<div class="grow"><div class="name">${esc(u.name)}</div><div class="muted small">${esc(detail)}</div></div>${spinner}${status}</li>`;
  }).join("");
}

/** Upload files to the chosen folder (one request per file, like Spring would). */
async function uploadFiles(files) {
  const folder = $("#upload-folder").value.trim() || "general";
  const allowed = [".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".md", ".png", ".jpg", ".jpeg"];
  for (const file of files) {
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    const id = `${file.name.toLowerCase().replace(/\.[^.]+$/, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40)}-${crypto.randomUUID().slice(0, 6)}`;
    const item = { id, name: file.name, ext, phase: "uploading" };
    state.uploads.push(item);
    renderUploads();
    if (!allowed.includes(ext)) {
      item.error = `Unsupported type ${ext}`;
      renderUploads();
      continue;
    }
    const form = new FormData();
    form.append("file", file);
    form.append("document_id", id);
    form.append("owner_id", state.user.id);
    form.append("folder_id", folder);
    try {
      const doc = await api("/api/v1/documents", { method: "POST", body: form });
      item.phase = "queued";
      if (!doc.queued) item.note = "Same content was already indexed";
    } catch (e) {
      item.error = e.message;
    }
    renderUploads();
  }
  await refreshDocs();
  renderDocuments();
}

/** Open the side drawer with a document's ingestion trace and chunks. */
async function openDoc(id) {
  const dialog = $("#doc-dialog");
  $("#doc-dialog-title").textContent = "Loading…";
  $("#doc-dialog-sub").textContent = "";
  $("#doc-dialog-body").innerHTML = `<p class="empty">${icon("i-loader", "icon spin")}</p>`;
  if (!dialog.open) dialog.showModal();
  try {
    const [doc, chunks] = await Promise.all([api(`/api/v1/documents/${encodeURIComponent(id)}`), api(`/api/v1/documents/${encodeURIComponent(id)}/chunks?limit=300`)]);
    renderDocDrawer(doc, chunks);
  } catch (e) {
    $("#doc-dialog-body").innerHTML = `<div class="alert">${icon("i-alert")}${esc(e.message)}</div>`;
  }
}

/** Drawer content for one document. */
function renderDocDrawer(doc, chunks) {
  const t = doc.trace || {};
  $("#doc-dialog-title").textContent = doc.filename;
  $("#doc-dialog-sub").textContent = `${doc.id} · folder ${doc.folder_id || "—"} · ${(doc.size_bytes / 1024).toFixed(1)} KB · owner ${doc.owner_id}`;
  const rows = stageRows(INGEST_STAGES, t);
  const ocr = t.ocr;
  const ocrPages = ocr ? `<ul class="ocr-pages" role="list">${ocr.per_page.map((p, i) => `<li><span class="chip">Page ${i + 1}</span>
      <span class="chip ${p.engine === "surya" ? "primary" : ""}">${p.engine === "paddle" ? "PaddleOCR" : p.engine === "surya" ? "Surya" : esc(p.engine)}</span>
      <span>${fmtMs(p.ms)}</span><span class="muted">confidence ${p.confidence ?? "—"}</span>${p.khmer ? `<span class="chip">Khmer text</span>` : ""}
      ${p.fallback_reason ? `<span class="muted">· fallback: ${esc(p.fallback_reason)}</span>` : ""}</li>`).join("")}</ul>` : "";
  $("#doc-dialog-body").innerHTML = `
    <div class="drawer-kpis">
      <div><div class="v">${doc.page_count ?? "—"}</div><div class="l">Pages</div></div>
      <div><div class="v">${fmtNum(doc.chunk_count)}</div><div class="l">Chunks</div></div>
      <div><div class="v">${fmtNum(t.tokens)}</div><div class="l">Tokens</div></div>
      <div><div class="v">${fmtMs(t.total_ms)}</div><div class="l">Index time</div></div>
    </div>
    ${doc.status === "failed" ? `<div class="alert">${icon("i-alert")}<span>${esc(doc.error || "Indexing failed")}</span></div>` : ""}
    <div class="drawer-actions">
      ${statusBadge(doc.status)}
      ${doc.status === "failed" ? `<button type="button" class="btn" data-retry="${esc(doc.id)}">${icon("i-refresh")}Retry indexing</button>` : ""}
      <button type="button" class="btn btn-danger" data-delete="${esc(doc.id)}">${icon("i-trash")}Delete document</button>
    </div>
    <section aria-labelledby="ing-title"><h3 id="ing-title">Ingestion pipeline</h3>
      ${rows.length ? `<p class="muted small" style="margin:4px 0 10px">Total ${fmtMs(t.total_ms)} · LangGraph: parse → split → embed → store</p>${waterfallHTML(rows, "Ingestion stage timings")}` : `<p class="muted small">No timings yet (indexed before tracing was added, or still running).</p>`}
      ${ocrPages ? `<h3 style="margin-top:14px">OCR per page</h3>${ocrPages}` : ""}
    </section>
    <section aria-labelledby="chunks-title"><h3 id="chunks-title">Chunks (${chunks.length})</h3>
      <div class="chunks" style="margin-top:8px">${chunks.map((c) => `<details class="chunk"><summary><span class="chip">#${c.chunk_index}</span>${c.page ? `<span class="chip">p.${c.page}</span>` : ""}
        <span class="s-sec">${esc(c.section_path || "")}</span><span class="muted">${c.token_count} tok</span></summary><pre>${esc(c.content)}</pre></details>`).join("")}</div>
    </section>`;
}

/* --------------------------------------------------------------------------- assistant */

/** AI assistant page: sessions, knowledge access, messages. */
async function loadAssistant() {
  const [, sessions] = await Promise.all([refreshDocs(), api("/api/v1/chat/sessions")]);
  state.sessions = sessions;
  if (!state.sessionId || !sessions.some((s) => s.id === state.sessionId)) state.sessionId = sessions[0]?.id || null;
  renderFolderSelect();
  renderSessions();
  await renderConversation();
}

/** Folder choices for a new folder chat. */
function renderFolderSelect() {
  const folders = [...new Set(state.docs.filter((d) => d.status === "ready").map((d) => d.folder_id).filter(Boolean))].sort();
  $("#new-chat-folder").innerHTML = folders.map((f) => `<option>${esc(f)}</option>`).join("") || `<option value="">No folders yet</option>`;
  $("#new-chat-folder").hidden = $('input[name="chat-scope"]:checked').value !== "folder";
}

/** Session list (filtered by the search box). */
function renderSessions() {
  const term = $("#session-filter").value.toLowerCase();
  $("#session-list").innerHTML = state.sessions.filter((s) => !term || (s.title || "").toLowerCase().includes(term)).map((s) => `<li>
    <button type="button" class="session-btn" data-session="${s.id}" ${s.id === state.sessionId ? 'aria-current="true"' : ""}>
      <span class="s-title">${esc(s.title || "New chat")}</span>
      <span class="s-meta">${s.scope === "folder" ? `<span class="chip">${icon("i-folder", "icon icon-xs")}${esc(s.folder_id)}</span>` : `<span class="chip">General</span>`}${relTime(s.last_message_at || s.created_at)}</span>
    </button></li>`).join("") || `<li class="empty">No chats yet.</li>`;
}

/** Documents the current session may use (default: all ready docs, or the folder's docs). */
function allowedFor(session) {
  if (!state.allowed.has(session.id)) {
    const docs = state.docs.filter((d) => d.status === "ready" && (session.scope !== "folder" || d.folder_id === session.folder_id));
    state.allowed.set(session.id, new Set(docs.map((d) => d.id)));
  }
  return state.allowed.get(session.id);
}

/** Knowledge-access checklist for the current session. */
function renderAllowList() {
  const session = state.sessions.find((s) => s.id === state.sessionId);
  if (!session) {
    $("#allow-list").innerHTML = "";
    $("#allow-count").textContent = "";
    return;
  }
  const allowed = allowedFor(session);
  const docs = state.docs.filter((d) => d.status === "ready" && (session.scope !== "folder" || d.folder_id === session.folder_id));
  $("#allow-list").innerHTML = docs.map((d) => `<li><label><input type="checkbox" data-allow="${esc(d.id)}" ${allowed.has(d.id) ? "checked" : ""}>
      <span class="a-name">${esc(d.filename)}</span><span class="a-meta">${esc(d.folder_id || "—")} · ${d.chunk_count} chunks</span></label></li>`).join("") || `<li class="empty">No indexed documents${session.scope === "folder" ? " in this folder" : ""}.</li>`;
  $("#allow-count").textContent = `${[...allowed].filter((id) => docs.some((d) => d.id === id)).length} of ${docs.length} allowed`;
}

/** Header, messages and composer for the current session. */
async function renderConversation() {
  const session = state.sessions.find((s) => s.id === state.sessionId);
  $("#rename-chat").hidden = $("#delete-chat").hidden = !session;
  $("#chat-scope-chip").hidden = !session;
  $("#question").disabled = $("#send").disabled = !session || state.streaming;
  renderAllowList();
  if (!session) {
    $("#chat-title").textContent = "Start a conversation";
    $("#messages").innerHTML = welcomeHTML(false);
    return;
  }
  $("#chat-title").textContent = session.title || "New chat";
  $("#chat-scope-chip").textContent = session.scope === "folder" ? `Folder · ${session.folder_id}` : "General";
  const messages = await api(`/api/v1/chat/sessions/${session.id}/messages`);
  $("#messages").innerHTML = messages.length ? messages.map(messageHTML).join("") : welcomeHTML(true);
  const lastBot = [...messages].reverse().find((m) => m.role === "assistant" && m.trace?.sources);
  renderLastRetrieval(lastBot ? { search_query: lastBot.trace.search_query, sources: lastBot.trace.sources } : null);
  scrollMessages();
}

/** Empty-state with example questions. */
function welcomeHTML(hasSession) {
  const examples = ["Which OCR engines does the system use?", "How much is the meal allowance?", "Summarise the attendance rules", "What file types can be uploaded?"];
  return `<div class="welcome"><span class="tile">${icon("i-sparkles")}</span><h3>${hasSession ? "Ask about your HRD documents" : "Create a chat to begin"}</h3>
    <p>Answers come only from the documents allowed on the right, with numbered citations and a timing breakdown of every pipeline stage.</p>
    ${hasSession ? `<div class="suggestions">${examples.map((q) => `<button type="button" class="suggestion" data-suggest="${esc(q)}">${esc(q)}</button>`).join("")}</div>` : ""}</div>`;
}

/** One chat message (user or assistant). */
function messageHTML(m) {
  if (m.role === "user") {
    return `<div class="msg msg-user"><span class="msg-avatar">${esc(initials(state.user.id))}</span><div class="msg-body"><div class="bubble">${esc(m.content).replace(/\n/g, "<br>")}</div></div></div>`;
  }
  return `<div class="msg msg-bot" data-message="${m.id}"><span class="msg-avatar">AI</span><div class="msg-body">
    <div class="bubble">${renderMarkdown(m.content)}</div>${sourcesHTML(m.citations)}${timingHTML(m.trace)}</div></div>`;
}

/** Citation chips under an answer. */
function sourcesHTML(citations) {
  if (!citations?.length) return "";
  return `<div class="sources">${citations.map((c) => `<span class="source-chip" data-n="${c.n}" title="${esc(c.snippet || "")}"><b>[${c.n}]</b><span>${esc(c.filename)}${c.page ? ` · p.${c.page}` : ""}${c.section ? ` · ${esc(c.section)}` : ""}</span></span>`).join("")}</div>`;
}

/** Timing summary + expandable waterfall for one answer. */
function timingHTML(trace) {
  if (!trace || trace.total_ms == null) return "";
  const rows = stageRows(CHAT_STAGES, trace, trace.total_ms);
  const retrieval = (trace.embed_ms || 0) + (trace.dense_ms || 0) + (trace.lexical_ms || 0) + (trace.fuse_ms || 0);
  const id = `wf-${Math.random().toString(36).slice(2, 9)}`;
  return `<div class="timing">
    <div class="timing-summary">${icon("i-clock")}<span>Total <strong>${fmtMs(trace.total_ms)}</strong></span>
      <span>First token <strong>${fmtMs(trace.first_token_ms)}</strong></span><span>Retrieval <strong>${fmtMs(retrieval)}</strong></span>
      <span>Rerank <strong>${fmtMs(trace.rerank_ms)}</strong></span>${trace.tokens_per_s ? `<span><strong>${trace.tokens_per_s}</strong> tok/s</span>` : ""}
      ${trace.llm_skipped ? `<span class="chip">LLM skipped: ${esc(trace.llm_skipped)}</span>` : ""}
      <button type="button" class="link-btn toggle" aria-expanded="false" aria-controls="${id}">Pipeline details</button></div>
    ${stripHTML(rows)}
    <div id="${id}" hidden>${waterfallHTML(rows, "Answer pipeline stage timings")}
      <p class="muted small" style="margin-top:6px">Model ${esc(trace.model || "")}${trace.search_query ? ` · search query “${esc(trace.search_query)}”` : ""}</p></div>
  </div>`;
}

/** Minimal, safe Markdown for answers: paragraphs, lists, bold, code, tables and [n] citations. */
function renderMarkdown(text) {
  const inline = (s) => s
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[(\d{1,2})\]/g, '<button type="button" class="cite" data-cite="$1" aria-label="Source $1">$1</button>');
  const lines = esc(text || "").split("\n");
  const out = [];
  let list = null;
  let para = [];
  let table = [];
  const flush = () => {
    if (para.length) out.push(`<p>${inline(para.join("<br>"))}</p>`);
    para = [];
    if (list) out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`);
    list = null;
    if (table.length) {
      const [head, ...body] = table;
      out.push(`<table><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${body.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
    }
    table = [];
  };
  for (const raw of lines) {
    const line = raw.trim();
    let m;
    if (!line) { flush(); continue; }
    if (/^\|.*\|$/.test(line)) {
      if (/^\|[\s:|-]+\|$/.test(line)) continue;
      if (para.length || list) flush();
      table.push(line.slice(1, -1).split("|").map((c) => c.trim()));
      continue;
    }
    if ((m = line.match(/^[-*•]\s+(.*)$/)) || (m = line.match(/^\d+[.)]\s+(.*)$/))) {
      const tag = /^\d/.test(line) ? "ol" : "ul";
      if (para.length || table.length || (list && list.tag !== tag)) flush();
      list = list || { tag, items: [] };
      list.items.push(m[1]);
      continue;
    }
    if ((m = line.match(/^#{1,4}\s+(.*)$/))) { flush(); out.push(`<p><strong>${inline(m[1])}</strong></p>`); continue; }
    if (list || table.length) flush();
    para.push(line);
  }
  flush();
  return out.join("");
}

/** Right panel: the search query and ranked sources of the latest answer. */
function renderLastRetrieval(data) {
  const box = $("#last-retrieval");
  if (!data?.sources) {
    box.innerHTML = "Ask a question to see the ranked sources.";
    return;
  }
  const best = Math.max(...data.sources.map((s) => s.rerank_score ?? s.score ?? 0), 0.0001);
  box.innerHTML = `${data.search_query ? `<div class="ret-query">${icon("i-search", "icon icon-xs")} ${esc(data.search_query)}</div>` : ""}
    ${data.sources.length ? `<ol class="ret-list" role="list">${data.sources.map((s) => {
      const score = s.rerank_score ?? s.score ?? 0;
      return `<li><div class="ret-top"><b>[${s.n}]</b><span>${esc(s.filename)}${s.page ? ` · p.${s.page}` : ""}</span><em>${score.toFixed(3)}</em></div>
        <div class="score-track" aria-hidden="true"><div class="score-fill" style="width:${(score / best) * 100}%"></div></div>
        <div class="ret-meta">${s.rerank_score != null ? "rerank score" : "fused score"} · vector #${s.dense_rank ?? "—"} · keyword #${s.lexical_rank ?? "—"}</div></li>`;
    }).join("")}</ol>` : `<p>No passage passed the relevance threshold.</p>`}`;
}

/** Keep the newest message in view. */
function scrollMessages() {
  const box = $("#messages");
  box.scrollTop = box.scrollHeight;
}

/** Send a question and stream the answer. */
async function ask(question) {
  const session = state.sessions.find((s) => s.id === state.sessionId);
  if (!session || !question.trim() || state.streaming) return;
  state.streaming = true;
  $("#question").value = "";
  autosize();
  $("#question").disabled = $("#send").disabled = true;
  const box = $("#messages");
  if ($(".welcome", box)) box.innerHTML = "";
  box.insertAdjacentHTML("beforeend", messageHTML({ role: "user", content: question }));
  box.insertAdjacentHTML("beforeend", `<div class="msg msg-bot"><span class="msg-avatar">AI</span><div class="msg-body"><div class="bubble streaming" aria-busy="true"><span class="stream-text"></span></div></div></div>`);
  const msgEl = box.lastElementChild;
  const bubble = $(".bubble", msgEl);
  const textEl = $(".stream-text", bubble);
  scrollMessages();
  let answer = "";
  let citations = [];
  let trace = null;
  let failed = null;
  try {
    const res = await api(`/api/v1/chat/sessions/${session.id}/messages`, {
      method: "POST", raw: true,
      json: { content: question, allowed_document_ids: [...allowedFor(session)], stream: true, debug: $("#debug-toggle").checked },
    });
    if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 200)}`);
    for await (const { event, data } of readSSE(res)) {
      if (event === "token") {
        answer += data.text;
        textEl.textContent = answer;
        scrollMessages();
      } else if (event === "retrieval") {
        renderLastRetrieval(data);
      } else if (event === "citations") {
        citations = data.citations;
      } else if (event === "trace") {
        trace = data.trace;
      } else if (event === "error") {
        failed = data.message;
      }
    }
  } catch (e) {
    failed = e.message;
  }
  bubble.classList.remove("streaming");
  bubble.removeAttribute("aria-busy");
  bubble.innerHTML = failed ? `<div class="alert">${icon("i-alert")}<span>${esc(failed)}</span></div>` : renderMarkdown(answer);
  if (!failed) {
    bubble.insertAdjacentHTML("afterend", sourcesHTML(citations) + timingHTML(trace));
    if (trace && !$("#debug-toggle").checked) renderLastRetrieval({ search_query: trace.search_query, sources: trace.sources });
  }
  state.streaming = false;
  $("#question").disabled = $("#send").disabled = false;
  $("#question").focus();
  scrollMessages();
  state.sessions = await api("/api/v1/chat/sessions").catch(() => state.sessions);
  const current = state.sessions.find((s) => s.id === state.sessionId);
  if (current) $("#chat-title").textContent = current.title || "New chat";
  renderSessions();
}

/** Grow the question box with its content (up to the CSS max-height). */
function autosize() {
  const ta = $("#question");
  ta.style.height = "auto";
  ta.style.height = `${Math.min(ta.scrollHeight, 180)}px`;
}

/* --------------------------------------------------------------------------- pipeline log */

/** Pipeline log page: stage averages and the per-run table. */
async function loadPipeline() {
  const [stats, traces] = await Promise.all([api("/api/v1/stats"), api("/api/v1/traces?limit=100")]);
  state.stats = stats;
  state.traces = traces;
  renderPipeline();
}

/** Render the stage cards and the table for the selected log kind. */
function renderPipeline() {
  const chat = state.logKind === "chat";
  const stages = chat ? state.stats.chat.stages : state.stats.ingestion.stages;
  const cards = chat
    ? [["Rewrite query", "rewrite_ms", "var(--st-rewrite)"], ["Embed query", "embed_ms", "var(--st-embed)"], ["Vector search", "dense_ms", "var(--st-dense)"], ["Keyword search", "lexical_ms", "var(--st-lexical)"],
       ["Rerank", "rerank_ms", "var(--st-rerank)"], ["LLM first token", "llm_ttft_ms", "var(--st-ttft)"], ["Generation", "generate_ms", "var(--st-generate)"], ["Total answer", "total_ms", "var(--primary)"]]
    : [["Queue wait", "queue_ms", "var(--st-queue)"], ["Parse & extract", "extract_ms", "var(--st-extract)"], ["OCR", "ocr_ms", "var(--st-ocr)"], ["Chunk", "chunk_ms", "var(--st-chunk)"],
       ["Embed", "embed_ms", "var(--st-embed)"], ["Store", "store_ms", "var(--st-store)"], ["Total indexing", "total_ms", "var(--primary)"]];
  $("#stage-cards").innerHTML = cards.map(([label, key, color]) => `<div class="card stage-card"><span class="l"><span class="swatch-dot" style="background:${color}"></span>${label}</span>
      <span class="v">${fmtMs(stages[key]?.avg)}</span><span class="s">median ${fmtMs(stages[key]?.p50)} · ${stages[key]?.n ?? 0} runs</span></div>`).join("");

  const table = $("#log-table");
  $("#log-caption").textContent = chat ? "Recent chat answers with per-stage timings" : "Recent indexing runs with per-stage timings";
  if (chat) {
    $("thead", table).innerHTML = `<tr><th scope="col">Question</th><th scope="col" class="num">Rewrite</th><th scope="col" class="num">Embed</th><th scope="col" class="num">Search</th>
      <th scope="col" class="num">Rerank</th><th scope="col" class="num">LLM 1st token</th><th scope="col" class="num">Generate</th><th scope="col" class="num">First token</th><th scope="col" class="num">Total</th><th scope="col" class="num">tok/s</th></tr>`;
    $("tbody", table).innerHTML = state.traces.chat.map((r) => {
      const t = r.trace;
      const row = `<tr data-trace-row="${r.id}" aria-expanded="${state.openTraceId === r.id}">
        <td class="q"><span class="q-text">${esc(r.question || r.title || "—")}</span><span class="q-meta">${esc(r.user_id)} · ${relTime(r.created_at)} · ${(t.sources || []).length} sources · ${r.citations} cited</span></td>
        <td class="num ${msClass(t.rewrite_ms, 10000, 3000)}">${fmtMs(t.rewrite_ms)}</td><td class="num ${msClass(t.embed_ms, 3000, 1000)}">${fmtMs(t.embed_ms)}</td>
        <td class="num ${msClass((t.dense_ms || 0) + (t.lexical_ms || 0), 3000, 1000)}">${fmtMs(t.dense_ms != null ? (t.dense_ms || 0) + (t.lexical_ms || 0) : null)}</td>
        <td class="num ${msClass(t.rerank_ms, 20000, 8000)}">${fmtMs(t.rerank_ms)}</td><td class="num ${msClass(t.llm_ttft_ms, 20000, 8000)}">${fmtMs(t.llm_ttft_ms)}</td>
        <td class="num ${msClass(t.generate_ms, 40000, 15000)}">${fmtMs(t.generate_ms)}</td><td class="num ${msClass(t.first_token_ms, 30000, 12000)}">${fmtMs(t.first_token_ms)}</td>
        <td class="num ${msClass(t.total_ms)}"><strong>${fmtMs(t.total_ms)}</strong></td><td class="num">${t.tokens_per_s ?? "—"}</td></tr>`;
      if (state.openTraceId !== r.id) return row;
      const sources = (t.sources || []).map((s) => `<li>[${s.n}] ${esc(s.filename)}${s.page ? ` · p.${s.page}` : ""} · ${s.rerank_score != null ? `rerank ${Number(s.rerank_score).toFixed(3)}` : `rrf ${s.rrf_score}`}</li>`).join("");
      return `${row}<tr class="detail-row"><td colspan="10"><div class="detail-grid">
        <div><h3>Stage waterfall</h3>${waterfallHTML(stageRows(CHAT_STAGES, t, t.total_ms), "Stage timings for this answer")}</div>
        <div><h3>Details</h3><dl class="kv"><dt>Model</dt><dd>${esc(t.model)}</dd><dt>Search query</dt><dd>${esc(t.search_query || "—")}</dd>
          <dt>Allowed docs</dt><dd>${t.allowed_documents ?? "—"}</dd><dt>Tokens</dt><dd>${fmtNum(t.input_tokens)} in · ${fmtNum(t.output_tokens)} out</dd>
          <dt>Answer</dt><dd>${esc(r.answer)}</dd></dl>${sources ? `<h3 style="margin-top:12px">Sources</h3><ol class="small" style="margin:6px 0 0;padding-left:18px">${sources}</ol>` : ""}</div>
      </div></td></tr>`;
    }).join("") || `<tr><td colspan="10" class="empty">No answers logged yet. Ask the assistant a question.</td></tr>`;
  } else {
    $("thead", table).innerHTML = `<tr><th scope="col">Document</th><th scope="col">OCR</th><th scope="col" class="num">Pages</th><th scope="col" class="num">Queue</th><th scope="col" class="num">Extract</th>
      <th scope="col" class="num">OCR time</th><th scope="col" class="num">Chunk</th><th scope="col" class="num">Embed</th><th scope="col" class="num">Store</th><th scope="col" class="num">Total</th></tr>`;
    $("tbody", table).innerHTML = state.traces.ingestion.map((r) => {
      const t = r.trace;
      const key = `doc:${r.id}`;
      const engines = t.ocr ? Object.keys(t.ocr.engines || {}).map((e) => `<span class="chip ${e === "surya" ? "primary" : ""}">${e === "paddle" ? "PaddleOCR" : e === "surya" ? "Surya" : esc(e)}</span>`).join(" ") : `<span class="muted">—</span>`;
      const row = `<tr data-trace-row="${esc(key)}" aria-expanded="${state.openTraceId === key}">
        <td class="q"><div class="doc-cell">${fileTile(r.extension)}<div class="grow"><span class="q-text">${esc(r.filename)}</span><span class="q-meta">${esc(r.folder_id || "—")} · ${r.chunk_count} chunks · ${relTime(r.indexed_at)}</span></div></div></td>
        <td>${engines}</td><td class="num">${r.page_count ?? "—"}</td><td class="num">${fmtMs(t.queue_ms)}</td><td class="num">${fmtMs(t.extract_ms)}</td>
        <td class="num ${msClass(t.ocr?.ms, 120000, 30000)}">${fmtMs(t.ocr?.ms)}</td><td class="num">${fmtMs(t.chunk_ms)}</td><td class="num ${msClass(t.embed_ms, 60000, 20000)}">${fmtMs(t.embed_ms)}</td>
        <td class="num">${fmtMs(t.store_ms)}</td><td class="num ${msClass(t.total_ms, 120000, 30000)}"><strong>${fmtMs(t.total_ms)}</strong></td></tr>`;
      if (state.openTraceId !== key) return row;
      const pages = t.ocr ? t.ocr.per_page.map((p, i) => `<li>Page ${i + 1}: ${esc(p.engine)} · ${fmtMs(p.ms)} · conf ${p.confidence ?? "—"}${p.fallback_reason ? ` · ${esc(p.fallback_reason)}` : ""}</li>`).join("") : "";
      return `${row}<tr class="detail-row"><td colspan="10"><div class="detail-grid">
        <div><h3>Stage waterfall</h3>${waterfallHTML(stageRows(INGEST_STAGES, t), "Stage timings for this indexing run")}</div>
        <div><h3>Details</h3><dl class="kv"><dt>Chunks</dt><dd>${t.chunks ?? "—"} (${fmtNum(t.tokens)} tokens)</dd><dt>Embedding</dt><dd>${esc(t.embed_model || "—")} · ${t.embed_chunks_per_s ?? "—"} chunks/s</dd>
          <dt>OCR</dt><dd>${t.ocr ? esc(ocrSummary(t.ocr)) : "not needed"}</dd></dl>${pages ? `<ol class="small" style="margin:8px 0 0;padding-left:18px">${pages}</ol>` : ""}</div>
      </div></td></tr>`;
    }).join("") || `<tr><td colspan="10" class="empty">No indexing runs logged yet. Upload a document.</td></tr>`;
  }
}

/* --------------------------------------------------------------------------- models */

/** Model stack page. */
async function loadModels() {
  const [info, stats] = await Promise.all([api("/api/v1/info"), api("/api/v1/stats")]);
  state.info = info;
  await refreshHealth();
  const cs = stats.chat.stages;
  const is = stats.ingestion.stages;
  const card = (ic, tile, title, role, [cls, text], name, rows) => `<article class="card model-card">
      <div class="top"><span class="tile ${tile}">${icon(ic)}</span><div><h2>${title}</h2><p>${role}</p></div><span class="badge ${cls}">${text}</span></div>
      <div class="model-name">${esc(name)}</div>
      <dl class="kv">${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl></article>`;
  $("#model-grid").innerHTML = [
    card("i-sparkles", "violet", "LLM", "Writes the answer from the sources", checkState("llm"), info.llm.model, [
      ["Endpoint", esc(info.llm.base_url)], ["Thinking", esc(info.llm.thinking)], ["Context window", fmtNum(info.llm.context_window)], ["Max answer tokens", fmtNum(info.llm.max_tokens)],
      ["Avg first token", fmtMs(cs.llm_ttft_ms.avg)], ["Avg speed", `${cs.tokens_per_s.avg ?? "—"} tok/s`]]),
    card("i-layers", "green", "Embeddings", "Turns chunks and questions into vectors", checkState("embeddings"), info.embeddings.model, [
      ["Endpoint", esc(info.embeddings.base_url)], ["Dimensions", info.embeddings.dim], ["Avg query embed", fmtMs(cs.embed_ms.avg)], ["Avg document embed", fmtMs(is.embed_ms.avg)]]),
    card("i-gauge", "amber", "Reranker", "Re-scores candidates, keeps the best", checkState("reranker"), info.reranker.model, [
      ["API", esc(info.reranker.api)], ["Candidates", info.reranker.candidates], ["Min score", info.reranker.min_score], ["Avg rerank", fmtMs(cs.rerank_ms.avg)]]),
    card("i-scan", "pink", "OCR", "Reads scans and photos", info.ocr.mode === "auto" && !info.ocr.surya_installed ? ["warn", "Surya missing"] : ["ok", "Ready"],
      info.ocr.mode === "auto" ? "auto: PaddleOCR → Surya (surya-ocr 0.17)" : info.ocr.mode, [
      ["Mode", esc(info.ocr.mode)], ["Surya worker", info.ocr.surya_installed ? "installed (.venv-surya)" : "not installed"], ["Fallback below confidence", info.ocr.fallback_min_confidence], ["Avg OCR per document", fmtMs(is.ocr_ms.avg)]]),
    card("i-database", "blue", "Vector database", "Stores chunks, vectors and chats", checkState("database"), info.vector_db, [
      ["Dense search", `top ${info.retrieval.dense_k} · HNSW`], ["Keyword search", `top ${info.retrieval.lexical_k} · Postgres full-text`], ["Fusion", `RRF k=${info.retrieval.rrf_k}`],
      ["Avg vector search", fmtMs(cs.dense_ms.avg)], ["Avg keyword search", fmtMs(cs.lexical_ms.avg)]]),
    card("i-file", "red", "Chunking", "Structure-aware splitter", ["ok", "Ready"], `${info.chunking.max_tokens} tokens · ${info.chunking.overlap_tokens} overlap`, [
      ["Tokenizer", esc(info.chunking.tokenizer)], ["Sources per answer", info.retrieval.context_top_k], ["Avg chunking time", fmtMs(is.chunk_ms.avg)]]),
  ].join("");
}

/* --------------------------------------------------------------------------- events */

/** Light-dismiss for dialogs in browsers without the closedby attribute. */
function dialogFallback(dialog) {
  if ("closedBy" in HTMLDialogElement.prototype) return;
  dialog.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    const r = dialog.getBoundingClientRect();
    const inside = r.top <= event.clientY && event.clientY <= r.top + r.height && r.left <= event.clientX && event.clientX <= r.left + r.width;
    if (!inside) dialog.close();
  });
}

/** Show the acting user in the top bar. */
function renderUser() {
  const [bg, fg] = avatarColors(state.user.id);
  $("#user-avatar").textContent = initials(state.user.id);
  $("#user-avatar").style.cssText = `background:${bg};color:${fg}`;
  $("#user-name").textContent = state.user.id;
  $("#user-role").textContent = state.user.role === "ADMIN" ? "HRD Admin" : "Instructor";
  for (const el of $$(".js-user-name")) el.textContent = state.user.id;
}

/** Wire up all event handlers. */
function bindEvents() {
  addEventListener("hashchange", () => route());

  // global search -> documents
  $("#global-search").addEventListener("submit", (e) => {
    e.preventDefault();
    state.docFilter = $("#global-search-input").value.trim();
    $("#doc-filter").value = state.docFilter;
    if (location.hash !== "#documents") location.hash = "#documents";
    else renderDocuments();
  });
  addEventListener("keydown", (e) => {
    if (e.key === "/" && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) {
      e.preventDefault();
      $("#global-search-input").focus();
    }
  });

  // acting user
  $("#user-chip").addEventListener("click", () => {
    $("#user-id-input").value = state.user.id;
    $("#user-role-input").value = state.user.role;
    $("#user-dialog").showModal();
  });
  $("#user-dialog").addEventListener("close", () => {
    if ($("#user-dialog").returnValue !== "save") return;
    state.user = { id: $("#user-id-input").value.trim() || "admin-1", role: $("#user-role-input").value };
    saveUser();
    renderUser();
    state.sessionId = null;
    route(false);
  });

  // overview list clicks
  document.addEventListener("click", (e) => {
    const traceBtn = e.target.closest("[data-trace]");
    if (traceBtn) {
      state.openTraceId = Number(traceBtn.dataset.trace);
      state.logKind = "chat";
      $('input[name="log-kind"][value="chat"]').checked = true;
      location.hash = "#pipeline";
      return;
    }
    const docBtn = e.target.closest("button[data-doc], tr[data-doc]");
    if (docBtn && !e.target.closest("a")) openDoc(docBtn.dataset.doc);
  });

  // documents
  const dz = $("#dropzone");
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragging"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragging"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragging");
    if (e.dataTransfer.files.length) uploadFiles([...e.dataTransfer.files]);
  });
  $("#file-input").addEventListener("change", (e) => {
    uploadFiles([...e.target.files]);
    e.target.value = "";
  });
  $("#upload-form").addEventListener("submit", (e) => e.preventDefault());
  $("#doc-filter").addEventListener("input", (e) => { state.docFilter = e.target.value; renderDocuments(); });
  for (const r of $$('input[name="doc-status"]')) r.addEventListener("change", (e) => { state.docStatus = e.target.value; renderDocuments(); });

  // drawer actions
  $("#doc-dialog").addEventListener("click", async (e) => {
    const retry = e.target.closest("[data-retry]");
    const del = e.target.closest("[data-delete]");
    try {
      if (retry) {
        await api(`/api/v1/documents/${encodeURIComponent(retry.dataset.retry)}/retry`, { method: "POST" });
        toast("Queued for indexing again");
        $("#doc-dialog").close();
        await refreshDocs();
        renderDocuments();
      }
      if (del) {
        if (del.dataset.confirm !== "1") {
          del.dataset.confirm = "1";
          del.lastChild.textContent = "Click again to delete";
          return;
        }
        await api(`/api/v1/documents/${encodeURIComponent(del.dataset.delete)}`, { method: "DELETE" });
        toast("Document deleted");
        $("#doc-dialog").close();
        route(false);
      }
    } catch (err) {
      toast(err.message, "bad");
    }
  });

  // assistant
  for (const r of $$('input[name="chat-scope"]')) r.addEventListener("change", renderFolderSelect);
  $("#new-chat").addEventListener("click", async () => {
    const scope = $('input[name="chat-scope"]:checked').value;
    const folder = $("#new-chat-folder").value;
    if (scope === "folder" && !folder) return toast("Index a document in a folder first", "bad");
    try {
      const session = await api("/api/v1/chat/sessions", { method: "POST", json: { scope, folder_id: scope === "folder" ? folder : null } });
      state.sessionId = session.id;
      state.sessions = await api("/api/v1/chat/sessions");
      renderSessions();
      await renderConversation();
      $("#question").focus();
    } catch (err) {
      toast(err.message, "bad");
    }
  });
  $("#session-filter").addEventListener("input", renderSessions);
  $("#session-list").addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-session]");
    if (!btn || state.streaming) return;
    state.sessionId = btn.dataset.session;
    renderSessions();
    await renderConversation().catch((err) => toast(err.message, "bad"));
  });
  $("#rename-chat").addEventListener("click", () => {
    $("#rename-input").value = state.sessions.find((s) => s.id === state.sessionId)?.title || "";
    $("#rename-dialog").showModal();
  });
  $("#rename-dialog").addEventListener("close", async () => {
    if ($("#rename-dialog").returnValue !== "save") return;
    try {
      await api(`/api/v1/chat/sessions/${state.sessionId}`, { method: "PATCH", json: { title: $("#rename-input").value.trim() } });
      state.sessions = await api("/api/v1/chat/sessions");
      renderSessions();
      await renderConversation();
    } catch (err) {
      toast(err.message, "bad");
    }
  });
  $("#delete-chat").addEventListener("click", async () => {
    if (!confirm("Delete this chat and its messages?")) return;
    try {
      await api(`/api/v1/chat/sessions/${state.sessionId}`, { method: "DELETE" });
      state.sessionId = null;
      await loadAssistant();
    } catch (err) {
      toast(err.message, "bad");
    }
  });
  $("#toggle-context").addEventListener("click", (e) => {
    const layout = $("#chat-layout");
    const hidden = layout.classList.toggle("no-context");
    e.currentTarget.setAttribute("aria-expanded", String(!hidden));
  });
  $("#allow-list").addEventListener("change", (e) => {
    const box = e.target.closest("[data-allow]");
    const session = state.sessions.find((s) => s.id === state.sessionId);
    if (!box || !session) return;
    const set = allowedFor(session);
    if (box.checked) set.add(box.dataset.allow);
    else set.delete(box.dataset.allow);
    renderAllowList();
  });
  $("#allow-all").addEventListener("click", () => {
    const session = state.sessions.find((s) => s.id === state.sessionId);
    if (!session) return;
    state.allowed.delete(session.id);
    renderAllowList();
  });
  $("#allow-none").addEventListener("click", () => {
    if (!state.sessionId) return;
    state.allowed.set(state.sessionId, new Set());
    renderAllowList();
  });
  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    ask($("#question").value);
  });
  $("#question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      ask($("#question").value);
    }
  });
  $("#question").addEventListener("input", autosize);
  $("#messages").addEventListener("click", (e) => {
    const suggestion = e.target.closest("[data-suggest]");
    if (suggestion) return ask(suggestion.dataset.suggest);
    const toggle = e.target.closest(".timing .toggle");
    if (toggle) {
      const panel = document.getElementById(toggle.getAttribute("aria-controls"));
      const open = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", String(!open));
      panel.hidden = open;
      return;
    }
    const cite = e.target.closest(".cite");
    if (cite) {
      const chip = $(`.source-chip[data-n="${cite.dataset.cite}"]`, cite.closest(".msg-body"));
      if (chip) {
        chip.classList.add("flash");
        chip.scrollIntoView({ block: "nearest" });
        setTimeout(() => chip.classList.remove("flash"), 1400);
      }
    }
  });

  // pipeline log
  for (const r of $$('input[name="log-kind"]')) r.addEventListener("change", (e) => { state.logKind = e.target.value; state.openTraceId = null; renderPipeline(); });
  $("#refresh-log").addEventListener("click", () => loadPipeline().catch((err) => toast(err.message, "bad")));
  $("#log-table").addEventListener("click", (e) => {
    const row = e.target.closest("[data-trace-row]");
    if (!row) return;
    const key = state.logKind === "chat" ? Number(row.dataset.traceRow) : row.dataset.traceRow;
    state.openTraceId = state.openTraceId === key ? null : key;
    renderPipeline();
  });

  for (const d of $$("dialog")) dialogFallback(d);
}

/* --------------------------------------------------------------------------- start */

/** Boot the console. */
async function init() {
  try { sessionStorage.removeItem("console-reloaded"); } catch (e) { /* ignore */ }
  setupThemeToggle();
  renderUser();
  bindEvents();
  await refreshHealth();
  setInterval(refreshHealth, 20000);
  let hasSavedUser = false;
  try { hasSavedUser = Boolean(localStorage.getItem("console-user")); } catch (e) { /* storage blocked */ }
  if (!hasSavedUser) {
    try {
      const info = await api("/api/v1/info");
      state.info = info;
      state.user.id = info.console_user_id || state.user.id;
      renderUser();
    } catch (e) { /* keep default */ }
  }
  route(false);
}

init();
