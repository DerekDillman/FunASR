/* Local Recorder frontend: mic capture -> websocket -> live transcript,
   plus browsing, playback and summarization of saved sessions. */

const $ = (id) => document.getElementById(id);

const state = {
  ready: false,
  recording: false,
  ws: null,
  audioCtx: null,
  mediaStream: null,
  workletNode: null,
  startTime: 0,
  timerHandle: null,
  currentSessionId: null,
  segments: [],
};

// ---------------------------------------------------------------- status

async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    const st = await res.json();
    const asrChip = $("asr-status");
    if (st.asr.loaded) {
      asrChip.textContent = `ASR: ready (${st.asr.engine === "mock" ? "mock" : modelShort(st.asr.offline_model)})`;
      asrChip.className = "chip ok";
      state.ready = true;
      $("record-btn").disabled = state.recording ? true : false;
    } else if (st.asr.load_error) {
      asrChip.textContent = "ASR: failed to load";
      asrChip.className = "chip err";
      asrChip.title = st.asr.load_error;
    } else {
      asrChip.textContent = "ASR: loading models…";
      asrChip.className = "chip warn";
      setTimeout(pollStatus, 2500);
    }
    const llmChip = $("llm-status");
    const s = st.summarizer;
    if (s.available && s.model_pulled) {
      llmChip.textContent = `Summarizer: ${s.model}`;
      llmChip.className = "chip ok";
    } else if (s.available) {
      llmChip.textContent = `Summarizer: pull ${s.model}`;
      llmChip.className = "chip warn";
      llmChip.title = `Ollama is running but the model isn't pulled. Run: ollama pull ${s.model}`;
    } else {
      llmChip.textContent = "Summarizer: fallback (no Ollama)";
      llmChip.className = "chip warn";
      llmChip.title = s.hint || "Install Ollama for LLM summaries; extractive fallback is used otherwise.";
    }
  } catch (e) {
    setTimeout(pollStatus, 3000);
  }
}

function modelShort(name) {
  if (!name) return "?";
  const base = name.split("/").pop();
  return base.length > 22 ? base.slice(0, 22) + "…" : base;
}

// ---------------------------------------------------------------- recording

async function startRecording() {
  if (!state.ready || state.recording) return;
  hideError();
  state.segments = [];
  $("live-transcript").innerHTML = "";
  showRecordPanel();

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (e) {
    showError("Microphone access denied or unavailable: " + e.message);
    return;
  }

  const wsProto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${wsProto}://${location.host}/ws/transcribe`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => ws.send(JSON.stringify({ type: "start", name: $("rec-name").value.trim() }));

  ws.onmessage = async (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "ready") {
      state.currentSessionId = msg.session_id;
      await startMic(stream, ws);
    } else if (msg.type === "partial") {
      renderPartial(msg.text);
    } else if (msg.type === "segment") {
      state.segments.push(msg);
      renderSegment(msg);
    } else if (msg.type === "stopped") {
      ws.close();
      await refreshSessions();
      if (msg.session_id) openSession(msg.session_id);
    } else if (msg.type === "error") {
      showError(msg.message);
      stopRecording(true);
    }
  };

  ws.onerror = () => {
    showError("Connection to the local server failed.");
    stopRecording(true);
  };
  ws.onclose = () => {
    if (state.recording) stopRecording(true);
  };

  state.ws = ws;
  state.mediaStream = stream;
}

async function startMic(stream, ws) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  await ctx.audioWorklet.addModule("pcm-worklet.js");
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "pcm-capture");
  node.port.onmessage = (e) => {
    if (ws.readyState === WebSocket.OPEN) ws.send(e.data);
    drawLevel(new Int16Array(e.data));
  };
  source.connect(node);
  // Keep the graph alive without echoing the mic to the speakers.
  const sink = ctx.createGain();
  sink.gain.value = 0;
  node.connect(sink).connect(ctx.destination);

  state.audioCtx = ctx;
  state.workletNode = node;
  state.recording = true;
  state.startTime = Date.now();
  state.timerHandle = setInterval(updateTimer, 250);

  const btn = $("record-btn");
  btn.textContent = "■ Stop";
  btn.classList.add("recording");
  btn.disabled = false;
}

function stopRecording(abort = false) {
  const wasRecording = state.recording;
  state.recording = false;
  clearInterval(state.timerHandle);

  if (state.mediaStream) state.mediaStream.getTracks().forEach((t) => t.stop());
  if (state.workletNode) state.workletNode.port.onmessage = null;
  if (state.audioCtx) state.audioCtx.close().catch(() => {});
  state.mediaStream = state.audioCtx = state.workletNode = null;

  const ws = state.ws;
  if (ws && ws.readyState === WebSocket.OPEN) {
    if (abort) ws.close();
    else ws.send(JSON.stringify({ type: "stop" })); // wait for "stopped"
  }
  if (abort) state.ws = null;

  const btn = $("record-btn");
  btn.textContent = "● Record";
  btn.classList.remove("recording");
  btn.disabled = !state.ready;
  drawLevel(null);
  if (wasRecording && abort) refreshSessions();
}

function updateTimer() {
  const s = Math.floor((Date.now() - state.startTime) / 1000);
  $("rec-timer").textContent =
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function drawLevel(samples) {
  const canvas = $("level-meter");
  const ctx2d = canvas.getContext("2d");
  ctx2d.clearRect(0, 0, canvas.width, canvas.height);
  if (!samples) return;
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  const rms = Math.sqrt(sum / samples.length) / 32768;
  const level = Math.min(1, rms * 6);
  ctx2d.fillStyle = level > 0.85 ? "#e5534b" : "#3fb950";
  ctx2d.fillRect(2, 4, (canvas.width - 4) * level, canvas.height - 8);
}

// ---------------------------------------------------------------- live transcript

function fmtMs(ms) {
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

function renderSegment(seg) {
  const box = $("live-transcript");
  removePartial(box);
  const div = document.createElement("div");
  div.className = "seg";
  div.innerHTML = `<span class="ts">${fmtMs(seg.start_ms)}</span>`;
  div.appendChild(document.createTextNode(seg.text));
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function renderPartial(text) {
  const box = $("live-transcript");
  let p = box.querySelector(".partial");
  if (!text) { if (p) p.remove(); return; }
  if (!p) {
    p = document.createElement("div");
    p.className = "partial";
    box.appendChild(p);
  }
  p.textContent = text + " …";
  box.scrollTop = box.scrollHeight;
}

function removePartial(box) {
  const p = box.querySelector(".partial");
  if (p) p.remove();
}

function showError(msg) {
  const el = $("rec-error");
  el.textContent = msg;
  el.classList.remove("hidden");
}
function hideError() { $("rec-error").classList.add("hidden"); }

// ---------------------------------------------------------------- sessions

async function refreshSessions() {
  const list = await (await fetch("/api/sessions")).json();
  const ul = $("session-list");
  ul.innerHTML = "";
  if (!list.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No recordings yet.";
    ul.appendChild(li);
    return;
  }
  for (const s of list) {
    const li = document.createElement("li");
    li.dataset.id = s.id;
    if (s.id === state.currentSessionId) li.classList.add("active");
    const date = new Date(s.created_at * 1000).toLocaleString();
    li.innerHTML =
      `<div class="s-name"></div>
       <div class="s-meta">${date} · ${fmtMs(s.duration_ms)}${s.has_summary ? " · ✨" : ""}</div>`;
    li.querySelector(".s-name").textContent = s.name;
    li.onclick = () => openSession(s.id);
    ul.appendChild(li);
  }
}

async function openSession(id) {
  if (state.recording) return;
  const res = await fetch(`/api/sessions/${id}`);
  if (!res.ok) return;
  const meta = await res.json();
  state.currentSessionId = id;

  $("record-panel").classList.add("hidden");
  $("session-panel").classList.remove("hidden");
  $("session-name").value = meta.name;
  $("session-meta").textContent =
    `${new Date(meta.created_at * 1000).toLocaleString()} · ${fmtMs(meta.duration_ms)} · ${meta.segments.length} segments`;
  $("session-audio").src = `/api/sessions/${id}/audio`;
  $("download-link").href = `/api/sessions/${id}/transcript.md`;

  const box = $("session-transcript");
  box.innerHTML = "";
  if (!meta.segments.length) {
    box.innerHTML = '<div class="placeholder">No speech was transcribed in this recording.</div>';
  }
  for (const seg of meta.segments) {
    const div = document.createElement("div");
    div.className = "seg";
    div.innerHTML = `<span class="ts">${fmtMs(seg.start_ms)}</span>`;
    div.appendChild(document.createTextNode(seg.text));
    box.appendChild(div);
  }

  const styles = Object.keys(meta.summaries || {});
  if (styles.length) {
    const style = $("summary-style").value;
    showSummary(meta.summaries[styles.includes(style) ? style : styles[0]]);
  } else {
    $("summary-view").classList.add("hidden");
  }
  refreshSessions();
}

function showRecordPanel() {
  $("session-panel").classList.add("hidden");
  $("record-panel").classList.remove("hidden");
}

async function summarize() {
  const id = state.currentSessionId;
  if (!id) return;
  const btn = $("summarize-btn");
  btn.disabled = true;
  $("summary-spinner").classList.remove("hidden");
  try {
    const res = await fetch(`/api/sessions/${id}/summarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ style: $("summary-style").value }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "summarization failed");
    showSummary(data);
    refreshSessions();
  } catch (e) {
    alert("Summarization failed: " + e.message);
  } finally {
    btn.disabled = false;
    $("summary-spinner").classList.add("hidden");
  }
}

function showSummary(summary) {
  const view = $("summary-view");
  view.innerHTML = renderMarkdown(summary.text) +
    `<div class="summary-meta">style: ${escapeHtml(summary.style)} · via ${escapeHtml(summary.method)} · ${summary.elapsed_s}s</div>`;
  view.classList.remove("hidden");
}

// Minimal markdown renderer (headings, bullets, numbered lists, bold, quotes).
function renderMarkdown(md) {
  const lines = md.split("\n");
  let html = "", inUl = false, inOl = false;
  const close = () => {
    if (inUl) { html += "</ul>"; inUl = false; }
    if (inOl) { html += "</ol>"; inOl = false; }
  };
  for (const raw of lines) {
    const line = raw.trimEnd();
    const inline = (s) => escapeHtml(s)
      .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
      .replace(/`(.+?)`/g, "<code>$1</code>");
    if (/^#{1,6}\s/.test(line)) {
      close();
      html += `<h2>${inline(line.replace(/^#{1,6}\s*/, ""))}</h2>`;
    } else if (/^\s*[-*]\s+/.test(line)) {
      if (!inUl) { close(); html += "<ul>"; inUl = true; }
      html += `<li>${inline(line.replace(/^\s*[-*]\s+/, ""))}</li>`;
    } else if (/^\s*\d+[.)]\s+/.test(line)) {
      if (!inOl) { close(); html += "<ol>"; inOl = true; }
      html += `<li>${inline(line.replace(/^\s*\d+[.)]\s+/, ""))}</li>`;
    } else if (/^>\s?/.test(line)) {
      close();
      html += `<blockquote>${inline(line.replace(/^>\s?/, ""))}</blockquote>`;
    } else if (line.trim() === "") {
      close();
    } else {
      close();
      html += `<p>${inline(line)}</p>`;
    }
  }
  close();
  return html;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------------------------------------------------------------- wiring

$("record-btn").onclick = () => (state.recording ? stopRecording() : startRecording());
$("refresh-btn").onclick = refreshSessions;
$("back-btn").onclick = () => { state.currentSessionId = null; showRecordPanel(); refreshSessions(); };
$("summarize-btn").onclick = summarize;
$("delete-btn").onclick = async () => {
  const id = state.currentSessionId;
  if (!id || !confirm("Delete this recording and its transcript?")) return;
  await fetch(`/api/sessions/${id}`, { method: "DELETE" });
  state.currentSessionId = null;
  showRecordPanel();
  refreshSessions();
};
$("session-name").onchange = async (e) => {
  const id = state.currentSessionId;
  if (!id) return;
  await fetch(`/api/sessions/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: e.target.value }),
  });
  refreshSessions();
};
window.addEventListener("beforeunload", () => { if (state.recording) stopRecording(); });

pollStatus();
refreshSessions();
