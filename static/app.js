"use strict";

const API_BASE = "/api/v1/context";
const MAX_FILE_SIZE_MB = 20;

// ---------- Elements ----------
const uploadView = document.getElementById("upload-view");
const chatView = document.getElementById("chat-view");
const dropzone = document.getElementById("dropzone");
const dropzoneInner = dropzone.querySelector(".dropzone-inner");
const fileInput = document.getElementById("file-input");
const uploadProgress = document.getElementById("upload-progress");
const uploadStatus = document.getElementById("upload-status");
const uploadError = document.getElementById("upload-error");
const sessionBadge = document.getElementById("session-badge");
const docName = document.getElementById("doc-name");
const docChunks = document.getElementById("doc-chunks");
const clearBtn = document.getElementById("clear-btn");
const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");

// ---------- State ----------
let sessionId = null;
let fileName = null;
let busy = false;

// Restore session across page reloads
const saved = sessionStorage.getItem("contexto");
if (saved) {
  try {
    const s = JSON.parse(saved);
    if (s.sessionId) enterChat(s.sessionId, s.fileName, s.totalChunks, true);
  } catch { sessionStorage.removeItem("contexto"); }
}

// ---------- Upload ----------
dropzone.addEventListener("click", () => !busy && fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if ((e.key === "Enter" || e.key === " ") && !busy) {
    e.preventDefault();
    fileInput.click();
  }
});

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    if (!busy) dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  if (busy) return;
  const file = e.dataTransfer.files?.[0];
  if (file) uploadFile(file);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) uploadFile(file);
  fileInput.value = "";
});

async function uploadFile(file) {
  hideError();

  if (!file.name.toLowerCase().endsWith(".pdf")) {
    return showError("Only PDF files are accepted.");
  }
  if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
    return showError(`File exceeds the ${MAX_FILE_SIZE_MB} MB size limit.`);
  }
  if (file.size === 0) {
    return showError("The selected file is empty.");
  }

  busy = true;
  dropzoneInner.classList.add("hidden");
  uploadProgress.classList.remove("hidden");
  uploadStatus.textContent = `Uploading & indexing “${file.name}”…`;

  try {
    const form = new FormData();
    form.append("file", file);

    const res = await fetch(`${API_BASE}/ingest`, { method: "POST", body: form });
    const data = await parseJson(res);
    if (!res.ok) throw new Error(data?.detail || `Upload failed (${res.status})`);

    enterChat(data.session_id, file.name, data.total_chunks, false);
  } catch (err) {
    showError(err.message || "Upload failed. Is the server running?");
  } finally {
    busy = false;
    dropzoneInner.classList.remove("hidden");
    uploadProgress.classList.add("hidden");
  }
}

// ---------- Chat ----------
function enterChat(id, name, totalChunks, restored) {
  sessionId = id;
  fileName = name || "document.pdf";
  sessionStorage.setItem(
    "contexto",
    JSON.stringify({ sessionId: id, fileName: fileName, totalChunks })
  );

  docName.textContent = fileName;
  docChunks.textContent = totalChunks ? `${totalChunks} chunks indexed` : "";
  sessionBadge.classList.remove("hidden");
  uploadView.classList.add("hidden");
  chatView.classList.remove("hidden");

  if (!restored) {
    addSystemMessage(`“${fileName}” is ready — ask away!`);
  } else {
    addSystemMessage(`Resumed session for “${fileName}”.`);
  }
  chatInput.focus();
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  sendQuery();
});

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
});

chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + "px";
});

async function sendQuery() {
  const query = chatInput.value.trim();
  if (!query || busy || !sessionId) return;

  busy = true;
  sendBtn.disabled = true;
  chatInput.value = "";
  chatInput.style.height = "auto";

  addMessage("user", query);
  const typing = addTyping();
  let bubble = null;

  try {
    const res = await fetch(`${API_BASE}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, query }),
    });

    if (res.status === 404) {
      // Session expired / server restarted
      typing.remove();
      addMessage("bot error", "This session no longer exists. Please upload your document again.");
      setTimeout(resetToUpload, 1800);
      return;
    }
    if (!res.ok) {
      const data = await parseJson(res);
      throw new Error(data?.detail || `Request failed (${res.status})`);
    }

    let streamError = null;

    for await (const [event, data] of readEvents(res)) {
      if (event === "sources") {
        typing.remove();
        bubble = startAnswer(data.sources || []);
      } else if (event === "token") {
        if (!bubble) bubble = startAnswer([]);
        bubble.append(data.t);
      } else if (event === "error") {
        streamError = data.detail;
        break;
      }
    }

    if (streamError) throw new Error(streamError);
    if (bubble) bubble.finish();
  } catch (err) {
    typing.remove();
    if (bubble) bubble.finish();
    addMessage("bot error", err.message || "Something went wrong. Please try again.");
  } finally {
    busy = false;
    sendBtn.disabled = false;
    chatInput.focus();
  }
}

/** Parse a `text/event-stream` body into [eventName, data] pairs. */
async function* readEvents(res) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    let split;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);

      let name = "message";
      let payload = null;
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) name = line.slice(7);
        else if (line.startsWith("data: ")) payload = line.slice(6);
      }
      if (payload === null) continue;
      try {
        yield [name, JSON.parse(payload)];
      } catch {
        /* ignore malformed frame */
      }
    }
  }
}

// ---------- Clear session ----------
clearBtn.addEventListener("click", async () => {
  if (!sessionId) return;
  if (!confirm("Remove this document and start over?")) return;

  try {
    await fetch(`${API_BASE}/clear`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
  } catch {
    // Best-effort: reset the UI even if the server call fails
  }
  resetToUpload();
});

function resetToUpload() {
  sessionId = null;
  fileName = null;
  sessionStorage.removeItem("contexto");
  messagesEl.innerHTML = "";
  sessionBadge.classList.add("hidden");
  chatView.classList.add("hidden");
  uploadView.classList.remove("hidden");
  hideError();
}

// ---------- DOM helpers ----------
function addMessage(cls, text) {
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  scrollToBottom();
  return div;
}

function addSystemMessage(text) {
  return addMessage("system", text);
}

/**
 * Create an answer bubble that fills in as tokens arrive.
 * Citations render immediately — they are known before generation starts.
 */
function startAnswer(sources) {
  const div = document.createElement("div");
  div.className = "msg bot streaming";

  const body = document.createElement("span");
  body.className = "answer-body";
  div.appendChild(body);

  if (sources.length) {
    div.appendChild(buildSources(sources));
  }

  messagesEl.appendChild(div);
  scrollToBottom();

  return {
    append(text) {
      body.textContent += text;
      scrollToBottom();
    },
    finish() {
      div.classList.remove("streaming");
      if (!body.textContent) body.textContent = "(no answer returned)";
    },
  };
}

function buildSources(sources) {
  const details = document.createElement("details");
  details.className = "sources";
  const summary = document.createElement("summary");
  summary.textContent = `📎 ${sources.length} source${sources.length > 1 ? "s" : ""}`;
  details.appendChild(summary);

  for (const src of sources) {
    const item = document.createElement("div");
    item.className = "source-item";
    const page = document.createElement("span");
    page.className = "source-page";
    page.textContent = `Page ${src.page}`;
    const excerpt = document.createElement("div");
    excerpt.className = "source-excerpt";
    excerpt.textContent = src.excerpt;
    item.append(page, excerpt);
    details.appendChild(item);
  }
  return details;
}

function addTyping() {
  const div = document.createElement("div");
  div.className = "msg bot";
  div.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  messagesEl.appendChild(div);
  scrollToBottom();
  return div;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showError(msg) {
  uploadError.textContent = msg;
  uploadError.classList.remove("hidden");
}

function hideError() {
  uploadError.classList.add("hidden");
}

async function parseJson(res) {
  try {
    return await res.json();
  } catch {
    return null;
  }
}
