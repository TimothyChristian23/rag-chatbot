const state = {
  sessionId: getStoredSessionId(),
  busy: false,
};

const messagesEl = document.querySelector("#messages");
const sourcesEl = document.querySelector("#sources");
const chatForm = document.querySelector("#chatForm");
const questionInput = document.querySelector("#questionInput");
const sendButton = document.querySelector("#sendButton");
const sessionInput = document.querySelector("#sessionInput");
const newSessionButton = document.querySelector("#newSessionButton");
const clearChatButton = document.querySelector("#clearChatButton");
const uploadForm = document.querySelector("#uploadForm");
const documentInput = document.querySelector("#documentInput");
const uploadButton = document.querySelector("#uploadButton");
const uploadStatus = document.querySelector("#uploadStatus");
const apiStatus = document.querySelector("#apiStatus");
const sourceSnippetsEl = document.querySelector("#sourceSnippets");

sessionInput.value = state.sessionId;

chatForm.addEventListener("submit", handleChatSubmit);
uploadForm.addEventListener("submit", handleUpload);
newSessionButton.addEventListener("click", createNewSession);
clearChatButton.addEventListener("click", clearCurrentSession);
sessionInput.addEventListener("change", switchSession);

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

checkHealth();
loadHistory();

async function handleChatSubmit(event) {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || state.busy) return;

  setBusy(true);
  questionInput.value = "";
  appendMessage("user", question);
  const pendingId = appendMessage("assistant", "Thinking...");

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        question,
      }),
    });
    const payload = await parseJsonResponse(response);
    renderHistory(payload.history);
    renderSources(payload.sources);
    renderSourceSnippets(payload.source_snippets);
  } catch (error) {
    removeMessage(pendingId);
    appendMessage("assistant", error.message);
  } finally {
    setBusy(false);
    questionInput.focus();
  }
}

async function handleUpload(event) {
  event.preventDefault();
  const file = documentInput.files[0];
  if (!file) {
    uploadStatus.textContent = "Choose a PDF or TXT file first.";
    return;
  }

  uploadButton.disabled = true;
  uploadStatus.textContent = "Uploading and rebuilding the vector store...";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/ingest", {
      method: "POST",
      body: formData,
    });
    const payload = await parseJsonResponse(response);
    uploadStatus.textContent = `${payload.message}. ${payload.chunks} chunks indexed.`;
    documentInput.value = "";
    checkHealth();
  } catch (error) {
    uploadStatus.textContent = error.message;
  } finally {
    uploadButton.disabled = false;
  }
}

async function loadHistory() {
  try {
    const response = await fetch(`/chat/sessions/${encodeURIComponent(state.sessionId)}`);
    const payload = await parseJsonResponse(response);
    renderHistory(payload.history);
  } catch (error) {
    renderHistory([]);
    appendMessage("assistant", "I could not load this session history.");
  }
}

async function clearCurrentSession() {
  try {
    const response = await fetch(`/chat/sessions/${encodeURIComponent(state.sessionId)}`, {
      method: "DELETE",
    });
    const payload = await parseJsonResponse(response);
    renderHistory(payload.history);
    renderSources([]);
    renderSourceSnippets([]);
    questionInput.focus();
  } catch (error) {
    appendMessage("assistant", error.message);
  }
}

function createNewSession() {
  state.sessionId = makeSessionId();
  storeSessionId(state.sessionId);
  sessionInput.value = state.sessionId;
  renderHistory([]);
  renderSources([]);
  renderSourceSnippets([]);
  questionInput.focus();
}

function switchSession() {
  state.sessionId = normalizeSessionId(sessionInput.value);
  sessionInput.value = state.sessionId;
  storeSessionId(state.sessionId);
  renderSources([]);
  renderSourceSnippets([]);
  loadHistory();
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const payload = await parseJsonResponse(response);
    apiStatus.textContent = payload.vectorstore_available ? "Ready" : "Needs docs";
    apiStatus.dataset.state = payload.vectorstore_available ? "ready" : "warning";
  } catch {
    apiStatus.textContent = "Offline";
    apiStatus.dataset.state = "error";
  }
}

async function parseJsonResponse(response) {
  let payload = {};
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }

  if (!response.ok) {
    const message = payload.detail || "Request failed. Check the API logs for details.";
    throw new Error(Array.isArray(message) ? message[0]?.msg || "Request failed." : message);
  }

  return payload;
}

function renderHistory(history) {
  messagesEl.innerHTML = "";
  if (!history.length) {
    const empty = document.createElement("div");
    empty.className = "welcome-state";
    empty.innerHTML = `
      <h2>Ask a practical training question</h2>
      <p>Start with eligibility, timing, school reporting, or document questions.</p>
    `;
    messagesEl.appendChild(empty);
    return;
  }

  history.forEach((message) => appendMessage(message.role, message.content));
}

function appendMessage(role, content) {
  messagesEl.querySelector(".welcome-state")?.remove();

  const id = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
  const article = document.createElement("article");
  article.className = `message ${role === "user" ? "message-user" : "message-assistant"}`;
  article.dataset.messageId = id;

  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "Assistant";

  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = content;

  article.append(label, body);
  messagesEl.appendChild(article);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return id;
}

function removeMessage(id) {
  messagesEl.querySelector(`[data-message-id="${id}"]`)?.remove();
}

function renderSources(sources) {
  sourcesEl.innerHTML = "";
  if (!sources || !sources.length) {
    const empty = document.createElement("span");
    empty.className = "empty-state";
    empty.textContent = "No sources yet";
    sourcesEl.appendChild(empty);
    return;
  }

  sources.forEach((source) => {
    const chip = document.createElement("span");
    chip.className = "source-chip";
    chip.textContent = source;
    sourcesEl.appendChild(chip);
  });
}

function renderSourceSnippets(snippets) {
  sourceSnippetsEl.innerHTML = "";
  if (!snippets || !snippets.length) return;

  snippets.forEach((item) => {
    const article = document.createElement("article");
    article.className = "snippet-card";

    const title = document.createElement("div");
    title.className = "snippet-title";
    title.textContent = `#${item.rank} ${item.source}`;

    const meta = document.createElement("div");
    meta.className = "snippet-meta";
    meta.textContent = `Page ${item.page}`;

    const text = document.createElement("p");
    text.textContent = item.snippet;

    article.append(title, meta, text);
    sourceSnippetsEl.appendChild(article);
  });
}

function setBusy(isBusy) {
  state.busy = isBusy;
  sendButton.disabled = isBusy;
  questionInput.disabled = isBusy;
}

function getStoredSessionId() {
  return normalizeSessionId(localStorage.getItem("opt-assistant-session-id"));
}

function storeSessionId(sessionId) {
  localStorage.setItem("opt-assistant-session-id", sessionId);
}

function normalizeSessionId(value) {
  return (value || "default").trim() || "default";
}

function makeSessionId() {
  if (crypto.randomUUID) {
    return `session-${crypto.randomUUID().slice(0, 8)}`;
  }
  return `session-${Date.now().toString(36)}`;
}
