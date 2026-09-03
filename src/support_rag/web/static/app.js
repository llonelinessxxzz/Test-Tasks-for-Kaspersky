"use strict";
const $ = (id) => document.getElementById(id);
let csrf = null;
let demoMode = false;
let activeChat = null;
let busy = false;
let currentMessages = [];
let selectionVersion = 0;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
    headers: {"Content-Type": "application/json", ...(csrf ? {"X-CSRF-Token": csrf} : {}), ...options.headers}
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401 && path !== "/api/login") showLogin();
    const detail = typeof payload.detail === "string" ? payload.detail : "Не удалось выполнить запрос.";
    throw new Error(detail);
  }
  return payload;
}

function showLogin() {
  csrf = null;
  activeChat = null;
  currentMessages = [];
  $("workspace").hidden = true;
  $("login-screen").hidden = demoMode;
  $("session-loading").hidden = !demoMode;
  if (demoMode) {
    $("session-message").textContent = "Сессия завершена. Откройте чат снова.";
    $("session-retry").hidden = false;
  }
  $("messages").replaceChildren();
  $("chat-list").replaceChildren();
}

async function openWorkspace() {
  $("session-loading").hidden = true;
  $("login-screen").hidden = true;
  $("workspace").hidden = false;
  $("access-code").value = "";
  resizeQuestion();
  const chats = await refreshChats();
  if (chats.length) await selectChat(chats[0].id);
  else renderEmpty();
  api("/api/health").then((status) => {
    const ready = status.retrieval && status.generation;
    $("service-status").classList.toggle("offline", !ready);
    $("service-status").replaceChildren(element("span", "dot"), document.createTextNode(ready ? "WORK" : "Сервисы запускаются"));
  }).catch(() => {
    $("service-status").classList.add("offline");
    $("service-status").textContent = "Нет связи";
  });
}

async function refreshChats() {
  const chats = await api("/api/chats");
  $("chat-count").textContent = chats.length;
  $("chat-list").replaceChildren();
  if (!chats.length) $("chat-list").append(element("p", "empty-history", "Здесь появятся ваши диалоги."));
  for (const chat of chats) {
    const row = element("div", "chat-entry" + (chat.id === activeChat ? " active" : ""));
    const open = element("button", "chat-open", chat.title);
    open.title = chat.title;
    open.setAttribute("aria-current", chat.id === activeChat ? "page" : "false");
    open.addEventListener("click", () => selectChat(chat.id).catch(showError));
    const remove = element("button", "chat-remove", "×");
    remove.setAttribute("aria-label", "Удалить диалог: " + chat.title);
    remove.addEventListener("click", () => removeChat(chat.id).catch(showError));
    row.append(open, remove);
    $("chat-list").append(row);
  }
  return chats;
}

function renderEmpty() {
  activeChat = null;
  currentMessages = [];
  $("chat-title").textContent = "Новый диалог";
  $("messages").replaceChildren();
  $("welcome").hidden = false;
  $("request-error").hidden = true;
}

async function selectChat(id) {
  if (busy) return;
  const version = ++selectionVersion;
  const chat = await api("/api/chats/" + id);
  if (version !== selectionVersion) return;
  activeChat = id;
  currentMessages = chat.messages;
  renderChat(chat);
  await refreshChats();
  $("sidebar").classList.remove("open");
}

function renderChat(chat) {
  $("chat-title").textContent = chat.title;
  $("welcome").hidden = chat.messages.length > 0;
  $("messages").replaceChildren();
  for (const message of chat.messages) renderMessage(message);
  scrollBottom();
}

function renderMessage(message) {
  const block = element("article", "message " + message.role);
  if (message.role === "user") {
    block.append(element("div", "user-bubble", message.content));
  } else {
    const heading = element("div", "assistant-heading");
    heading.append(element("span", "assistant-symbol", "✳"), document.createTextNode("Support"), element("span", "draft-label", "ЧЕРНОВИК"));
    block.append(heading, element("div", "answer-text", message.content));
    const data = message.payload;
    if (data) {
      const badge = element("span", "confidence " + data.confidence.level, data.confidence.label);
      badge.title = data.confidence.reason;
      const meta = element("div", "answer-meta");
      meta.append(badge, element("span", "", data.latency_seconds.toFixed(1) + " с"));
      block.append(meta);
      const grouped = new Map();
      for (const source of data.response.sources.filter(s => s.cited)) {
        if (!grouped.has(source.document_id)) grouped.set(source.document_id, {source, numbers: []});
        grouped.get(source.document_id).numbers.push(source.source_number);
      }
      if (grouped.size) {
        const sources = element("div", "sources");
        for (const {source, numbers} of grouped.values()) {
          let url;
          try { url = new URL(source.source_url); } catch { continue; }
          if (!["https:", "http:"].includes(url.protocol)) continue;
          const link = element("a", "source-card");
          link.href = url.href; link.target = "_blank"; link.rel = "noopener noreferrer";
          link.append(element("small", "", "SOURCE " + numbers.join(", ") + " · " + url.hostname), document.createTextNode(source.title + " ↗"));
          sources.append(link);
        }
        block.append(sources);
      }
      const details = element("details", "query-details");
      details.append(element("summary", "", "О проверке ответа"));
      details.append(element("p", "", data.confidence.reason));
      if (data.history_used) details.append(element("p", "", "Вопрос с учётом диалога: " + data.retrieval_query));
      block.append(details);
    }
  }
  $("messages").append(block);
}

function showError(error) {
  $("request-error").textContent = error.message || "Не удалось выполнить запрос.";
  $("request-error").hidden = false;
}
function scrollBottom() { $("conversation").scrollTop = $("conversation").scrollHeight; }
function resizeQuestion() {
  const field = $("question");
  field.style.height = "auto";
  field.style.height = field.scrollHeight + "px";
}
function setBusy(value) {
  busy = value;
  $("send").disabled = value;
  $("question").disabled = value;
  $("new-chat").disabled = value;
  $("logout").disabled = value;
  for (const button of $("chat-list").querySelectorAll("button")) button.disabled = value;
}

async function confirmAction(title, copy) {
  $("confirm-title").textContent = title;
  $("confirm-copy").textContent = copy;
  const dialog = $("confirm-dialog");
  dialog.returnValue = "";
  dialog.showModal();
  return new Promise(resolve => dialog.addEventListener("close", () => resolve(dialog.returnValue === "confirm"), {once: true}));
}

async function removeChat(id) {
  if (busy || !await confirmAction("Удалить диалог?", "Диалог и все его сообщения будут удалены. Отменить это действие нельзя.")) return;
  await api("/api/chats/" + id, {method: "DELETE"});
  if (activeChat === id) renderEmpty();
  const chats = await refreshChats();
  if (!activeChat && chats.length) await selectChat(chats[0].id);
}

$("login-form").addEventListener("submit", async event => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button[type="submit"]');
  button.disabled = true; $("login-error").textContent = "";
  try {
    const result = await api("/api/login", {method: "POST", body: JSON.stringify({password: $("access-code").value})});
    csrf = result.csrf;
    await openWorkspace();
  } catch (error) { $("login-error").textContent = error.message; }
  finally { button.disabled = false; }
});
$("show-code").addEventListener("click", () => {
  const show = $("access-code").type === "password";
  $("access-code").type = show ? "text" : "password";
  $("show-code").textContent = show ? "Скрыть" : "Показать";
});
$("new-chat").addEventListener("click", async () => {
  if (busy) return;
  ++selectionVersion;
  renderEmpty();
  await refreshChats().catch(showError);
  $("sidebar").classList.remove("open");
  $("question").focus();
});
$("toggle-sidebar").addEventListener("click", () => $("sidebar").classList.toggle("open"));
$("logout").addEventListener("click", async () => {
  if (!await confirmAction("Завершить сессию?", "После выхода история этой сессии не будет доступна при новом входе.")) return;
  try {
    await api("/api/logout", {method: "POST"});
    showLogin();
    if (demoMode) await startSession();
  }
  catch (error) { showError(error); }
});
document.querySelectorAll("[data-question]").forEach(button => button.addEventListener("click", () => {
  $("question").value = button.dataset.question;
  resizeQuestion();
  $("question").focus();
}));
$("question").addEventListener("input", resizeQuestion);
window.addEventListener("resize", resizeQuestion);
$("question").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!busy) $("chat-form").requestSubmit();
  }
});
$("chat-form").addEventListener("submit", async event => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question || busy) return;
  $("request-error").hidden = true;
  setBusy(true);
  try {
    if (!activeChat) {
      const chat = await api("/api/chats", {method: "POST"});
      activeChat = chat.id;
      currentMessages = [];
    }
    $("welcome").hidden = true;
    renderMessage({role: "user", content: question});
    const pending = element("div", "pending");
    pending.append(element("span", "dot"), document.createTextNode("Ищу источники и готовлю ответ…"));
    $("messages").append(pending); scrollBottom();
    const chat = await api("/api/chats/" + activeChat + "/messages", {
      method: "POST", body: JSON.stringify({question, use_history: $("use-history").checked})
    });
    currentMessages = chat.messages;
    renderChat(chat);
    $("question").value = "";
    resizeQuestion();
    await refreshChats();
  } catch (error) {
    if (csrf) {
      renderChat({title: $("chat-title").textContent, messages: currentMessages});
      showError(error);
    }
  } finally { setBusy(false); $("question").focus(); }
});
async function startSession() {
  $("workspace").hidden = true;
  $("login-screen").hidden = true;
  $("session-loading").hidden = false;
  $("session-message").textContent = "Открываем чат…";
  $("session-retry").hidden = true;
  try {
    let result = await api("/api/session");
    demoMode = result.demo_mode === true;
    if (!result.authenticated && demoMode) result = await api("/api/session", {method: "POST"});
    if (result.authenticated) { csrf = result.csrf; await openWorkspace(); }
    else showLogin();
  } catch (error) {
    $("workspace").hidden = true;
    $("login-screen").hidden = true;
    $("session-loading").hidden = false;
    $("session-message").textContent = error.message || "Не удалось подключиться к серверу.";
    $("session-retry").hidden = false;
  }
}
$("session-retry").addEventListener("click", startSession);
startSession();
