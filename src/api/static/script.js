const API = '/api';
let currentSessionId = null;

// DOM refs
const el = {
  messages: document.getElementById('messages'),
  input: document.getElementById('question-input'),
  sendBtn: document.getElementById('send-btn'),
  statusBar: document.getElementById('status-bar'),
  statusText: document.getElementById('status-text'),
  sessionList: document.getElementById('session-list'),
  newChatBtn: document.getElementById('new-chat-btn'),
  multiHop: document.getElementById('multi-hop'),
  uploadBtn: document.getElementById('upload-btn'),
  fileInput: document.getElementById('file-input'),
  modal: document.getElementById('upload-modal'),
  modalFileInput: document.getElementById('modal-file-input'),
  dropZone: document.getElementById('drop-zone'),
  fileList: document.getElementById('file-list'),
  uploadSubmit: document.getElementById('upload-submit'),
  uploadStatus: document.getElementById('upload-status'),
  closeModal: document.querySelector('.close-modal'),
  toggleSidebar: document.getElementById('toggle-sidebar'),
  sidebar: document.getElementById('sidebar'),
};

// Auto-resize textarea
el.input.addEventListener('input', () => {
  el.input.style.height = 'auto';
  el.input.style.height = Math.min(el.input.scrollHeight, 120) + 'px';
});

// Send on Enter (Shift+Enter for newline)
el.input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
el.sendBtn.addEventListener('click', sendMessage);

// New chat
el.newChatBtn.addEventListener('click', createNewSession);
el.toggleSidebar.addEventListener('click', () => el.sidebar.classList.toggle('hidden'));

// Upload
el.uploadBtn.addEventListener('click', () => el.modal.classList.remove('hidden'));
el.closeModal.addEventListener('click', () => el.modal.classList.add('hidden'));
el.modal.addEventListener('click', (e) => { if (e.target === el.modal) el.modal.classList.add('hidden'); });
el.dropZone.addEventListener('click', () => el.modalFileInput.click());
el.modalFileInput.addEventListener('change', handleFiles);

// Drag & drop
el.dropZone.addEventListener('dragover', (e) => { e.preventDefault(); el.dropZone.classList.add('dragover'); });
el.dropZone.addEventListener('dragleave', () => el.dropZone.classList.remove('dragover'));
el.dropZone.addEventListener('drop', (e) => { e.preventDefault(); el.dropZone.classList.remove('dragover'); handleFileList(e.dataTransfer.files); });
el.uploadSubmit.addEventListener('click', uploadFiles);

// Init
async function init() {
  await createNewSession();
  loadSessionList();
}
init();

// --- Chat ---
async function createNewSession() {
  setBusy(true);
  try {
    const res = await fetch(`${API}/session/create`, { method: 'POST' });
    const data = await res.json();
    currentSessionId = data.session_id;
    el.messages.innerHTML = '';
    el.input.disabled = false;
    el.input.focus();
    loadSessionList();
  } finally { setBusy(false); }
}

async function sendMessage() {
  const question = el.input.value.trim();
  if (!question || el.sendBtn.disabled) return;

  el.input.value = '';
  el.input.style.height = 'auto';
  appendMessage('user', question);
  setBusy(true, '正在检索并生成答案...');

  try {
    const res = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        session_id: currentSessionId,
        use_multi_hop: el.multiHop.checked,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    removeLastAssistant();
    appendMessage('assistant', data.answer || '未生成答案');
    loadSessionList();
  } catch (err) {
    appendMessage('assistant', `❌ 请求失败：${err.message}`);
  } finally { setBusy(false); }
}

function appendMessage(role, content) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = `<span class="role-tag">${role === 'user' ? '你' : 'RAG 助手'}</span>${formatContent(content)}`;
  el.messages.appendChild(div);
  el.messages.scrollTop = el.messages.scrollHeight;
  return div;
}

function removeLastAssistant() {
  const msgs = el.messages.querySelectorAll('.msg.assistant');
  if (msgs.length) msgs[msgs.length - 1].remove();
}

function formatContent(text) {
  if (typeof marked === 'undefined') {
    return text.replace(/\n/g, '<br>');
  }
  return marked.parse(text, { breaks: true, gfm: true });
}

function setBusy(busy, text) {
  el.sendBtn.disabled = busy;
  el.input.disabled = busy;
  el.statusBar.classList.toggle('hidden', !busy);
  if (text) el.statusText.textContent = text;
}

// --- Sessions ---
async function loadSessionList() {
  // We don't have a list-all endpoint, so we just keep track locally
  // For simplicity, show sessions from our session_manager by visiting history
  // Actually, since we're using in-memory sessions, let's just show the current one
  // The session list would need a GET /api/sessions endpoint - skipping for simplicity
}

// --- File Upload ---
const selectedFiles = [];

function handleFiles(e) { handleFileList(e.target.files); }
function handleFileList(files) {
  for (const f of files) {
    if (!selectedFiles.find(sf => sf.name === f.name && sf.size === f.size)) {
      selectedFiles.push(f);
    }
  }
  renderFileList();
}

function renderFileList() {
  el.fileList.innerHTML = selectedFiles.map(f => `<li>📄 ${f.name} (${(f.size/1024).toFixed(1)} KB)</li>`).join('');
  el.uploadSubmit.disabled = selectedFiles.length === 0;
}

async function uploadFiles() {
  if (selectedFiles.length === 0) return;
  el.uploadSubmit.disabled = true;
  el.uploadStatus.textContent = '正在构建索引...';
  el.uploadStatus.className = '';

  try {
    const formData = new FormData();
    for (const f of selectedFiles) {
      formData.append('files', f);
    }

    const res = await fetch(`${API}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    el.uploadStatus.textContent = data.message;
    el.uploadStatus.className = 'success';

    setTimeout(() => {
      el.uploadStatus.textContent = '';
      el.uploadStatus.className = '';
      selectedFiles.length = 0;
      renderFileList();
      el.modal.classList.add('hidden');
    }, 3000);
  } catch (err) {
    el.uploadStatus.textContent = `上传失败：${err.message}`;
    el.uploadStatus.className = 'error';
  } finally { el.uploadSubmit.disabled = false; }
}

// --- Session click (switch session) ---
// For the session list, we'd need persistent storage. Since we use in-memory,
// clicking on a session isn't fully supported. We'll add simple local tracking.
