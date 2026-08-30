/**
 * FinSight Conversational AI Frontend Client
 * 
 * Connects to FastAPI POST /ask with multi-turn session persistence,
 * execution mode tracking (REAL_LLM vs MOCK_FALLBACK), and structured facts rendering.
 */

// --- Configuration & State ---
const CONFIG = {
  API_BASE_URL: window.location.origin.includes("http") ? "" : "http://127.0.0.1:8000",
  USER_ID: 1, // Aarav Sharma (Default synthetic test user)
};

const state = {
  conversationId: null,
  conversationStatus: "active",
  executionMode: "READY",
  isLoading: false,
};

// --- DOM Elements ---
const chatContainer = document.getElementById("chatContainer");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const typingIndicator = document.getElementById("typingIndicator");
const sessionIdDisplay = document.getElementById("sessionIdDisplay");
const sessionStatusIndicator = document.getElementById("sessionStatusIndicator");
const executionModeText = document.getElementById("executionModeText");
const executionBadge = document.getElementById("executionBadge");
const convStatusText = document.getElementById("convStatusText");
const quickChips = document.querySelectorAll(".chip");

// --- Initialization ---
document.addEventListener("DOMContentLoaded", () => {
  updateSessionUI();

  // Quick chip click listeners
  quickChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-query");
      if (query && !state.isLoading) {
        chatInput.value = query;
        handleSendMessage();
      }
    });
  });

  // New Chat button listener
  newChatBtn.addEventListener("click", handleNewChat);

  // Form submit listener
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleSendMessage();
  });

  chatInput.focus();
});

// --- Chat Actions ---

/**
 * Handles sending a user query to POST /ask with session memory.
 */
async function handleSendMessage() {
  const query = chatInput.value.trim();
  if (!query || state.isLoading) return;

  // 1. Clear input & append user message
  chatInput.value = "";
  appendUserMessage(query);
  setLoading(true);

  // 2. Build payload matching exact contract
  const payload = {
    user_id: CONFIG.USER_ID,
    query: query,
    conversation_id: state.conversationId,
  };

  try {
    const url = `${CONFIG.API_BASE_URL}/ask`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned HTTP ${response.status}`);
    }

    const data = await response.json();

    // 3. Update conversation state
    state.conversationId = data.conversation_id || state.conversationId;
    state.conversationStatus = data.conversation_status || "active";
    state.executionMode = data.execution_mode || "REAL_LLM";

    updateSessionUI();

    // 4. Append assistant response bubble
    appendAssistantMessage(data);

  } catch (error) {
    console.error("Error asking FinSight copilot:", error);
    appendErrorMessage(`I encountered an issue: ${error.message}. Please make sure the FastAPI backend is running.`);
  } finally {
    setLoading(false);
    chatInput.focus();
  }
}

/**
 * Resets conversational memory and starts a new session.
 */
function handleNewChat() {
  state.conversationId = null;
  state.conversationStatus = "active";
  updateSessionUI();

  // Add system notice divider
  const noticeDiv = document.createElement("div");
  noticeDiv.className = "session-notice";
  noticeDiv.style.textAlign = "center";
  noticeDiv.style.margin = "12px 0";
  noticeDiv.style.fontSize = "0.75rem";
  noticeDiv.style.color = "var(--text-muted)";
  noticeDiv.innerHTML = `<span>─── Started fresh conversation session ───</span>`;
  chatContainer.appendChild(noticeDiv);
  scrollToBottom();
}

// --- UI Rendering Helpers ---

function updateSessionUI() {
  // Update Session ID display
  sessionIdDisplay.textContent = state.conversationId ? state.conversationId : "None (Fresh)";
  
  // Update Status Pill
  if (state.conversationStatus === "awaiting_clarification") {
    sessionStatusIndicator.className = "status-pill clarify";
    sessionStatusIndicator.textContent = "Awaiting Clarification";
    convStatusText.textContent = "Clarifying";
    convStatusText.style.color = "var(--accent-amber)";
  } else {
    sessionStatusIndicator.className = "status-pill active";
    sessionStatusIndicator.textContent = state.conversationId ? "Active Turn" : "Idle";
    convStatusText.textContent = state.conversationId ? "Active" : "New";
    convStatusText.style.color = "var(--text-primary)";
  }

  // Update Execution Mode Badge
  executionModeText.textContent = state.executionMode;
  if (state.executionMode === "REAL_LLM") {
    executionModeText.style.color = "var(--accent-green)";
  } else if (state.executionMode === "MOCK_FALLBACK") {
    executionModeText.style.color = "var(--accent-amber)";
  } else {
    executionModeText.style.color = "var(--accent-primary)";
  }
}

function appendUserMessage(text) {
  const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  
  const wrapper = document.createElement("div");
  wrapper.className = "message-wrapper user";
  wrapper.innerHTML = `
    <div class="avatar user-avatar">You</div>
    <div class="message-content">
      <div class="message-bubble">
        <p>${escapeHtml(text)}</p>
      </div>
      <div class="message-meta">
        <span class="timestamp">${timeStr}</span>
      </div>
    </div>
  `;

  chatContainer.appendChild(wrapper);
  scrollToBottom();
}

function appendAssistantMessage(data) {
  const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const answerText = data.answer_text || "No response received.";
  const execMode = data.execution_mode || state.executionMode;
  const isMock = execMode === "MOCK_FALLBACK";

  // Build optional structured facts card
  const factsHtml = buildStructuredFactsHtml(data.structured_data);

  const wrapper = document.createElement("div");
  wrapper.className = "message-wrapper assistant";
  wrapper.innerHTML = `
    <div class="avatar assistant-avatar">FS</div>
    <div class="message-content">
      <div class="message-bubble">
        <p>${escapeHtml(answerText)}</p>
        ${factsHtml}
      </div>
      <div class="message-meta">
        <span class="timestamp">${timeStr}</span>
        <span class="engine-tag ${isMock ? 'mock' : ''}" title="Execution engine used for this turn">
          ${escapeHtml(execMode)}
        </span>
        ${data.conversation_status === 'awaiting_clarification' ? '<span class="status-pill clarify">Clarification Requested</span>' : ''}
      </div>
    </div>
  `;

  chatContainer.appendChild(wrapper);
  scrollToBottom();
}

function appendErrorMessage(errorText) {
  const wrapper = document.createElement("div");
  wrapper.className = "message-wrapper assistant";
  wrapper.innerHTML = `
    <div class="avatar assistant-avatar" style="background: #ef4444;">!</div>
    <div class="message-content">
      <div class="message-bubble" style="border-color: rgba(239, 68, 68, 0.4); background: rgba(239, 68, 68, 0.1);">
        <p style="color: #fca5a5;">${escapeHtml(errorText)}</p>
      </div>
    </div>
  `;
  chatContainer.appendChild(wrapper);
  scrollToBottom();
}

function buildStructuredFactsHtml(structured) {
  if (!structured || typeof structured !== "object") return "";

  // Affordability breakdown
  if ("can_afford" in structured) {
    const canAfford = structured.can_afford;
    const balanceAfter = formatCurrency(structured.balance_after);
    const bills = formatCurrency(structured.upcoming_bills);
    const goalImpact = structured.savings_goal_impact_months;

    return `
      <div class="structured-facts-card">
        <div class="structured-header">
          <span>📊 Authoritative Engine Facts</span>
        </div>
        <div class="fact-row">
          <span class="fact-label">Affordability Decision:</span>
          <span class="fact-value ${canAfford ? 'positive' : 'negative'}">${canAfford ? '✅ Affordable' : '❌ Not Affordable'}</span>
        </div>
        <div class="fact-row">
          <span class="fact-label">Balance After Purchase:</span>
          <span class="fact-value">${balanceAfter}</span>
        </div>
        <div class="fact-row">
          <span class="fact-label">Reserved for Upcoming Bills:</span>
          <span class="fact-value">${bills}</span>
        </div>
        ${goalImpact ? `
        <div class="fact-row">
          <span class="fact-label">Goal Completion Delay:</span>
          <span class="fact-value">${goalImpact} month(s)</span>
        </div>` : ''}
      </div>
    `;
  }

  // Account balance fact
  if ("balance" in structured && !("total" in structured)) {
    return `
      <div class="structured-facts-card">
        <div class="structured-header"><span>💰 Authoritative Balance Fact</span></div>
        <div class="fact-row">
          <span class="fact-label">Verified Net Balance:</span>
          <span class="fact-value positive">${formatCurrency(structured.balance)}</span>
        </div>
      </div>
    `;
  }

  // Goal projection fact
  if ("current_months_remaining" in structured) {
    return `
      <div class="structured-facts-card">
        <div class="structured-header"><span>🎯 Goal Projection Timeline</span></div>
        <div class="fact-row">
          <span class="fact-label">Remaining Timeline:</span>
          <span class="fact-value">${structured.current_months_remaining} months</span>
        </div>
      </div>
    `;
  }

  return "";
}

function setLoading(isLoading) {
  state.isLoading = isLoading;
  typingIndicator.style.display = isLoading ? "flex" : "none";
  sendBtn.disabled = isLoading;
  chatInput.disabled = isLoading;
  if (isLoading) {
    scrollToBottom();
  }
}

function scrollToBottom() {
  setTimeout(() => {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }, 30);
}

function formatCurrency(val) {
  if (val === null || val === undefined) return "₹0.00";
  const num = parseFloat(val);
  if (isNaN(num)) return `₹${val}`;
  return `₹${num.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function escapeHtml(str) {
  if (!str) return "";
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
