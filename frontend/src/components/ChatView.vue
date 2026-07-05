<template>
  <main class="chat">
    <!-- 消息区域 -->
    <div class="messages" ref="messagesRef">
      <div v-if="messages.length === 0" class="welcome">
        <div class="welcome-icon">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
        </div>
        <h3>有什么我可以帮你的？</h3>
        <p>输入问题开始与知识库对话</p>
      </div>

      <div
        v-for="(msg, idx) in messages"
        :key="idx"
        class="msg-wrapper"
        :class="msg.role === 'user' ? 'is-user' : 'is-bot'"
      >
        <div class="msg">
          <div class="msg-avatar">
            <svg v-if="msg.role === 'user'" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
            </svg>
            <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 2a10 10 0 1 0 10 10h-10V2z"/><path d="M22 12A10 10 0 0 0 12 2v10h10z"/>
            </svg>
          </div>
          <div class="msg-bubble">{{ msg.content }}</div>
          <!-- 检索详情 -->
          <div v-if="msg.debug" class="debug-wrap">
            <button class="debug-toggle" @click="msg._showDebug = !msg._showDebug">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
              </svg>
              {{ msg._showDebug ? '收起' : '显示' }}检索详情
            </button>
            <div v-if="msg._showDebug" class="debug-panel">
              <div class="debug-row" v-if="msg.debug.rewritten_query">
                <span class="debug-label">改写后 query</span>
                <code class="debug-value">{{ msg.debug.rewritten_query }}</code>
              </div>
              <div v-if="msg.debug.chunks?.length" class="debug-row">
                <span class="debug-label">引用来源 ({{ msg.debug.chunks.length }} 个片段)</span>
              </div>
              <div v-for="(chunk, ci) in msg.debug.chunks" :key="ci" class="chunk-item">
                <div class="chunk-meta">
                  <span>文档 #{{ chunk.document_id }}</span>
                  <span>第 {{ chunk.page_number }} 页</span>
                  <span v-if="chunk.rrf_score">RRF: {{ chunk.rrf_score }}</span>
                  <span v-if="chunk.rerank_score">重排: {{ chunk.rerank_score }}</span>
                </div>
                <div class="chunk-text">{{ chunk.content }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 打字指示器 -->
      <div v-if="loading" class="msg-wrapper is-bot">
        <div class="msg">
          <div class="msg-avatar">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 2a10 10 0 1 0 10 10h-10V2z"/><path d="M22 12A10 10 0 0 0 12 2v10h10z"/>
            </svg>
          </div>
          <div class="msg-bubble typing">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
          </div>
        </div>
      </div>
    </div>

    <!-- 输入区域 -->
    <div class="input-area">
      <div class="input-inner">
        <textarea
          ref="textareaRef"
          v-model="inputText"
          class="input-field"
          placeholder="输入你的问题..."
          rows="1"
          @keydown.enter="onEnter"
          @input="autoResize"
          :disabled="loading"
        ></textarea>
        <button
          class="btn-send"
          :class="{ active: inputText.trim() && !loading }"
          :disabled="!inputText.trim() || loading"
          @click="sendMessage"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>
      <p class="input-hint">Enter 发送 · Shift+Enter 换行</p>
    </div>
  </main>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { chat } from '../api.js'

const props = defineProps({ knowledgeId: { type: Number, required: true } })

const messages = ref([])
const inputText = ref('')
const loading = ref(false)
const messagesRef = ref(null)
const textareaRef = ref(null)

watch(() => props.knowledgeId, () => {
  messages.value = []
  inputText.value = ''
})

function autoResize() {
  const el = textareaRef.value
  if (el) {
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }
}

function onEnter(e) {
  if (!e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
}

async function sendMessage() {
  const text = inputText.value.trim()
  if (!text || loading.value) return

  messages.value.push({ role: 'user', content: text })
  inputText.value = ''
  if (textareaRef.value) {
    textareaRef.value.style.height = 'auto'
  }
  scrollDown()

  loading.value = true
  try {
    const res = await chat(props.knowledgeId, messages.value)
    const newMessages = res.data.message
    const debugInfo = res.data.debug_info
    const last = newMessages[newMessages.length - 1]
    if (last && last.role === 'system') {
      messages.value.push({ role: 'assistant', content: last.content, debug: debugInfo })
    }
  } catch {
    ElMessage.error('请求失败，请检查后端是否运行')
  } finally {
    loading.value = false
    scrollDown()
  }
}

function scrollDown() {
  nextTick(() => {
    if (messagesRef.value) {
      messagesRef.value.scrollTo({
        top: messagesRef.value.scrollHeight,
        behavior: 'smooth',
      })
    }
  })
}
</script>

<style scoped>
.chat {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: var(--bg-card);
}

/* ═══ 消息区 ═══ */
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 32px 40px;
  scroll-behavior: smooth;
}

.messages::-webkit-scrollbar { width: 6px; }
.messages::-webkit-scrollbar-track { background: transparent; }
.messages::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
.messages::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

.welcome {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--text-muted);
  animation: fadeIn 0.6s ease;
}

.welcome-icon { margin-bottom: 16px; opacity: 0.5; }

.welcome h3 {
  font-family: var(--font-heading);
  font-size: 18px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 6px;
}

.welcome p { font-size: 13px; }

.msg-wrapper { margin-bottom: 24px; animation: msgIn 0.35s ease; }
.msg-wrapper.is-user { display: flex; justify-content: flex-end; }

.msg { display: flex; gap: 12px; max-width: 72%; }
.msg-wrapper.is-user .msg { flex-direction: row-reverse; }

.msg-avatar {
  width: 34px; height: 34px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; margin-top: 4px;
}

.msg-wrapper.is-user .msg-avatar { background: var(--color-primary); color: #fff; }
.msg-wrapper.is-bot .msg-avatar { background: #f0ebe4; color: var(--color-primary); }

.msg-bubble {
  padding: 12px 18px; border-radius: 14px; font-size: 14px;
  line-height: 1.7; white-space: pre-wrap; word-break: break-word;
}

.msg-wrapper.is-user .msg-bubble { background: var(--color-primary); color: #fff; border-bottom-right-radius: 4px; }
.msg-wrapper.is-bot .msg-bubble { background: #f7f4f0; color: var(--text-primary); border-bottom-left-radius: 4px; }

.typing { display: flex; gap: 5px; align-items: center; padding: 14px 20px !important; }

.typing-dot {
  width: 8px; height: 8px; background: var(--text-muted);
  border-radius: 50%; animation: typingBounce 1.4s infinite ease-in-out;
}

.typing-dot:nth-child(1) { animation-delay: 0s; }
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes typingBounce {
  0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
  40% { transform: translateY(-6px); opacity: 1; }
}

/* ═══ 输入区 ═══ */
.input-area {
  padding: 16px 40px 20px;
  background: var(--bg-card);
  border-top: 1px solid var(--border-light);
}

.input-inner {
  display: flex; gap: 10px; align-items: flex-end;
  background: #f7f4f0; border: 1px solid var(--border-light);
  border-radius: var(--radius-lg); padding: 8px 8px 8px 18px;
  transition: border-color var(--transition), box-shadow var(--transition);
}

.input-inner:focus-within { border-color: var(--color-primary); box-shadow: 0 0 0 3px rgba(45, 106, 106, 0.1); }

.input-field {
  flex: 1; border: none; background: transparent;
  font-family: var(--font-body); font-size: 14px;
  color: var(--text-primary); resize: none; outline: none;
  line-height: 1.6; max-height: 160px;
}

.input-field::placeholder { color: var(--text-muted); }

.btn-send {
  width: 38px; height: 38px; border: none; border-radius: 50%;
  background: var(--border-color); color: #fff; cursor: pointer;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  transition: all var(--transition);
}

.btn-send.active { background: var(--color-primary); }
.btn-send.active:hover { background: var(--color-primary-hover); transform: scale(1.05); }
.btn-send.active:active { transform: scale(0.95); }
.btn-send:disabled { cursor: not-allowed; }

.input-hint { margin-top: 6px; font-size: 11px; color: var(--text-muted); text-align: right; }

@keyframes msgIn {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

/* ═══ 检索详情 ═══ */
.debug-wrap { margin-top: 8px; }

.debug-toggle {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 10px; border: 1px solid var(--border-light);
  background: var(--bg-card); color: var(--text-muted);
  border-radius: 12px; font-size: 11px; cursor: pointer;
  transition: all var(--transition);
}

.debug-toggle:hover { border-color: var(--color-primary); color: var(--color-primary); }

.debug-panel {
  margin-top: 8px; background: #faf8f5;
  border: 1px solid var(--border-light); border-radius: var(--radius-md);
  padding: 12px; font-size: 12px;
}

.debug-row { margin-bottom: 10px; }

.debug-label {
  display: block; font-weight: 600; color: var(--text-secondary);
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;
}

.debug-value {
  display: block; padding: 6px 10px; background: var(--bg-card);
  border: 1px solid var(--border-light); border-radius: var(--radius-sm);
  color: var(--color-primary); font-size: 12px; line-height: 1.5;
}

.chunk-item {
  padding: 8px 10px; margin-bottom: 6px; background: var(--bg-card);
  border: 1px solid var(--border-light); border-radius: var(--radius-sm);
}

.chunk-item:last-child { margin-bottom: 0; }

.chunk-meta { display: flex; gap: 10px; font-size: 11px; color: var(--text-muted); margin-bottom: 4px; }

.chunk-text {
  font-size: 12px; color: var(--text-secondary); line-height: 1.6;
  overflow: hidden; text-overflow: ellipsis;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
}
</style>
