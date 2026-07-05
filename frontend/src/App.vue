<template>
  <div class="app-root">
    <!-- 背景装饰 -->
    <div class="bg-ornament top-right"></div>

    <Sidebar
      :currentKbId="currentKbId"
      @select-kb="handleSelectKb"
      @create-kb="handleCreateKb"
    />
    <ChatView
      v-if="currentKbId"
      :key="currentKbId"
      :knowledgeId="currentKbId"
    />
    <div v-else class="empty-state">
      <div class="empty-inner">
        <div class="empty-icon">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
            <line x1="8" y1="7" x2="16" y2="7"/>
            <line x1="8" y1="11" x2="14" y2="11"/>
          </svg>
        </div>
        <h2>欢迎使用知识库</h2>
        <p>请从左侧选择一个知识库开始对话</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import Sidebar from './components/Sidebar.vue'
import ChatView from './components/ChatView.vue'

const currentKbId = ref(null)

function handleSelectKb(id) {
  currentKbId.value = id
}

function handleCreateKb(id) {
  currentKbId.value = id
}
</script>

<style>
/* ═══════════ 设计系统 ═══════════ */
:root {
  /* 颜色 */
  --bg-page: #f4f0ea;
  --bg-sidebar: #ede7df;
  --bg-card: #ffffff;
  --bg-hover: #e5ded5;
  --bg-active: #dce4e0;

  --text-primary: #1f1f1f;
  --text-secondary: #6b6258;
  --text-muted: #a69a8e;

  --color-primary: #2d6a6a;
  --color-primary-hover: #235454;
  --color-primary-light: #eaf2f2;
  --color-accent: #c97d32;
  --color-accent-light: #fdf4e8;
  --color-danger: #b54a3a;

  --border-color: #ddd6ce;
  --border-light: #e8e2da;

  --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.06);
  --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.08);
  --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.1);

  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 16px;

  --font-heading: 'Noto Serif SC', 'STSong', 'SimSun', serif;
  --font-body: -apple-system, 'PingFang SC', 'Microsoft YaHei', 'Noto Sans SC', sans-serif;

  --transition: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body, #app {
  height: 100%;
}

body {
  font-family: var(--font-body);
  color: var(--text-primary);
  background: var(--bg-page);
  -webkit-font-smoothing: antialiased;
}

/* ═══════════ Element Plus 覆盖 ═══════════ */
.el-dialog {
  --el-dialog-bg-color: var(--bg-card);
  border-radius: var(--radius-lg) !important;
  box-shadow: var(--shadow-lg) !important;
}

.el-dialog__title {
  font-family: var(--font-heading);
  font-weight: 600;
  color: var(--text-primary);
}

.el-button--primary {
  --el-button-bg-color: var(--color-primary);
  --el-button-border-color: var(--color-primary);
  --el-button-hover-bg-color: var(--color-primary-hover);
  --el-button-hover-border-color: var(--color-primary-hover);
}

.el-message {
  --el-message-bg-color: var(--bg-card);
  border-radius: var(--radius-md) !important;
  box-shadow: var(--shadow-md) !important;
}

/* ═══════════ 应用布局 ═══════════ */
.app-root {
  display: flex;
  height: 100vh;
  position: relative;
  overflow: hidden;
}

/* 背景装饰 */
.bg-ornament {
  position: fixed;
  pointer-events: none;
  z-index: 0;
  opacity: 0.3;
}

.bg-ornament.top-right {
  top: -120px;
  right: -120px;
  width: 360px;
  height: 360px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(45, 106, 106, 0.08) 0%, transparent 70%);
}

/* 空状态 */
.empty-state {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  z-index: 1;
}

.empty-inner {
  text-align: center;
  color: var(--text-muted);
}

.empty-icon {
  margin-bottom: 20px;
  color: var(--border-color);
}

.empty-inner h2 {
  font-family: var(--font-heading);
  font-size: 20px;
  color: var(--text-secondary);
  margin-bottom: 8px;
  font-weight: 600;
}

.empty-inner p {
  font-size: 14px;
}
</style>
