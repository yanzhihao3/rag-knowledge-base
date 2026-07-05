<template>
  <aside class="sidebar">
    <!-- 品牌区域 -->
    <div class="brand">
      <div class="brand-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
          <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
        </svg>
      </div>
      <div class="brand-text">
        <span class="brand-title">知识库</span>
        <span class="brand-sub">智能问答系统</span>
      </div>
    </div>

    <!-- 操作栏 -->
    <div class="toolbar">
      <button class="btn-primary" @click="showCreateDialog = true">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
        </svg>
        新建知识库
      </button>
    </div>

    <!-- 知识库列表 -->
    <div class="kb-list">
      <div
        v-for="kb in kbList"
        :key="kb.knowledge_id"
        class="kb-item"
        :class="{ active: kb.knowledge_id === currentKbId }"
        @click="selectKb(kb.knowledge_id)"
      >
        <div class="kb-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
          </svg>
        </div>
        <div class="kb-info">
          <span class="kb-title">{{ kb.title }}</span>
          <span class="kb-meta">{{ kb.category }}</span>
        </div>
        <button
          class="btn-delete"
          @click.stop="handleDeleteKb(kb.knowledge_id)"
          title="删除"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
          </svg>
        </button>
      </div>
    </div>

    <!-- 分隔线 + 文档区域 -->
    <div v-if="currentKbId" class="doc-section">
      <div class="doc-header">
        <span class="doc-label">文档</span>
        <span class="doc-count">{{ docList.length }}</span>
        <button class="btn-ghost" @click="showUpload = true">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
          上传
        </button>
      </div>
      <div class="doc-list">
        <div v-if="docList.length === 0" class="doc-empty">
          暂无文档，上传 PDF 开始
        </div>
        <div
          v-for="doc in docList"
          :key="doc.document_id"
          class="doc-item"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>
          </svg>
          <span class="doc-name">{{ doc.title }}</span>
          <button
            class="btn-delete-doc"
            @click.stop="handleDeleteDoc(doc.document_id)"
            title="删除文档"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>
      </div>
    </div>

    <!-- 新建知识库对话框 -->
    <el-dialog v-model="showCreateDialog" title="新建知识库" width="420px" :close-on-click-modal="false">
      <el-form :model="createForm" label-width="60px">
        <el-form-item label="名称">
          <el-input v-model="createForm.title" placeholder="知识库名称" />
        </el-form-item>
        <el-form-item label="分类">
          <el-input v-model="createForm.category" placeholder="分类（如：技术、管理）" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showCreateDialog = false">取消</el-button>
        <el-button type="primary" @click="submitCreate">创建</el-button>
      </template>
    </el-dialog>

    <!-- 上传文档对话框 -->
    <el-dialog v-model="showUpload" title="上传文档" width="420px" :close-on-click-modal="false">
      <el-form :model="uploadForm" label-width="60px">
        <el-form-item label="名称">
          <el-input v-model="uploadForm.title" placeholder="文档名称" />
        </el-form-item>
        <el-form-item label="分类">
          <el-input v-model="uploadForm.category" placeholder="文档分类" />
        </el-form-item>
        <el-form-item label="文件">
          <el-upload
            :auto-upload="false"
            :limit="1"
            accept=".pdf"
            :on-change="handleFileChange"
            class="custom-upload"
          >
            <button class="btn-upload" type="button">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
              选择 PDF 文件
            </button>
          </el-upload>
          <div v-if="selectedFile" class="file-name">{{ selectedFile.name }}</div>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showUpload = false">取消</el-button>
        <el-button type="primary" @click="submitUpload" :loading="uploading">上传</el-button>
      </template>
    </el-dialog>
  </aside>
</template>

<script setup>
import { ref, watch, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  createKnowledgeBase, deleteKnowledgeBase,
  uploadDocument, deleteDocument,
} from '../api.js'

const emit = defineEmits(['selectKb', 'createKb'])
const props = defineProps({ currentKbId: { type: Number, default: null } })

const kbList = ref([])
const docList = ref([])
const showCreateDialog = ref(false)
const showUpload = ref(false)
const uploading = ref(false)
const selectedFile = ref(null)

const createForm = ref({ title: '', category: '' })
const uploadForm = ref({ title: '', category: '' })

async function loadKbList() {
  try {
    const res = await fetch('/v1/knowledge_base/list?token=1')
    const data = await res.json()
    kbList.value = data.knowledge_list || []
  } catch {
    ElMessage.error('加载知识库失败')
  }
}

async function loadDocList(kbId) {
  if (!kbId) { docList.value = []; return }
  try {
    const res = await fetch(`/v1/document/list?knowledge_id=${kbId}&token=1`)
    const data = await res.json()
    docList.value = data.document_list || []
  } catch {
    ElMessage.error('加载文档失败')
  }
}

function selectKb(id) {
  emit('selectKb', id)
}

async function submitCreate() {
  try {
    const res = await createKnowledgeBase(createForm.value.title, createForm.value.category)
    if (res.data.response_code === 200) {
      ElMessage.success('创建成功')
      showCreateDialog.value = false
      createForm.value = { title: '', category: '' }
      await loadKbList()
      emit('createKb', res.data.knowledge_id)
    }
  } catch {
    ElMessage.error('创建失败')
  }
}

async function handleDeleteKb(id) {
  try {
    await ElMessageBox.confirm('确定删除该知识库及其所有文档？', '警告')
    await deleteKnowledgeBase(id)
    ElMessage.success('删除成功')
    if (props.currentKbId === id) emit('selectKb', null)
    await loadKbList()
  } catch {}
}

function handleFileChange(file) {
  selectedFile.value = file.raw
}

async function submitUpload() {
  if (!selectedFile.value) { ElMessage.warning('请选择文件'); return }
  if (!uploadForm.value.title) { ElMessage.warning('请输入文档名称'); return }
  uploading.value = true
  try {
    await uploadDocument(
      props.currentKbId,
      uploadForm.value.title,
      uploadForm.value.category,
      selectedFile.value,
    )
    ElMessage.success('上传成功，后台解析中...')
    showUpload.value = false
    uploadForm.value = { title: '', category: '' }
    selectedFile.value = null
    setTimeout(() => loadDocList(props.currentKbId), 2000)
  } catch {
    ElMessage.error('上传失败')
  } finally {
    uploading.value = false
  }
}

async function handleDeleteDoc(id) {
  try {
    await ElMessageBox.confirm('确定删除该文档？', '警告')
    await deleteDocument(id)
    ElMessage.success('删除成功')
    await loadDocList(props.currentKbId)
  } catch {}
}

watch(() => props.currentKbId, (id) => { loadDocList(id) })
onMounted(loadKbList)
</script>

<style scoped>
.sidebar {
  width: 280px;
  min-width: 280px;
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  position: relative;
  z-index: 1;
}

/* ─── 品牌 ─── */
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 20px 20px 16px;
}

.brand-icon {
  width: 38px;
  height: 38px;
  background: var(--color-primary);
  color: #fff;
  border-radius: var(--radius-md);
  display: flex;
  align-items: center;
  justify-content: center;
}

.brand-text {
  display: flex;
  flex-direction: column;
}

.brand-title {
  font-family: var(--font-heading);
  font-size: 17px;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1.3;
}

.brand-sub {
  font-size: 11px;
  color: var(--text-muted);
  letter-spacing: 0.5px;
}

/* ─── 操作栏 ─── */
.toolbar {
  padding: 0 16px 12px;
}

.btn-primary {
  width: 100%;
  padding: 9px 0;
  background: var(--color-primary);
  color: #fff;
  border: none;
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-family: var(--font-body);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  transition: background var(--transition);
}

.btn-primary:hover {
  background: var(--color-primary-hover);
}

/* ─── 知识库列表 ─── */
.kb-list {
  flex: 1;
  overflow-y: auto;
  padding: 0 10px 8px;
}

.kb-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  margin-bottom: 2px;
  transition: all var(--transition);
  position: relative;
}

.kb-item:hover {
  background: var(--bg-hover);
}

.kb-item.active {
  background: var(--color-primary-light);
}

.kb-item.active .kb-title {
  color: var(--color-primary);
  font-weight: 600;
}

.kb-icon {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  background: var(--bg-card);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  flex-shrink: 0;
  border: 1px solid var(--border-light);
}

.kb-item.active .kb-icon {
  background: var(--color-primary);
  color: #fff;
  border-color: var(--color-primary);
}

.kb-info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.kb-title {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.kb-meta {
  font-size: 11px;
  color: var(--text-muted);
}

.btn-delete {
  opacity: 0;
  width: 26px;
  height: 26px;
  border: none;
  background: transparent;
  color: var(--text-muted);
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: all var(--transition);
}

.kb-item:hover .btn-delete {
  opacity: 1;
}

.btn-delete:hover {
  background: rgba(181, 74, 58, 0.1);
  color: var(--color-danger);
}

/* ─── 文档区域 ─── */
.doc-section {
  border-top: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  background: rgba(0,0,0,0.015);
}

.doc-header {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 12px 16px 8px;
}

.doc-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.8px;
}

.doc-count {
  font-size: 11px;
  color: var(--text-muted);
  background: var(--bg-hover);
  padding: 0 6px;
  border-radius: 8px;
  line-height: 18px;
}

.btn-ghost {
  margin-left: auto;
  border: none;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 12px;
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  border-radius: var(--radius-sm);
  transition: all var(--transition);
}

.btn-ghost:hover {
  background: var(--bg-hover);
  color: var(--color-primary);
}

.doc-list {
  padding: 0 10px 10px;
  max-height: 180px;
  overflow-y: auto;
}

.doc-empty {
  padding: 12px;
  text-align: center;
  color: var(--text-muted);
  font-size: 12px;
}

.doc-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 8px;
  border-radius: var(--radius-sm);
  font-size: 12px;
  color: var(--text-secondary);
  transition: background var(--transition);
}

.doc-item:hover {
  background: var(--bg-hover);
}

.doc-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.btn-delete-doc {
  opacity: 0;
  width: 22px;
  height: 22px;
  border: none;
  background: transparent;
  color: var(--text-muted);
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all var(--transition);
}

.doc-item:hover .btn-delete-doc {
  opacity: 1;
}

.btn-delete-doc:hover {
  background: rgba(181, 74, 58, 0.1);
  color: var(--color-danger);
}

/* ─── 上传按钮（对话框内） ─── */
:deep(.custom-upload) {
  display: block;
}

.btn-upload {
  padding: 8px 16px;
  background: var(--color-primary-light);
  color: var(--color-primary);
  border: 1px dashed var(--color-primary);
  border-radius: var(--radius-sm);
  font-size: 13px;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  transition: all var(--transition);
}

.btn-upload:hover {
  background: var(--color-primary);
  color: #fff;
}

.file-name {
  margin-top: 6px;
  font-size: 12px;
  color: var(--text-secondary);
}
</style>
