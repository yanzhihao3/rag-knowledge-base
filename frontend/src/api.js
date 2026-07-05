import axios from 'axios'

const http = axios.create({
  baseURL: '/',
  timeout: 60000,
})

// ========== 知识库 ==========
export function getKnowledgeBase(knowledgeId, token = '1') {
  return http.get('/v1/knowledge_base', { params: { knowledge_id: knowledgeId, token } })
}

export function createKnowledgeBase(title, category, token = '1') {
  return http.post('/v1/knowledge_base', { title, category, owner_id: 0, department_id: 0, token })
}

export function deleteKnowledgeBase(knowledgeId, token = '1') {
  return http.delete('/v1/knowledge_base', { params: { knowledge_id: knowledgeId, token } })
}

// ========== 文档 ==========
export function getDocument(documentId, token = '1') {
  return http.get('/v1/document', { params: { document_id: documentId, token } })
}

export function uploadDocument(knowledgeId, title, category, file, token = '1') {
  const formData = new FormData()
  formData.append('knowledge_id', knowledgeId)
  formData.append('title', title)
  formData.append('category', category)
  formData.append('file', file)
  formData.append('token', token)
  return http.post('/v1/document', formData)
}

export function deleteDocument(documentId, token = '1') {
  return http.delete('/v1/document', { params: { document_id: documentId, token } })
}

// ========== 聊天 ==========
export function chat(knowledgeId, messages) {
  return http.post('/chat', { knowledge_id: knowledgeId, message: messages })
}
