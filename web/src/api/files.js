/**
 * Files API — 文件上传与管理
 *
 * POST   /api/v1/files                上传文件 (multipart/form-data)
 * POST   /api/v1/files/from-url       从 URL 抓取文件
 * GET    /api/v1/files                文件列表 (分页)
 * GET    /api/v1/files/{id}           文件元信息
 * GET    /api/v1/files/{id}/download  下载/重定向
 * DELETE /api/v1/files/{id}           删除文件
 */

import { get, del, request } from './client'

/**
 * 上传文件 (multipart)
 * @param {File} file          - 浏览器 File 对象
 * @param {object} [options]
 * @param {string} [options.originalName] - 覆盖文件名
 * @param {string} [options.mimeType]     - 覆盖 MIME 类型
 * @returns {Promise<{
 *   file_id: string,
 *   content_hash: string,
 *   original_name: string,
 *   display_name: string,
 *   size_bytes: number,
 *   mime_type: string,
 *   uploaded_at: string
 * }>}
 */
export function uploadFile(file, options = {}) {
  const form = new FormData()
  form.append('file', file)
  if (options.originalName) form.append('original_name', options.originalName)
  if (options.mimeType) form.append('mime_type', options.mimeType)
  return request('/api/v1/files', { method: 'POST', body: form })
}

/**
 * 从 URL 抓取文件 (支持 http/https/s3/oss)
 * @param {string} url               - 源 URL
 * @param {object} [options]
 * @param {string} [options.originalName]
 * @param {string} [options.mimeType]
 * @returns {Promise<FileOut>}
 */
export const uploadFromUrl = (url, options = {}) =>
  request('/api/v1/files/from-url', {
    method: 'POST',
    body: {
      url,
      original_name: options.originalName || null,
      mime_type: options.mimeType || null,
    },
  })

/**
 * 文件列表
 * @param {object} [params]
 * @param {number} [params.limit=50]
 * @param {number} [params.offset=0]
 * @returns {Promise<{ items: FileOut[], total: number, limit: number, offset: number }>}
 */
export const listFiles = (params = {}) =>
  get('/api/v1/files', { limit: 50, offset: 0, ...params })

/**
 * 获取单个文件元信息
 * @param {string} fileId
 * @returns {Promise<FileOut>}
 */
export const getFile = (fileId) => get(`/api/v1/files/${fileId}`)

/**
 * 构建文件下载 URL (用于 <a href> 或 window.open)
 * @param {string} fileId
 * @returns {string}
 */
export const fileDownloadUrl = (fileId) =>
  `${import.meta.env.VITE_API_BASE || ''}/api/v1/files/${fileId}/download`

/**
 * 构建文件预览 URL (用于 <iframe> 内嵌显示，始终 inline 不重定向)
 * @param {string} fileId
 * @returns {string}
 */
export const filePreviewUrl = (fileId) =>
  `${import.meta.env.VITE_API_BASE || ''}/api/v1/files/${fileId}/preview`

/**
 * 删除文件
 * @param {string} fileId
 * @returns {Promise<null>}
 */
export const deleteFile = (fileId) => del(`/api/v1/files/${fileId}`)
