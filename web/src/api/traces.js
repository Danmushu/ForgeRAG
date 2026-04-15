/**
 * Traces API — 查询审计日志 (可视化检索流程)
 *
 * GET    /api/v1/traces              查询历史列表
 * GET    /api/v1/traces/{id}         单个 trace 详情 (含完整 phases)
 * DELETE /api/v1/traces/{id}         删除
 *
 * trace_json 结构 (用于前端渲染 Gantt/瀑布图):
 * {
 *   query, timestamp, total_ms, total_llm_ms, total_llm_calls,
 *   phases: [
 *     { name, started_at_ms, duration_ms, inputs, outputs, llm_calls, details }
 *   ],
 *   generation: { model, latency_ms, usage, citations_used, answer_length }
 * }
 */

import { get, del } from './client'

/**
 * 查询历史列表 (分页,按时间倒序)
 * @param {object} [params]
 * @param {number} [params.limit=50]
 * @param {number} [params.offset=0]
 * @returns {Promise<{
 *   items: Array<{
 *     trace_id: string, query: string, timestamp: string,
 *     total_ms: number, total_llm_ms: number, total_llm_calls: number,
 *     answer_model: string|null, finish_reason: string|null,
 *     citations_used: string[]
 *   }>,
 *   total: number, limit: number, offset: number
 * }>}
 */
export const listTraces = (params = {}) =>
  get('/api/v1/traces', { limit: 50, offset: 0, ...params })

/**
 * 单个 trace 详情 (含完整 trace_json)
 * @param {string} traceId
 * @returns {Promise<{
 *   trace_id: string, query: string, timestamp: string,
 *   total_ms: number, total_llm_ms: number, total_llm_calls: number,
 *   answer_text: string|null, answer_model: string|null,
 *   finish_reason: string|null, citations_used: string[],
 *   trace_json: TraceJSON, metadata_json: object
 * }>}
 */
export const getTrace = (traceId) => get(`/api/v1/traces/${traceId}`)

/**
 * 删除 trace
 * @param {string} traceId
 * @returns {Promise<{ deleted: string }>}
 */
export const deleteTrace = (traceId) => del(`/api/v1/traces/${traceId}`)
