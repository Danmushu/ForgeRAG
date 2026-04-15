/**
 * Settings API — 运行时可编辑配置 (前端设置页面)
 *
 * GET    /api/v1/settings                       全部设置 (按 group 分组)
 * GET    /api/v1/settings/group/{group}         按组查
 * GET    /api/v1/settings/key/{key}             单个设置
 * PUT    /api/v1/settings/key/{key}             修改单个
 * PUT    /api/v1/settings                       批量修改
 * DELETE /api/v1/settings/key/{key}             重置为默认
 * POST   /api/v1/settings/reset-all             重置全部
 * POST   /api/v1/settings/reset-group/{group}   重置整组
 * POST   /api/v1/settings/apply                 强制重新应用
 *
 * Setting 结构:
 * {
 *   key:          "retrieval.rerank.enabled",  // 点分路径
 *   value_json:   false,                       // 当前值
 *   group_name:   "rerank",                    // 前端 tab 分组
 *   label:        "Enable rerank",             // 显示标签
 *   description:  "Use LLM to ...",            // 说明文字
 *   value_type:   "bool",                      // bool/int/float/string/enum/secret
 *   enum_options: null,                         // value_type=enum 时的可选值
 *   updated_at:   "2026-04-10T..."
 * }
 *
 * 分组 (group_name):
 *   llm                 → 生成模型配置
 *   embedding           → 嵌入模型配置
 *   retrieval_vector    → 路径A: 向量检索 (开关 + top_k)
 *   retrieval_bm25      → 路径B: BM25 检索 (开关 + top_k + k1 + b)
 *   retrieval_tree      → 路径C: 树导航 (开关 + LLM + workers + 早停)
 *   retrieval_fusion    → 融合参数 (RRF k + budget)
 *   retrieval_expansion → 上下文扩展 (query expansion + 后代/兄弟/跨引用)
 *   rerank              → 重排 (开关 + model)
 *   images              → 图片 VLM 描述
 *   parsing             → 解析策略 (MinerU + chunk size)
 *
 * 前端渲染规则:
 *   value_type=bool   → toggle 开关
 *   value_type=int    → number input
 *   value_type=float  → number input (step=0.1)
 *   value_type=string → text input
 *   value_type=secret → password input (显示 ***)
 *   value_type=enum   → select dropdown (options 来自 enum_options)
 */

import { get, put, del, post } from './client'

/**
 * 全部设置 (按 group 分组)
 * @returns {Promise<{
 *   groups: Record<string, SettingOut[]>
 * }>}
 */
export const getAllSettings = () => get('/api/v1/settings')

/**
 * 按组获取设置
 * @param {string} groupName - e.g. 'retrieval_vector', 'llm', 'rerank'
 * @returns {Promise<SettingOut[]>}
 */
export const getSettingsByGroup = (groupName) =>
  get(`/api/v1/settings/group/${groupName}`)

/**
 * 获取单个设置
 * @param {string} key - e.g. 'retrieval.rerank.enabled'
 * @returns {Promise<SettingOut>}
 */
export const getSetting = (key) =>
  get(`/api/v1/settings/key/${key}`)

/**
 * 修改单个设置 (即时生效,不需要重启)
 * @param {string} key    - e.g. 'retrieval.vector.top_k'
 * @param {any} value     - 新值 (类型必须匹配 value_type)
 * @returns {Promise<SettingOut>}
 *
 * @example
 * await updateSetting('retrieval.rerank.enabled', true)
 * await updateSetting('retrieval.vector.top_k', 30)
 * await updateSetting('answering.generator.model', 'openai/gpt-4o')
 */
export const updateSetting = (key, value) =>
  put(`/api/v1/settings/key/${key}`, { value_json: value })

/**
 * 批量修改多个设置
 * @param {Array<{key: string, value_json: any}>} settings
 * @returns {Promise<SettingOut[]>}
 *
 * @example
 * await batchUpdateSettings([
 *   { key: 'retrieval.vector.enabled', value_json: true },
 *   { key: 'retrieval.bm25.enabled', value_json: false },
 * ])
 */
export const batchUpdateSettings = (settings) =>
  put('/api/v1/settings', { settings })

/**
 * 重置单个设置为 yaml 默认值
 * @param {string} key
 * @returns {Promise<{ reset: string }>}
 */
export const resetSetting = (key) =>
  del(`/api/v1/settings/key/${key}`)

/**
 * 重置整组为 yaml 默认值
 * @param {string} groupName
 * @returns {Promise<{ group: string, reset: number, reseeded: number }>}
 */
export const resetGroup = (groupName) =>
  post(`/api/v1/settings/reset-group/${groupName}`)

/**
 * 重置全部设置为 yaml 默认值
 * @returns {Promise<{ reset: number, reseeded: number }>}
 */
export const resetAllSettings = () =>
  post('/api/v1/settings/reset-all')

/**
 * 强制重新应用所有 DB 覆盖到 live config
 * @returns {Promise<{ applied: number }>}
 */
export const applySettings = () =>
  post('/api/v1/settings/apply')
