<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { getStats, getRetrievalStatus, getAllSettings, updateSetting, uploadAndIngest, getInfrastructure, listLLMProviders, createLLMProvider, updateLLMProvider, deleteLLMProvider } from '@/api'
import { request } from '@/api/client'
import { TrashIcon, ClipboardDocumentIcon, ClipboardDocumentCheckIcon, ArrowPathIcon } from '@heroicons/vue/24/outline'
import Spinner from '@/components/Spinner.vue'

const stats = ref({ documents: 0, chunks: 0, files: 0, bm25_indexed: 0 })
const status = ref({})
const settings = ref({})
const infra = ref({ storage_mode: '', storage_root: '', relational_backend: '', relational_path: '', vector_backend: '', vector_detail: '', graph_backend: '', graph_detail: '' })
const restarting = ref(false)
async function restartServer() {
  restarting.value = true
  try {
    await request('/api/v1/system/restart', { method: 'POST' })
  } catch {}
  // Server will die — poll until it's back
  const poll = setInterval(async () => {
    try {
      const r = await fetch('/api/v1/health')
      if (r.ok) { clearInterval(poll); restarting.value = false; await load() }
    } catch {}
  }, 2000)
  // Safety timeout
  setTimeout(() => { clearInterval(poll); restarting.value = false }, 30000)
}
const tasks = ref([])
const pop = reactive({ node: null, x: 0, y: 0 })
const hover = reactive({ node: null })

/* ── LLM Providers ── */
const providers = ref([])
const showNewProvider = ref(false)
const newProv = reactive({ name: '', provider_type: 'chat', api_base: '', model_name: '', api_key: '' })

async function loadProviders() {
  try { providers.value = await listLLMProviders() } catch {}
}
function providersByType(type) {
  return providers.value.filter(p => p.provider_type === type)
}
function providerName(id) {
  const p = providers.value.find(x => x.id === id)
  return p ? p.name : id || '—'
}
/* which provider type does a setting key need? */
function providerTypeForKey(key) {
  if (key.includes('embedder') || key.includes('embedding')) return 'embedding'
  if (key.includes('rerank')) return 'reranker'
  if (key.includes('image_enrichment')) return 'vlm'
  return 'chat'
}
function providersForKey(key) {
  return providersByType(providerTypeForKey(key))
}
async function doCreateProvider() {
  if (!newProv.name || !newProv.model_name) return
  try {
    await createLLMProvider({
      name: newProv.name, providerType: newProv.provider_type,
      apiBase: newProv.api_base || null, modelName: newProv.model_name,
      apiKey: newProv.api_key || null,
    })
    newProv.name = ''; newProv.provider_type = 'chat'; newProv.api_base = ''; newProv.model_name = ''; newProv.api_key = ''
    showNewProvider.value = false
    await loadProviders()
  } catch (e) { console.error('doCreateProvider failed:', e) }
}
/* ── Edit provider popover ── */
const editProv = reactive({ show: false, id: null, name: '', provider_type: 'chat', api_base: '', model_name: '', api_key: '', x: 0, y: 0 })
function openEditProvider(p, e) {
  e.stopPropagation()
  if (editProv.show && editProv.id === p.id) { editProv.show = false; return }
  const r = e.currentTarget.getBoundingClientRect()
  editProv.id = p.id; editProv.name = p.name; editProv.provider_type = p.provider_type
  editProv.model_name = p.model_name; editProv.api_base = p.api_base || ''; editProv.api_key = ''
  editProv.x = r.right + 8; editProv.y = Math.min(r.top, window.innerHeight - 360)
  editProv.show = true
}
async function doSaveProvider() {
  if (!editProv.name || !editProv.model_name) return
  try {
    await updateLLMProvider(editProv.id, {
      name: editProv.name, providerType: editProv.provider_type,
      modelName: editProv.model_name, apiBase: editProv.api_base || null,
      ...(editProv.api_key ? { apiKey: editProv.api_key } : {}),
    })
    editProv.show = false; await loadProviders()
  } catch (e) { console.error('doSaveProvider failed:', e) }
}
async function doDeleteProvider(id) {
  try {
    await deleteLLMProvider(id)
    editProv.show = false; await loadProviders()
  } catch (e) { console.error('doDeleteProvider failed:', e) }
}

/* ── module descriptions ── */
const moduleDesc = {
  filestore: {
    title: 'Storage & Cache',
    desc: 'Content-addressed blob storage for uploaded documents and extracted figures. Supports local filesystem, Amazon S3, and Alibaba OSS. Files are deduplicated by SHA-256 content hash. Also manages on-disk caches for BM25 index and embedding vectors to speed up restarts.',
    tech: 'SHA-256 dedup · Local / S3 / OSS backends · BM25 pickle cache · embedding MD5 cache',
  },
  parser: {
    title: 'Document Parser',
    desc: 'Multi-format document parsing with automatic backend routing. PyMuPDF for fast text extraction, MinerU for layout-aware parsing (tables, formulas, complex layouts), VLM for scanned or visually complex documents. Supports PDF, DOCX, PPTX, XLSX, HTML, Markdown, and TXT.',
    tech: 'PyMuPDF (fast) · MinerU (layout-aware, table/formula) · VLM image enrichment · block-level bbox preservation',
  },
  tree_builder: {
    title: 'Tree Builder',
    desc: 'Builds a document\'s hierarchical structure using LLM-based page-group analysis. Pages are grouped into windows, and an LLM infers logical section boundaries, titles, and summaries in a single call. TOC and heading signals are passed as hints but the LLM makes all structural decisions. The resulting tree with per-node summaries powers PageIndex-style tree navigation during retrieval.',
    tech: 'Page-group LLM inference · structural hints from TOC/headings · per-node summary generation · large-node subdivision · quality scoring',
  },
  chunker: {
    title: 'Chunker',
    desc: 'Tree-aware chunk generation that respects document structure. Walks the tree in preorder, packing blocks into chunks within section boundaries. Tables, figures, and formulas can be isolated into standalone chunks. Cross-references are resolved for later expansion during retrieval.',
    tech: 'Token-based greedy packing · configurable target/max/min tokens · table/figure/formula isolation · cross-reference resolution',
  },
  embedding: {
    title: 'Embedder',
    desc: 'Generates dense vector representations for semantic search. Uses LiteLLM for unified access to any embedding provider (OpenAI, Cohere, local models, etc.). Includes an on-disk cache keyed by content hash to avoid redundant API calls on re-ingestion.',
    tech: 'LiteLLM unified interface · any provider via LiteLLM · dimension validation · batch embedding · MD5-keyed disk cache',
  },
  database: {
    title: 'Persistence',
    desc: 'Unified persistence layer for documents, chunks, tree structures, conversations, traces, and runtime settings. Settings stored in DB override YAML config and take effect immediately — no restart needed. Supports incremental operations: versioned documents, incremental BM25 updates, per-document vector upsert/delete.',
    tech: 'SQLAlchemy 2.0 · SQLite / PostgreSQL / MySQL · ChromaDB / pgvector · NetworkX / Neo4j · hot-reload config · Alembic migrations',
  },
  qu: {
    title: 'Query Understanding',
    desc: 'A single LLM call that performs intent classification (factual, comparison, summary, greeting, etc.), retrieval routing (skip retrieval for greetings, skip expensive paths for simple lookups), and query expansion (synonym/translation variants for broader recall).',
    tech: 'Intent classification · retrieval path routing · query expansion · skip_paths · direct_answer for non-retrieval intents',
  },
  vector: {
    title: 'Vector Search',
    desc: 'Semantic similarity retrieval using dense embeddings. Finds chunks whose meaning is closest to the query regardless of exact keyword overlap — the primary recall path for conceptual and paraphrased queries.',
    tech: 'Cosine similarity · top-k retrieval · ChromaDB / pgvector backend · batch query embedding',
  },
  bm25: {
    title: 'BM25 Search',
    desc: 'Sparse keyword retrieval using the BM25 ranking function. Excels at exact-match and terminology-heavy queries (e.g. formula numbers, proper nouns). Index is persisted to disk and updated incrementally as documents are ingested.',
    tech: 'BM25 ranking · pickle-persisted index · incremental updates · deduplication · expanded query support',
  },
  tree: {
    title: 'Tree Navigation',
    desc: 'PageIndex-style LLM reasoning over document hierarchy. BM25 and vector search first identify candidate documents and "hot" regions, which are annotated onto the tree outline. The LLM then verifies which hot regions are truly relevant and identifies additional sections that may contain answers. Returns nodes with relevance scores.',
    tech: 'Heat-map annotated outline · verify + expand paradigm · LLM relevance scoring · parallel per-document · early-stop',
  },
  fusion: {
    title: 'RRF Fusion',
    desc: 'Combines results from tree navigation and knowledge graph paths using Reciprocal Rank Fusion. BM25 and vector serve as pre-filters for tree navigation; when tree navigation is unavailable, they enter RRF as fallback.',
    tech: 'Reciprocal Rank Fusion · configurable k constant · tree + KG primary paths · BM25/vector fallback',
  },
  expansion: {
    title: 'Context Expansion',
    desc: 'Enriches retrieved chunks with surrounding context from the document tree. Descendant expansion pulls child content for title-level hits (PageIndex-style). Sibling expansion adds neighboring sections. Cross-reference expansion follows document links.',
    tech: 'Descendant (PageIndex-style) · sibling · cross-reference · tree-based navigation · configurable toggles & discounts',
  },
  rerank: {
    title: 'Rerank',
    desc: 'LLM-powered re-scoring of candidate chunks after fusion and expansion. Reads each chunk against the query and assigns a fine-grained relevance score, promoting the most useful passages to the top of the final context window.',
    tech: 'LLM cross-encoder scoring · applied after fusion + expansion · configurable model · top-k selection for generator',
  },
  kg: {
    title: 'Knowledge Graph',
    desc: 'Multi-hop entity-relation reasoning for questions that no keyword or vector search can answer (e.g. "Which suppliers of Apple also supply Samsung?"). Extracts entities from the query, then traverses the graph through multiple hops. Dual-level retrieval: local (direct neighbors) and global (keyword entity search) are weighted and fused.',
    tech: 'LLM entity extraction · NetworkX / Neo4j · local + global dual retrieval · multi-hop BFS · configurable hop depth + weights',
  },
  kg_extraction: {
    title: 'KG Extraction',
    desc: 'LLM-powered entity and relation extraction during ingestion. Extracts structured entities (people, organizations, concepts) and typed relationships from each chunk in parallel. Entities are deduplicated and merged across chunks, building a document-spanning knowledge graph for multi-hop reasoning.',
    tech: 'LLM structured extraction · parallel chunk processing · entity/relation deduplication · weight accumulation · provenance tracking',
  },
  generator: {
    title: 'Generator',
    desc: 'Produces grounded answers by feeding top-ranked chunks as context to an LLM. Supports SSE streaming for real-time token delivery and multi-turn conversations with context-aware follow-up. Generates pixel-precise bbox citations pointing back to original document locations.',
    tech: 'LiteLLM (any provider) · SSE streaming · multi-turn context · bbox citations · hot-configurable model/temperature/prompts',
  },
  answer: {
    title: 'Answer + Bbox Citations',
    desc: 'The pipeline output: a natural-language answer with inline citations. Each citation carries original PDF bounding-box coordinates, section path, and file reference — enabling pixel-precise highlight-on-click in the built-in PDF viewer.',
    tech: 'Inline [c_N] citation tags · PDF bbox coordinates · section path · file reference · highlight-on-click',
  },
}

onMounted(load)
async function load() {
  try { stats.value = await getStats() } catch {}
  try { status.value = await getRetrievalStatus() } catch {}
  try { settings.value = (await getAllSettings()).groups || {} } catch {}
  try { infra.value = await getInfrastructure() } catch {}
  await loadProviders()
}

function gv(key) {
  for (const items of Object.values(settings.value)) {
    const f = items.find(s => s.key === key); if (f) return f.value_json
  }
  return null
}

async function toggle(key) {
  const map = { vector: 'retrieval.vector.enabled', bm25: 'retrieval.bm25.enabled', tree: 'retrieval.tree_path.enabled', rerank: 'retrieval.rerank.enabled', qu: 'retrieval.query_understanding.enabled', kg: 'retrieval.kg_path.enabled', kg_extraction: 'retrieval.kg_extraction.enabled' }
  if (!map[key]) return
  try { await updateSetting(map[key], !gv(map[key])); await load() }
  catch (e) { console.error('toggle failed:', e); load() }
}
function isOn(key) {
  const map = { vector: 'vector_enabled', bm25: 'bm25_enabled', tree: 'tree_enabled', rerank: 'rerank_enabled', qu: 'query_understanding_enabled', kg: 'kg_enabled', kg_extraction: 'kg_extraction_enabled' }
  return status.value[map[key]] === true
}

const _RIGHT_NODES = new Set(['generator', 'answer', 'rerank'])
function openPop(node, e) {
  e.stopPropagation()
  if (pop.node === node) { pop.node = null; return }
  const r = e.currentTarget.getBoundingClientRect()
  const popW = 300
  if (_RIGHT_NODES.has(node)) {
    pop.x = r.left - popW - 8
  } else {
    pop.x = r.right + 8
  }
  pop.node = node; pop.y = Math.min(r.top, window.innerHeight - 500)
}

const groupMap = {
  filestore: ['cache'],
  parser: ['parser', 'images'],
  tree_builder: ['tree_builder'],
  chunker: ['chunker'],
  embedding: ['embedding'],
  database: [],
  qu: ['query_understanding', 'prompts_qu'],
  vector: ['retrieval_vector'], bm25: ['retrieval_bm25'], tree: ['retrieval_tree', 'prompts_tree'],
  kg: ['kg'], kg_extraction: ['kg_extraction'],
  fusion: ['retrieval_fusion'],
  expansion: ['context_expansion'],
  rerank: ['rerank', 'prompts_rerank'], generator: ['llm', 'prompts_gen'],
}
function popItems() {
  const gs = groupMap[pop.node] || [], out = []
  for (const g of gs) {
    if (!settings.value[g]) continue
    for (const s of settings.value[g]) out.push(s)
  }

  // tree_builder: hide provider_id + summary_max_workers when llm_enabled is off;
  // keep the three LLM-related items adjacent (llm_enabled → provider_id → summary_max_workers)
  if (pop.node === 'tree_builder') {
    const llmOn = out.find(s => s.key === 'parser.tree_builder.llm_enabled')?.value_json
    const llmKeys = new Set([
      'parser.tree_builder.llm_enabled',
      'parser.tree_builder.provider_id',
      'parser.tree_builder.summary_max_workers',
    ])
    const llmGroup = ['parser.tree_builder.llm_enabled', 'parser.tree_builder.provider_id', 'parser.tree_builder.summary_max_workers']
    let filtered = llmOn
      ? out
      : out.filter(s => s.key === 'parser.tree_builder.llm_enabled' || !llmKeys.has(s.key))
    // Sort: LLM group first (in order), then the rest with bools first
    const rest = filtered.filter(s => !llmKeys.has(s.key))
    rest.sort((a, b) => (a.value_type === 'bool' ? -1 : 0) - (b.value_type === 'bool' ? -1 : 0))
    const top = llmGroup.map(k => filtered.find(s => s.key === k)).filter(Boolean)
    return [...top, ...rest]
  }

  return out.sort((a, b) => (a.value_type === 'bool' ? -1 : 0) - (b.value_type === 'bool' ? -1 : 0))
}
function isParentDisabled(item) {
  if (item.key.endsWith('.enabled') && item.value_type === 'bool') return false
  const gs = groupMap[pop.node] || []
  for (const g of gs) {
    for (const s of (settings.value[g] || [])) {
      if (s.key.endsWith('.enabled') && s.value_type === 'bool' && !s.value_json) {
        const prefix = s.key.replace(/\.enabled$/, '.')
        if (item.key.startsWith(prefix) && item.key !== s.key) return true
      }
    }
  }
  return false
}
async function saveSetting(key, val) {
  try { await updateSetting(key, val); await load() }
  catch (e) { console.error('saveSetting failed:', e); load() }
}

const copiedKey = ref(null)
function copyPrompt(it) {
  const text = it.value_json || it.default_value || ''
  navigator.clipboard.writeText(text)
  copiedKey.value = it.key
  setTimeout(() => { if (copiedKey.value === it.key) copiedKey.value = null }, 1500)
}

const dragging = ref(false)
function onDragOver(e) { e.preventDefault(); dragging.value = true }
function onDragLeave(e) { if (!e.currentTarget.contains(e.relatedTarget)) dragging.value = false }
async function onDrop(e) {
  e.preventDefault(); dragging.value = false
  for (const file of Array.from(e.dataTransfer?.files || []).filter(f => /\.(pdf|docx|pptx|xlsx)$/i.test(f.name))) {
    const id = Date.now().toString(36); tasks.value.push({ id, name: file.name, s: 'run' })
    try { await uploadAndIngest(file); tasks.value = tasks.value.map(t => t.id === id ? { ...t, s: 'ok' } : t) }
    catch { tasks.value = tasks.value.map(t => t.id === id ? { ...t, s: 'err' } : t) }
  }
  await load(); setTimeout(() => { tasks.value = tasks.value.filter(t => t.s !== 'ok') }, 2000)
}
</script>

<template>
  <div class="h-full flex flex-col bg-bg relative"
    @dragover="onDragOver" @dragleave="onDragLeave" @drop="onDrop" @click="pop.node = null; editProv.show = false">

    <!-- Apply & Restart (fixed top-right) -->
    <div class="fixed top-3 right-4 z-20">
      <button @click="restartServer" :disabled="restarting"
        class="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[11px] font-medium transition-colors"
        :class="restarting
          ? 'bg-amber-100 text-amber-700 cursor-wait'
          : 'bg-brand/10 text-brand hover:bg-brand/20'">
        <ArrowPathIcon class="w-3.5 h-3.5" :class="restarting ? 'animate-spin' : ''" />
        {{ restarting ? 'Restarting...' : 'Apply & Restart' }}
      </button>
    </div>


    <!-- Main: horizontal flow layout -->
    <div class="flex-1 flex items-start overflow-x-auto pt-20 pb-6 gap-12 pr-12 justify-center">

      <!-- COL 0: LLM Providers -->
      <div class="shrink-0 w-40 flex flex-col gap-1.5">
        <div class="text-[9px] text-t2 pl-3 uppercase tracking-widest">LLM Providers</div>

        <div v-for="p in providers" :key="p.id"
          class="node-card group cursor-pointer"
          :class="{ 'node-card--hover': hover.node === 'prov_' + p.id, 'node-card--active': editProv.show && editProv.id === p.id }"
          @mouseenter="hover.node = 'prov_' + p.id" @mouseleave="hover.node = null"
          @click="openEditProvider(p, $event)">
          <div class="text-[10px] text-t1 truncate">{{ p.name }}</div>
          <div class="cfg" style="margin-top:2px">
            <div class="flex items-center gap-1">
              <span class="tag" :class="{ 'tag--on': p.provider_type === 'chat' }">{{ p.provider_type }}</span>
            </div>
            <div class="truncate">{{ p.model_name }}</div>
            <div v-if="p.api_base" class="truncate">{{ p.api_base }}</div>
            <div>key: {{ p.api_key_set ? 'set' : 'none' }}</div>
          </div>
        </div>

        <div v-if="!providers.length && !showNewProvider" class="text-[9px] text-t3 px-3 py-2">No providers yet</div>

        <!-- New provider form -->
        <div v-if="showNewProvider" class="node-card space-y-1.5" @click.stop>
          <input v-model="newProv.name" placeholder="Name" class="prov-input" />
          <select v-model="newProv.provider_type" class="prov-input">
            <option value="chat">chat</option>
            <option value="embedding">embedding</option>
            <option value="reranker">reranker</option>
            <option value="vlm">vlm</option>
          </select>
          <input v-model="newProv.model_name" placeholder="Model (e.g. openai/gpt-4o)" class="prov-input" />
          <input v-model="newProv.api_base" placeholder="API Base (optional)" class="prov-input" />
          <input v-model="newProv.api_key" type="password" placeholder="API Key (optional)" class="prov-input" />
          <div class="flex gap-1.5">
            <button @click="doCreateProvider" class="flex-1 text-[9px] py-1 rounded bg-brand text-white hover:opacity-80">Save</button>
            <button @click="showNewProvider = false" class="flex-1 text-[9px] py-1 rounded border border-line text-t3 hover:bg-bg2">Cancel</button>
          </div>
        </div>

        <button v-else @click="showNewProvider = true"
          class="text-[10px] text-t3 px-3 py-1.5 rounded-md border border-dashed border-line hover:bg-bg2 transition-colors text-left">
          + New provider
        </button>
      </div>

      <!-- COL 1: Ingestion -->
      <div class="shrink-0 w-44 flex flex-col gap-3">
        <div class="text-[9px] text-t2 pl-3 uppercase tracking-widest">Ingestion</div>

        <!-- File Store -->
        <div class="node-card" :class="{ 'node-card--hover': hover.node === 'filestore', 'node-card--active': pop.node === 'filestore' }"
          @click.stop="openPop('filestore', $event)"
          @mouseenter="hover.node = 'filestore'" @mouseleave="hover.node = null">
          <div class="text-[11px] text-t1">Storage & Cache</div>
          <div class="cfg">
            <div class="flex items-center gap-1.5"><span class="tag">{{ infra.storage_mode || 'local' }}</span> <span class="truncate">{{ infra.storage_root }}</span></div>
            <div>BM25 cache: {{ gv('cache.bm25_persistence') !== false ? 'on' : 'off' }}</div>
            <div>embedding cache: {{ gv('cache.embedding_cache') !== false ? 'on' : 'off' }}</div>
            <div>documents: {{ stats.documents }}</div>
            <div>files: {{ stats.files }}</div>
          </div>
          <div v-for="t in tasks" :key="t.id" class="text-[9px] text-t3 mt-1 flex items-center gap-1">
            <Spinner v-if="t.s==='run'" size="xs" />
            <span v-else-if="t.s==='ok'" class="text-green-600">done</span>
            <span v-else class="text-red-500">err</span>
            <span class="truncate max-w-24">{{ t.name }}</span>
          </div>
        </div>

        <!-- Parser -->
        <div class="node-card" :class="{ 'node-card--hover': hover.node === 'parser', 'node-card--active': pop.node === 'parser' }"
          @click.stop="openPop('parser', $event)"
          @mouseenter="hover.node = 'parser'" @mouseleave="hover.node = null">
          <div class="text-[11px] text-t1">Document Parser</div>
          <div class="cfg">
            <div class="flex items-center gap-1.5"><span class="tag">PyMuPDF</span><span v-if="gv('parser.backends.mineru.enabled')" class="tag tag--on">MinerU</span></div>
            <div v-if="gv('parser.backends.mineru.enabled')">MinerU engine: {{ gv('parser.backends.mineru.backend') || 'pipeline' }}</div>
            <div>VLM images: {{ gv('image_enrichment.enabled') ? 'on' : 'off' }}</div>
            <div v-if="gv('image_enrichment.enabled')" class="truncate">VLM: {{ providerName(gv('image_enrichment.provider_id')) }}</div>
          </div>
        </div>

        <!-- Tree Builder -->
        <div class="node-card" :class="{ 'node-card--hover': hover.node === 'tree_builder', 'node-card--active': pop.node === 'tree_builder' }"
          @click.stop="openPop('tree_builder', $event)"
          @mouseenter="hover.node = 'tree_builder'" @mouseleave="hover.node = null">
          <div class="text-[10px] text-t1">Tree Builder</div>
          <div class="cfg">
            <div>LLM tree: {{ gv('parser.tree_builder.llm_enabled') ? 'on' : 'off' }}</div>
            <div v-if="gv('parser.tree_builder.llm_enabled') && gv('parser.tree_builder.provider_id')" class="truncate">model: {{ providerName(gv('parser.tree_builder.provider_id')) }}</div>
            <div>max depth: {{ gv('parser.tree_builder.max_reasonable_depth') || 6 }}</div>
            <div>min coverage: {{ gv('parser.tree_builder.min_coverage') ?? 0.5 }}</div>
          </div>
        </div>

        <!-- Chunker -->
        <div class="node-card" :class="{ 'node-card--hover': hover.node === 'chunker', 'node-card--active': pop.node === 'chunker' }"
          @click.stop="openPop('chunker', $event)"
          @mouseenter="hover.node = 'chunker'" @mouseleave="hover.node = null">
          <div class="text-[10px] text-t1">Chunker</div>
          <div class="cfg">
            <div>target: {{ gv('parser.chunker.target_tokens') || 600 }} tokens</div>
            <div>max: {{ gv('parser.chunker.max_tokens') || 1000 }} tokens</div>
            <div>min: {{ gv('parser.chunker.min_tokens') || 50 }} tokens</div>
            <div>isolate tables: {{ gv('parser.chunker.isolate_tables') !== false ? 'on' : 'off' }}</div>
            <div>isolate figures: {{ gv('parser.chunker.isolate_figures') !== false ? 'on' : 'off' }}</div>
            <div>isolate formulas: {{ gv('parser.chunker.isolate_formulas') ? 'on' : 'off' }}</div>
            <div>overlap: {{ gv('parser.chunker.overlap_blocks') || 0 }} blocks</div>
          </div>
        </div>

        <!-- Embedder -->
        <div class="node-card" :class="{ 'node-card--hover': hover.node === 'embedding', 'node-card--active': pop.node === 'embedding' }"
          @click.stop="openPop('embedding', $event)"
          @mouseenter="hover.node = 'embedding'" @mouseleave="hover.node = null">
          <div class="text-[10px] text-t1">Embedder</div>
          <div class="cfg">
            <div class="truncate">model: {{ providerName(gv('embedder.provider_id')) }}</div>
            <div>dimension: {{ gv('embedder.dimension') || '?' }}</div>
            <div>batch size: {{ gv('embedder.batch_size') || 32 }}</div>
          </div>
        </div>

        <!-- KG Extraction -->
        <div class="node-card"
          :class="[hover.node === 'kg_extraction' && 'node-card--hover', !isOn('kg_extraction') && 'opacity-50 node-card--off', pop.node === 'kg_extraction' && 'node-card--active']"
          @click.stop="openPop('kg_extraction', $event)"
          @mouseenter="hover.node = 'kg_extraction'" @mouseleave="hover.node = null">
          <div class="flex items-center justify-between">
            <div class="text-[10px]" :class="isOn('kg_extraction') ? 'text-t1' : 'text-t3'">KG Extraction</div>
            <button @click.stop="toggle('kg_extraction')" class="toggle shrink-0" :class="isOn('kg_extraction') ? 'bg-brand' : 'bg-gray-300'">
              <div class="toggle-dot" :style="{ transform: isOn('kg_extraction') ? 'translateX(13px)' : 'translateX(2px)' }"></div>
            </button>
          </div>
          <div class="cfg">
            <div>status: {{ isOn('kg_extraction') ? 'on' : 'off' }}</div>
            <div>workers: {{ gv('retrieval.kg_extraction.max_workers') || 5 }}</div>
            <div v-if="gv('retrieval.kg_extraction.provider_id')" class="truncate">model: {{ providerName(gv('retrieval.kg_extraction.provider_id')) }}</div>
          </div>
        </div>

        <!-- Database -->
        <div class="node-card" :class="{ 'node-card--hover': hover.node === 'database', 'node-card--active': pop.node === 'database' }"
          @click.stop="openPop('database', $event)"
          @mouseenter="hover.node = 'database'" @mouseleave="hover.node = null">
          <div class="text-[10px] text-t1">Persistence</div>
          <div class="cfg">
            <div class="flex items-center gap-1.5"><span class="tag">{{ infra.relational_backend || 'sqlite' }}</span> <span class="truncate">{{ infra.relational_path }}</span></div>
            <div class="flex items-center gap-1.5"><span class="tag">{{ infra.vector_backend || 'chromadb' }}</span> <span class="truncate">{{ infra.vector_detail }}</span></div>
            <div v-if="infra.graph_backend" class="flex items-center gap-1.5"><span class="tag" :class="{ 'tag--on': isOn('kg') }">{{ infra.graph_backend }}</span> <span class="truncate">{{ infra.graph_detail }}</span></div>
            <div>chunks: {{ stats.chunks }}</div>
            <div>BM25 indexed: {{ stats.bm25_indexed || 0 }}</div>
          </div>
        </div>
      </div>

      <!-- COL 2: Retrieval paths -->
      <div class="shrink-0 w-44 flex flex-col gap-2">
        <div class="text-[9px] text-t2 pl-3 uppercase tracking-widest">Retrieval</div>

        <!-- Query Understanding -->
        <div class="node-card"
          :class="[hover.node === 'qu' && 'node-card--hover', !isOn('qu') && 'opacity-50 node-card--off', pop.node === 'qu' && 'node-card--active']"
          @click.stop="openPop('qu', $event)"
          @mouseenter="hover.node = 'qu'" @mouseleave="hover.node = null">
          <div class="flex items-center justify-between">
            <div class="text-[10px] text-t1">Query Understanding</div>
            <button @click.stop="toggle('qu')" class="toggle shrink-0" :class="isOn('qu') ? 'bg-brand' : 'bg-gray-300'">
              <div class="toggle-dot" :style="{ transform: isOn('qu') ? 'translateX(13px)' : 'translateX(2px)' }"></div>
            </button>
          </div>
          <div class="cfg">
            <div>status: {{ isOn('qu') ? 'on' : 'off' }}</div>
            <div>max expansions: {{ gv('retrieval.query_understanding.max_expansions') || 3 }}</div>
            <div v-if="gv('retrieval.query_understanding.provider_id')" class="truncate">model: {{ providerName(gv('retrieval.query_understanding.provider_id')) }}</div>
          </div>
        </div>

        <!-- Vector -->
        <div class="node-card"
          :class="[hover.node === 'vector' && 'node-card--hover', !isOn('vector') && 'opacity-40 node-card--off', pop.node === 'vector' && 'node-card--active']"
          @click.stop="openPop('vector', $event)"
          @mouseenter="hover.node = 'vector'" @mouseleave="hover.node = null">
          <div class="flex items-center justify-between">
            <div class="text-[10px]" :class="isOn('vector') ? 'text-t1' : 'text-t3'">Vector</div>
            <button @click.stop="toggle('vector')" class="toggle shrink-0" :class="isOn('vector') ? 'bg-brand' : 'bg-gray-300'">
              <div class="toggle-dot" :style="{ transform: isOn('vector') ? 'translateX(13px)' : 'translateX(2px)' }"></div>
            </button>
          </div>
          <div class="cfg">
            <div>type: semantic similarity</div>
            <div>top-k: {{ gv('retrieval.vector.top_k') || 20 }}</div>
          </div>
        </div>

        <!-- BM25 -->
        <div class="node-card"
          :class="[hover.node === 'bm25' && 'node-card--hover', !isOn('bm25') && 'opacity-40 node-card--off', pop.node === 'bm25' && 'node-card--active']"
          @click.stop="openPop('bm25', $event)"
          @mouseenter="hover.node = 'bm25'" @mouseleave="hover.node = null">
          <div class="flex items-center justify-between">
            <div class="text-[10px]" :class="isOn('bm25') ? 'text-t1' : 'text-t3'">BM25</div>
            <button @click.stop="toggle('bm25')" class="toggle shrink-0" :class="isOn('bm25') ? 'bg-brand' : 'bg-gray-300'">
              <div class="toggle-dot" :style="{ transform: isOn('bm25') ? 'translateX(13px)' : 'translateX(2px)' }"></div>
            </button>
          </div>
          <div class="cfg">
            <div>type: keyword matching</div>
            <div>top-k: {{ gv('retrieval.bm25.top_k') || 20 }}</div>
            <div>k1: {{ gv('retrieval.bm25.k1') ?? 1.5 }}</div>
            <div>b: {{ gv('retrieval.bm25.b') ?? 0.75 }}</div>
            <div>doc prefilter: {{ gv('retrieval.bm25.doc_prefilter_top_k') || 5 }}</div>
          </div>
        </div>

        <!-- Tree Nav -->
        <div class="node-card"
          :class="[hover.node === 'tree' && 'node-card--hover', !isOn('tree') && 'opacity-40 node-card--off', pop.node === 'tree' && 'node-card--active']"
          @click.stop="openPop('tree', $event)"
          @mouseenter="hover.node = 'tree'" @mouseleave="hover.node = null">
          <div class="flex items-center justify-between">
            <div class="text-[10px]" :class="isOn('tree') ? 'text-t1' : 'text-t3'">Tree Navigation</div>
            <button @click.stop="toggle('tree')" class="toggle shrink-0" :class="isOn('tree') ? 'bg-brand' : 'bg-gray-300'">
              <div class="toggle-dot" :style="{ transform: isOn('tree') ? 'translateX(13px)' : 'translateX(2px)' }"></div>
            </button>
          </div>
          <div class="cfg">
            <div>type: structure traversal</div>
            <div>LLM nav: {{ gv('retrieval.tree_path.llm_nav_enabled') !== false ? 'on' : 'off' }}</div>
            <div>top-k: {{ gv('retrieval.tree_path.top_k') || 10 }}</div>
            <div v-if="gv('retrieval.tree_path.nav.provider_id')" class="truncate">model: {{ providerName(gv('retrieval.tree_path.nav.provider_id')) }}</div>
            <div>max nodes: {{ gv('retrieval.tree_path.nav.max_nodes') || 3 }}</div>
            <div>workers: {{ gv('retrieval.tree_path.nav.max_workers') || 4 }}</div>
            <div>target chunks: {{ gv('retrieval.tree_path.nav.target_chunks') || 15 }}</div>
          </div>
        </div>

        <!-- Knowledge Graph -->
        <div class="node-card"
          :class="[hover.node === 'kg' && 'node-card--hover', !isOn('kg') && 'opacity-40 node-card--off', pop.node === 'kg' && 'node-card--active']"
          @click.stop="openPop('kg', $event)"
          @mouseenter="hover.node = 'kg'" @mouseleave="hover.node = null">
          <div class="flex items-center justify-between">
            <div class="text-[10px]" :class="isOn('kg') ? 'text-t1' : 'text-t3'">Knowledge Graph</div>
            <button @click.stop="toggle('kg')" class="toggle shrink-0" :class="isOn('kg') ? 'bg-brand' : 'bg-gray-300'">
              <div class="toggle-dot" :style="{ transform: isOn('kg') ? 'translateX(13px)' : 'translateX(2px)' }"></div>
            </button>
          </div>
          <div class="cfg">
            <div>type: multi-hop traversal</div>
            <div>top-k: {{ gv('retrieval.kg_path.top_k') || 30 }}</div>
            <div>max hops: {{ gv('retrieval.kg_path.max_hops') || 2 }}</div>
            <div>local: {{ gv('retrieval.kg_path.local_weight') ?? 0.7 }} · global: {{ gv('retrieval.kg_path.global_weight') ?? 0.3 }}</div>
            <div v-if="gv('retrieval.kg_path.provider_id')" class="truncate">model: {{ providerName(gv('retrieval.kg_path.provider_id')) }}</div>
          </div>
        </div>
      </div>

      <!-- COL 3: Aggregate -->
      <div class="shrink-0 w-44 flex flex-col gap-2">
        <div class="text-[9px] text-t2 pl-3 uppercase tracking-widest">Aggregate</div>

        <!-- RRF Fusion -->
        <div class="node-card" :class="{ 'node-card--hover': hover.node === 'fusion', 'node-card--active': pop.node === 'fusion' }"
          @click.stop="openPop('fusion', $event)"
          @mouseenter="hover.node = 'fusion'" @mouseleave="hover.node = null">
          <div class="text-[11px] text-t1">RRF Fusion</div>
          <div class="cfg">
            <div>k: {{ gv('retrieval.merge.rrf_k') || 60 }}</div>
            <div>candidate limit: {{ gv('retrieval.merge.candidate_limit') || 50 }}</div>
            <div>budget multiplier: {{ gv('retrieval.merge.global_budget_multiplier') ?? 2.0 }}x</div>
          </div>
        </div>

        <!-- Context Expansion -->
        <div class="node-card" :class="{ 'node-card--hover': hover.node === 'expansion', 'node-card--active': pop.node === 'expansion' }"
          @click.stop="openPop('expansion', $event)"
          @mouseenter="hover.node = 'expansion'" @mouseleave="hover.node = null">
          <div class="text-[10px] text-t1">Context Expansion</div>
          <div class="cfg">
            <div class="flex items-center justify-between">
              <span>descendant</span>
              <span :class="status.descendant_expansion_enabled ? 'text-brand' : ''">{{ status.descendant_expansion_enabled ? 'on' : 'off' }}</span>
            </div>
            <div v-if="status.descendant_expansion_enabled" class="pl-2">max chunks: {{ gv('retrieval.merge.descendant_max_chunks') || 5 }}</div>
            <div v-if="status.descendant_expansion_enabled" class="pl-2">discount: {{ gv('retrieval.merge.descendant_score_discount') ?? 0.8 }}</div>
            <div class="flex items-center justify-between">
              <span>sibling</span>
              <span :class="status.sibling_expansion_enabled ? 'text-brand' : ''">{{ status.sibling_expansion_enabled ? 'on' : 'off' }}</span>
            </div>
            <div v-if="status.sibling_expansion_enabled" class="pl-2">max node size: {{ gv('retrieval.merge.sibling_max_node_size') || 3 }}</div>
            <div v-if="status.sibling_expansion_enabled" class="pl-2">discount: {{ gv('retrieval.merge.sibling_score_discount') ?? 0.6 }}</div>
            <div class="flex items-center justify-between">
              <span>cross-ref</span>
              <span :class="status.crossref_expansion_enabled ? 'text-brand' : ''">{{ status.crossref_expansion_enabled ? 'on' : 'off' }}</span>
            </div>
            <div v-if="status.crossref_expansion_enabled" class="pl-2">discount: {{ gv('retrieval.merge.crossref_score_discount') ?? 0.5 }}</div>
          </div>
        </div>

        <!-- Rerank -->
        <div class="node-card"
          :class="[hover.node === 'rerank' && 'node-card--hover', !isOn('rerank') && 'opacity-50 node-card--off', pop.node === 'rerank' && 'node-card--active']"
          @click.stop="openPop('rerank', $event)"
          @mouseenter="hover.node = 'rerank'" @mouseleave="hover.node = null">
          <div class="flex items-center justify-between">
            <div class="text-[10px] text-t1">Rerank</div>
            <button @click.stop="toggle('rerank')" class="toggle shrink-0" :class="isOn('rerank') ? 'bg-brand' : 'bg-gray-300'">
              <div class="toggle-dot" :style="{ transform: isOn('rerank') ? 'translateX(13px)' : 'translateX(2px)' }"></div>
            </button>
          </div>
          <div class="cfg">
            <div>backend: {{ isOn('rerank') ? gv('retrieval.rerank.backend') || 'litellm' : 'passthrough' }}</div>
            <div>top-k: {{ gv('retrieval.rerank.top_k') || 10 }}</div>
            <div v-if="isOn('rerank') && gv('retrieval.rerank.provider_id')" class="truncate">model: {{ providerName(gv('retrieval.rerank.provider_id')) }}</div>
          </div>
        </div>
      </div>

      <!-- COL 4: Generation -->
      <div class="shrink-0 w-40 flex flex-col gap-3">
        <div class="text-[9px] text-t2 pl-3 uppercase tracking-widest">Generation</div>

        <!-- Generator -->
        <div class="node-card" :class="{ 'node-card--hover': hover.node === 'generator', 'node-card--active': pop.node === 'generator' }"
          @click.stop="openPop('generator', $event)"
          @mouseenter="hover.node = 'generator'" @mouseleave="hover.node = null">
          <div class="text-[11px] text-t1">Generator</div>
          <div class="cfg">
            <div class="truncate">model: {{ providerName(gv('answering.generator.provider_id')) }}</div>
            <div>temperature: {{ gv('answering.generator.temperature') ?? 0.1 }}</div>
            <div>max tokens: {{ gv('answering.generator.max_tokens') || 2048 }}</div>
            <div>context chunks: {{ gv('answering.max_chunks') || 10 }}</div>
          </div>
        </div>

        <!-- Answer -->
        <div class="node-card" :class="{ 'node-card--hover': hover.node === 'answer', 'node-card--active': pop.node === 'answer' }"
          @mouseenter="hover.node = 'answer'" @mouseleave="hover.node = null"
          style="border: 1.5px solid var(--color-brand-bg);">
          <div class="text-[10px] text-brand font-medium">Answer + Bbox Citations</div>
          <div class="cfg">
            <div>bbox coordinates</div>
            <div>section path</div>
            <div>file reference</div>
          </div>
        </div>
      </div>

    </div>

    <!-- Popover (settings editor) -->
    <div v-if="pop.node" class="popover fadein" :style="{ left: pop.x + 'px', top: pop.y + 'px', width: '300px', maxHeight: '500px' }" @click.stop>
      <div class="flex items-center justify-between px-3 py-2 border-b border-line">
        <span class="text-xs text-t1 font-medium">{{ moduleDesc[pop.node]?.title || pop.node }}</span>
        <button @click="pop.node = null" class="text-xs text-t3 hover:text-t1">x</button>
      </div>
      <div class="px-3 py-3 overflow-y-auto space-y-2.5" style="max-height: 440px">
        <template v-if="pop.node === 'database'">
          <div class="space-y-2">
            <div>
              <div class="text-[10px] text-t3 mb-0.5">Relational</div>
              <div class="text-xs text-t1">{{ infra.relational_backend }}</div>
              <div class="text-[10px] text-t3">{{ infra.relational_path }}</div>
            </div>
            <div>
              <div class="text-[10px] text-t3 mb-0.5">Vector</div>
              <div class="text-xs text-t1">{{ infra.vector_backend }}</div>
              <div class="text-[10px] text-t3">{{ infra.vector_detail }}</div>
            </div>
            <div>
              <div class="text-[10px] text-t3 mb-0.5">Blob Storage</div>
              <div class="text-xs text-t1">{{ infra.storage_mode }}</div>
              <div class="text-[10px] text-t3">{{ infra.storage_root }}</div>
            </div>
            <div v-if="infra.graph_backend">
              <div class="text-[10px] text-t3 mb-0.5">Knowledge Graph</div>
              <div class="text-xs text-t1">{{ infra.graph_backend }}</div>
              <div class="text-[10px] text-t3">{{ infra.graph_detail }}</div>
            </div>
            <div class="text-[9px] text-t3 pt-1 border-t border-line">Configured in forgerag.yaml (restart required to change)</div>
          </div>
        </template>
        <template v-else>
        <div v-if="!popItems().length" class="text-xs py-2 text-center text-t3">No options</div>
        <template v-for="it in popItems()" :key="it.key">
          <div v-if="!isParentDisabled(it)">
            <div class="text-[11px] mb-1 text-t2">{{ it.label }}</div>
            <div v-if="it.value_type==='bool'" class="flex items-center gap-2">
              <button @click="it.value_json=!it.value_json; saveSetting(it.key, it.value_json)"
                class="toggle" :class="it.value_json ? 'bg-brand' : 'bg-gray-300'">
                <div class="toggle-dot" :style="{ transform: it.value_json ? 'translateX(13px)' : 'translateX(2px)' }"></div>
              </button>
              <span class="text-[10px] text-t3">{{ it.value_json ? 'on' : 'off' }}</span>
            </div>
            <input v-else-if="it.value_type==='int'||it.value_type==='float'" :value="it.value_json" type="number" :step="it.value_type==='float'?0.1:1"
              class="w-full px-2.5 py-1.5 rounded-md border border-line bg-bg text-xs text-t1 outline-none focus:border-brand transition-colors"
              @change="{ const v = it.value_type==='int' ? parseInt($event.target.value) : parseFloat($event.target.value); if (!isNaN(v)) saveSetting(it.key, v) }" />
            <select v-else-if="it.value_type==='enum'" :value="it.value_json"
              class="w-full px-2.5 py-1.5 rounded-md border border-line bg-bg text-xs text-t1 outline-none focus:border-brand"
              @change="saveSetting(it.key, $event.target.value)">
              <option v-for="o in (it.enum_options||[])" :key="o" :value="o">{{ o }}</option>
            </select>
            <!-- provider_id → select from LLM Providers -->
            <template v-else-if="it.key.endsWith('.provider_id')">
              <select :value="it.value_json || ''"
                class="w-full px-2.5 py-1.5 rounded-md border border-line bg-bg text-xs text-t1 outline-none focus:border-brand"
                @change="saveSetting(it.key, $event.target.value || null)">
                <option value="">— none —</option>
                <option v-for="p in providersForKey(it.key)" :key="p.id" :value="p.id">{{ p.name }} ({{ p.model_name }})</option>
              </select>
            </template>
            <div v-else-if="it.value_type==='textarea'" class="relative">
              <textarea :value="it.value_json||''" rows="6"
                class="w-full px-2.5 py-1.5 pr-8 rounded-md border border-line bg-bg text-xs text-t1 outline-none focus:border-brand font-mono leading-relaxed resize-y placeholder:text-t3/60 placeholder:font-sans"
                :placeholder="it.default_value || ''"
                @change="saveSetting(it.key, $event.target.value)" />
              <button @click="copyPrompt(it)"
                class="absolute top-1.5 right-1.5 p-1 rounded text-t3 hover:text-t1 hover:bg-bg3 transition-colors"
                :title="copiedKey===it.key ? 'Copied!' : 'Copy'">
                <ClipboardDocumentCheckIcon v-if="copiedKey===it.key" class="w-3.5 h-3.5 text-brand" />
                <ClipboardDocumentIcon v-else class="w-3.5 h-3.5" />
              </button>
            </div>
            <input v-else :value="it.value_json" type="text"
              class="w-full px-2.5 py-1.5 rounded-md border border-line bg-bg text-xs text-t1 outline-none focus:border-brand"
              @change="saveSetting(it.key, $event.target.value)" />
            <div v-if="it.description" class="text-[9px] mt-0.5 text-t3">{{ it.description }}</div>
          </div>
        </template>
        </template>
      </div>
    </div>

    <!-- Popover (edit LLM provider) -->
    <div v-if="editProv.show" class="popover fadein" :style="{ left: editProv.x + 'px', top: editProv.y + 'px', width: '260px' }" @click.stop>
      <div class="flex items-center justify-between px-3 py-2 border-b border-line">
        <span class="text-xs text-t1 font-medium">Edit Provider</span>
        <button @click="editProv.show = false" class="text-xs text-t3 hover:text-t1">x</button>
      </div>
      <div class="px-3 py-3 space-y-2">
        <div>
          <div class="text-[11px] mb-1 text-t2">Name</div>
          <input v-model="editProv.name" class="prov-input" />
        </div>
        <div>
          <div class="text-[11px] mb-1 text-t2">Type</div>
          <select v-model="editProv.provider_type" class="prov-input">
            <option value="chat">chat</option>
            <option value="embedding">embedding</option>
            <option value="reranker">reranker</option>
            <option value="vlm">vlm</option>
          </select>
        </div>
        <div>
          <div class="text-[11px] mb-1 text-t2">Model</div>
          <input v-model="editProv.model_name" placeholder="e.g. openai/gpt-4o" class="prov-input" />
        </div>
        <div>
          <div class="text-[11px] mb-1 text-t2">API Base</div>
          <input v-model="editProv.api_base" placeholder="optional" class="prov-input" />
        </div>
        <div>
          <div class="text-[11px] mb-1 text-t2">API Key</div>
          <input v-model="editProv.api_key" type="password" placeholder="leave empty to keep unchanged" class="prov-input" />
        </div>
        <div class="flex gap-1.5 pt-1">
          <button @click="doSaveProvider" class="flex-1 text-[9px] py-1.5 rounded bg-brand text-white hover:opacity-80">Save</button>
          <button @click="doDeleteProvider(editProv.id)"
            class="w-7 h-7 flex items-center justify-center rounded border border-line text-t3 transition-colors hover:bg-red-500 hover:border-red-500 hover:text-white">
            <TrashIcon class="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.node-card {
  padding: 10px 12px;
  border-radius: 6px;
  border: 1px solid var(--color-line);
  cursor: pointer;
  transition: background-color 0.15s;
}
.node-card--hover {
  background-color: var(--color-bg2);
}
.node-card--active {
  border-color: var(--color-brand);
  box-shadow: 0 0 0 1px var(--color-brand);
}
.node-card--off.node-card--hover {
  background-color: color-mix(in srgb, var(--color-bg2) 50%, transparent);
}
.cfg {
  margin-top: 6px;
  font-size: 9px;
  line-height: 1.7;
  color: var(--color-t3);
}
.tag {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 4px;
  background: var(--color-bg3);
  font-size: 9px;
  color: var(--color-t2);
}
.tag--on {
  background: var(--color-brand-bg);
  color: var(--color-brand);
}

.prov-input {
  width: 100%;
  padding: 4px 8px;
  border-radius: 4px;
  border: 1px solid var(--color-line);
  background: var(--color-bg);
  font-size: 10px;
  color: var(--color-t1);
  outline: none;
}
.prov-input:focus {
  border-color: var(--color-brand);
}

/* doc panel divider: follows content height, thinner than 1px */
.doc-divider {
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  width: 0.5px;
  height: 60%;
  background: var(--color-line2);
  opacity: 0.6;
}

/* description panel fade */
.desc-fade-enter-active { transition: opacity 0.2s ease-out; }
.desc-fade-leave-active { transition: opacity 0.1s ease-in; }
.desc-fade-enter-from { opacity: 0; }
.desc-fade-leave-to { opacity: 0; }
</style>
