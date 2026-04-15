<script setup>
import { ref, provide, onMounted, onUnmounted } from 'vue'
import { RouterView } from 'vue-router'
import { listConversations, deleteConversation, getBenchmarkStatus } from '@/api'
import AppSidebar from '@/components/AppSidebar.vue'

const convs = ref([])
const convId = ref(null)
const benchmarkRunning = ref(false)

async function loadConvs() {
  try { convs.value = (await listConversations({ limit: 100 })).items || [] } catch {}
}
loadConvs()

/* Poll benchmark status to lock/unlock tabs */
let _bmPoll = null
async function pollBenchmark() {
  try {
    const s = await getBenchmarkStatus()
    benchmarkRunning.value = ['generating', 'running', 'scoring'].includes(s?.status)
  } catch { benchmarkRunning.value = false }
}
onMounted(() => {
  pollBenchmark()
  _bmPoll = setInterval(pollBenchmark, 3000)
})
onUnmounted(() => { if (_bmPoll) clearInterval(_bmPoll) })

provide('benchmarkRunning', benchmarkRunning)

function selectConv(id) { convId.value = id }
function newChat() { convId.value = null }
async function delConv(id) {
  try { await deleteConversation(id) } catch {}
  if (convId.value === id) convId.value = null
  loadConvs()
}

provide('convId', convId)
provide('convs', convs)
provide('loadConvs', loadConvs)
</script>

<template>
  <div class="flex w-full h-screen">
    <AppSidebar
      :conversations="convs"
      :currentConvId="convId"
      :benchmarkRunning="benchmarkRunning"
      @select-conv="selectConv"
      @new-chat="newChat"
      @delete-conv="delConv"
    />
    <!-- Content: scrollbar stays at window edge, inner content offset for visual centering -->
    <div class="flex-1 min-w-0 h-full">
      <RouterView />
    </div>
  </div>
</template>
