<script setup>
const props = defineProps({
  node: Object,
  nodes: Object,
  depth: { type: Number, default: 0 },
  highlight: Set,
  filterNodeId: String,
  expanded: Object,
})
const emit = defineEmits(['toggle', 'select'])

function children() {
  if (!props.node?.children) return []
  return props.node.children.map(id => props.nodes[id]).filter(Boolean)
}

function isExpanded() {
  return props.expanded[props.node.node_id] !== false
}

function hasChildren() {
  return children().length > 0
}
</script>

<template>
  <div v-if="node">
    <div
      class="flex items-start gap-1 py-1 px-1.5 rounded cursor-pointer transition-colors hover:bg-bg2"
      :style="{ paddingLeft: depth * 16 + 6 + 'px' }"
      @click="emit('select', node.node_id)"
    >
      <!-- expand/collapse arrow -->
      <button
        v-if="hasChildren()"
        @click.stop="emit('toggle', node.node_id)"
        class="text-[10px] text-t3 w-3 shrink-0 mt-px select-none"
      >{{ isExpanded() ? '\u25BE' : '\u25B8' }}</button>
      <span v-else class="w-3 shrink-0"></span>

      <!-- content -->
      <div class="min-w-0 flex-1">
        <div class="text-[10px] truncate"
             :class="filterNodeId === node.node_id || highlight.has(node.node_id) ? 'text-t1 font-semibold' : 'text-t1'">
          {{ node.title || node.node_id }}
        </div>
        <div class="text-[8px]"
             :class="filterNodeId === node.node_id || highlight.has(node.node_id) ? 'text-t2 font-medium' : 'text-t3'">
          L{{ node.level }}
          <template v-if="node.page_start"> · p.{{ node.page_start }}{{ node.page_end && node.page_end !== node.page_start ? '-' + node.page_end : '' }}</template>
          <template v-if="node.table_count"> · {{ node.table_count }}T</template>
          <template v-if="node.figure_count"> · {{ node.figure_count }}F</template>
        </div>
      </div>
    </div>

    <!-- children -->
    <template v-if="hasChildren() && isExpanded()">
      <TreeNode
        v-for="child in children()" :key="child.node_id"
        :node="child"
        :nodes="nodes"
        :depth="depth + 1"
        :highlight="highlight"
        :filterNodeId="filterNodeId"
        :expanded="expanded"
        @toggle="emit('toggle', $event)"
        @select="emit('select', $event)"
      />
    </template>
  </div>
</template>
