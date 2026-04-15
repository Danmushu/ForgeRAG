<template>
  <span class="spinner" :class="sizeClass" />
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  size: { type: String, default: 'md', validator: v => ['xs','sm','md','lg'].includes(v) },
})

const sizeClass = computed(() => `s-${props.size}`)
</script>

<style scoped>
.spinner {
  display: inline-block;
  border-radius: 9999px;
  border-style: solid;
  border-color: color-mix(in srgb, var(--color-t3, #999) 25%, transparent);
  border-top-color: var(--color-t2, #666);
  animation: spin 0.6s linear infinite;
}
.s-xs  { width: 8px;  height: 8px;  border-width: 1px; }
.s-sm  { width: 10px; height: 10px; border-width: 1.5px; }
.s-md  { width: 14px; height: 14px; border-width: 1.5px; }
.s-lg  { width: 20px; height: 20px; border-width: 2px; }

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
