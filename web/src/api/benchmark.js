import { get, post } from './client'

export function startBenchmark(numQuestions = 30) {
  return post('/api/v1/benchmark/start', { num_questions: numQuestions })
}

export function cancelBenchmark() {
  return post('/api/v1/benchmark/cancel')
}

export function getBenchmarkStatus() {
  return get('/api/v1/benchmark/status')
}

export function downloadBenchmarkReport() {
  // Direct download — returns a blob
  const base = import.meta.env.VITE_API_BASE || ''
  return `${base}/api/v1/benchmark/report`
}
