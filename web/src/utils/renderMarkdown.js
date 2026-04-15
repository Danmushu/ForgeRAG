/**
 * Render Markdown + LaTeX to HTML.
 *
 * - Markdown: via `marked` (tables, code blocks, lists, bold, etc.)
 * - LaTeX:    via `katex` (inline $...$ and display $$...$$)
 *
 * LaTeX is processed BEFORE markdown so that `$...$` expressions
 * don't get mangled by the markdown parser.
 */

import { marked } from 'marked'
import katex from 'katex'
import 'katex/dist/katex.min.css'

// Configure marked for safe output
marked.setOptions({
  breaks: true,       // GFM line breaks
  gfm: true,          // GitHub Flavored Markdown
})

/**
 * Render LaTeX expressions in text.
 * Handles both inline ($...$) and display ($$...$$) math.
 */
function renderLatex(text) {
  // Display math: $$...$$  (must be processed first)
  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, expr) => {
    try {
      return katex.renderToString(expr.trim(), { displayMode: true, throwOnError: false })
    } catch {
      return `$$${expr}$$`
    }
  })

  // Inline math: $...$  (but not $$)
  // Negative lookbehind for $ and lookahead for $ to avoid matching $$
  text = text.replace(/(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g, (_, expr) => {
    try {
      return katex.renderToString(expr.trim(), { displayMode: false, throwOnError: false })
    } catch {
      return `$${expr}$`
    }
  })

  return text
}

/**
 * Render a string with Markdown + LaTeX support.
 * Returns an HTML string safe for v-html.
 *
 * @param {string} text - raw text (may contain markdown + latex)
 * @returns {string} HTML string
 */
export function renderMarkdown(text) {
  if (!text) return ''

  // 1. Protect code blocks from LaTeX processing
  const codeBlocks = []
  let processed = text.replace(/```[\s\S]*?```|`[^`]+`/g, (match) => {
    codeBlocks.push(match)
    return `__CODE_BLOCK_${codeBlocks.length - 1}__`
  })

  // 2. Render LaTeX
  processed = renderLatex(processed)

  // 3. Restore code blocks
  processed = processed.replace(/__CODE_BLOCK_(\d+)__/g, (_, i) => codeBlocks[parseInt(i)])

  // 4. Render Markdown
  const html = marked.parse(processed)

  return html
}
