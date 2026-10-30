# AgentForge landing page: design spec

Direction: a technical dossier. The page reads like a spec sheet from an engineering office:
monospace labels in uppercase with wide tracking, numbered sections, registration marks, hatched
fills, barcodes, a stripe logotype, cream "paper" panels inverted on a warm black ground, and one
signal orange. References: industrial label systems, routing tickets, brutalist engineering sites.

## 1. Tokens

| Token | Value | Use |
|---|---|---|
| `--bg` | `#0c0b09` | page ground (warm black) |
| `--bg-1` | `#15130f` | panels |
| `--bg-2` | `#1d1a15` | code, raised panels |
| `--line` / `--line-2` | `#2a261f` / `#3d382e` | hairlines, solid, never dashed |
| `--paper` | `#ede6d6` | primary ink on dark; fill of paper panels |
| `--paper-2` | `#cfc7b4` | body copy |
| `--mute` | `#8d8676` | labels, footnotes (5.4:1 on `--bg`) |
| `--accent` | `#ff4f1f` | signal orange: CTAs, ticker, index rules, chart emphasis |
| `--accent-2` / `--accent-dim` | `#ff7a45` / `#b8360f` | hover / pressed |
| `--on-paper` / `--on-paper-2` | `#0c0b09` / `#4a4538` | ink on paper panels |
| `--chart-base` | `#a8a08f` | the de-emphasis series (LRU baseline) |

Chart palette (`#ff4f1f`, `#a8a08f`) on `--bg-1`: CVD separation ΔE 11.6, normal-vision ΔE 20.8,
contrast ≥ 3:1 for both. The form is emphasis (one hue plus gray), so the gray sits below the
categorical chroma floor on purpose.

## 2. Type

- Display: Archivo Variable (self-hosted, `wdth` 62–125, `wght` 100–900). Headlines 800,
  tracking −0.02em, line-height 0.98. Wordmark 800 at `wdth` 118. Stat values 700.
- Mono: JetBrains Mono Variable. Labels 11.5px, 500, uppercase, tracking 0.12em. Data 13px,
  `tabular-nums` only in columns.
- Scale: H1 `clamp(42px, 6.4vw, 96px)`; H2 `clamp(30px, 4.2vw, 58px)`; H3 22px; body 17px/1.55;
  lede `clamp(17px, 1.4vw, 20px)`.

## 3. Layout

- Content width 1360px, side gutter `clamp(16px, 3vw, 40px)`, 12-column grid, 24px gutters.
- Every section opens with a head row: `0X / NAME` with a short accent rule on the left, a
  source note in mono on the right, then the H2.
- Sections are separated by full-width hairlines. A ruler strip (ticks every 8px, tall ticks
  every 80px) sits under the hero.
- Sticky top bar with a translucent ground and blur.

## 4. Primitives

- **Panel**: `--bg-1` fill, 1px `--line` border. **Paper panel**: `--paper` fill, dark ink.
- **Registration corners**: four 10px L-marks inset 6px, drawn with background gradients on an
  absolutely positioned overlay, so any panel can carry them.
- **Hatch**: `repeating-linear-gradient(45deg, accent 0 1px, transparent 1px 6px)`; used on the
  outline CTA, which fills solid on hover.
- **Barcode**: SVG of deterministic bar widths; paper or ink color.
- **Stripe mark**: the letters A and F built from five horizontal stripes (7 units high, 4 gap).
- **Chips**: mono 10.5px uppercase, 1px border; variants accent, fill, mute.
- **Ticker**: accent band, black uppercase mono, 60s linear marquee; static under
  `prefers-reduced-motion`.
- **Status dot**: 7px accent dot, 1.6s step blink; static under reduced motion.
- **Code panel**: `--bg-2`, mono 13px, prompt in accent, comments in `--mute`.

## 5. Sections

0. **Top bar**: mark + AGENTFORGE; two meta lines (what it is; status, language, license);
   nav Thesis · Architecture · Results · Machines · Roadmap; hatched "Source" CTA.
1. **Hero**: eyebrow `DOC 0001 / PROJECT THESIS / ACCEPTED 2026-10-30`; H1 "Compile the agent
   program and the model together."; lede from the README; primary CTA to the design doc,
   secondary to the research report; three micro facts. Right: a paper "dossier" ticket with the
   median coding-agent step as four stat tiles (119K / 875 / 214 / 95–96%), three fact rows, a
   barcode and a document id. Dot-grid ground fading downward.
2. **Ticker**: the vocabulary of the project, repeated.
3. **01 Problem**: three numbered cards (workloads, silicon, software) with source footers.
4. **02 Thesis**: a flow: Agent Program IR + Machine model → Compiler → residency, prefetch,
   placement, kernels. One-line analogy to register liveness.
5. **03 Architecture**: eight rows, L1–L8: name, crate path, role, status chip
   (baseline / skeleton / traits / planned). A strip of the seven crates.
6. **04 Results**: setup panel; paired-bar chart LRU vs plan (bars indexed to LRU = 100%,
   value at the tip, delta in muted ink, legend present, table view in a disclosure); three
   reading cards (holds / does not fix / loses); first-order arithmetic as a labeled bar table
   with recompute in accent and transfers in gray.
7. **05 Machines**: five tickets (paper and dark alternating) with class, memory, bandwidth,
   SRAM, FP8 rate, link, tiers, file name and barcode; an excerpt of `h200.toml`.
8. **06 Quickstart**: a terminal panel with the README commands and one-line comments.
9. **07 Roadmap**: M0–M6 rows with scope, done-when and status.
10. **Footer**: mark, one-line description, link columns (docs, project), target list, colophon.

## 6. Charts (dataviz rules applied)

- Bars 14px thick, 4px rounded data end, square at the baseline, 2px surface gap between
  paired bars, hairline axes, text in ink tokens never in the series color.
- Two series → legend always present; values at the bar tip; hover `title` on every bar; a
  table view with the full numbers behind a disclosure so nothing is gated by hover.
- One axis per chart. The comparison chart indexes each metric to its own LRU value, so
  metrics of different units share one layout honestly.

## 7. Responsive and accessibility

- Breakpoints: 1100px (meta lines hide), 900px (hero stacks, nav wraps under the brand),
  640px (cards single column, tickets single column, architecture rows stack).
- Landmarks: header, nav, main, labeled sections, footer; skip link; accent focus rings.
- Reduced motion disables the marquee, the blink and smooth scrolling.
- Minimum contrast 4.5:1 for all running text; 3:1 for chart marks against their surface.
