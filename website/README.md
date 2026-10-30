# AgentForge website

The landing page for AgentForge, built with [Astro](https://astro.build). Static output, no
client-side JavaScript, self-hosted fonts.

Design notes are in [DESIGN.md](DESIGN.md).

## Commands

Run from this directory. Astro 7 asks for Node 22.23.4 or newer.

```bash
npm install
```

```bash
npm run dev
```

```bash
npm run build
```

```bash
npm run preview
```

`npm run build` writes the site to `dist/`.

## Layout

```text
src/
  data/content.ts      copy and numbers for every section (links, layers, results, machines, roadmap)
  layouts/Base.astro   document shell, fonts, meta tags
  styles/global.css    tokens and primitives
  components/          one component per section plus Mark, Barcode, SectionHead
  pages/index.astro    the page
public/favicon.svg
```

Numbers on the page come from `docs/design/0001-approach.md` and the bundled machine files in
`crates/agentforge-machine/machines/`. Update `src/data/content.ts` when they change.
