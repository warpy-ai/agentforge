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

`npm run build` writes the site to `dist/`. The dev server and preview serve the page under the
GitHub Pages base path: http://localhost:4321/agentforge/.

## Deployment

The site is published to GitHub Pages at https://warpy-ai.github.io/agentforge/ by
[`.github/workflows/pages.yml`](../.github/workflows/pages.yml), which runs on every push to
`main` that touches `website/` (or by hand from the Actions tab). The repository setting
"Pages > Build and deployment > Source" must be "GitHub Actions".

`astro.config.mjs` sets `site` and `base` for that URL. For a custom domain, set `site` to the
new origin, remove `base`, and add a `CNAME` file to `public/`.

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

## Metadata

Title, description, keywords, Open Graph and Twitter tags, JSON-LD and the social card all come
from the `seo` object in `src/data/content.ts`. `npm run og` regenerates `public/og-image.png`
and `public/apple-touch-icon.png` from `scripts/og-image.mjs` after copy changes. `public/sitemap.xml`
lists the one page; update its `lastmod` when the content changes.

Numbers on the page come from `docs/design/0001-approach.md` and the bundled machine files in
`crates/agentforge-machine/machines/`. Update `src/data/content.ts` when they change.
