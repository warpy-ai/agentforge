// @ts-check
import { defineConfig } from 'astro/config';

// Deployed to GitHub Pages as a project site, so pages and assets live under /agentforge/.
// If the site moves to a custom domain, set `site` to that origin and remove `base`.
export default defineConfig({
  site: 'https://warpy-ai.github.io',
  base: '/agentforge',
  output: 'static',
  build: {
    inlineStylesheets: 'auto',
  },
});
