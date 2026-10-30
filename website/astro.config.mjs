// @ts-check
import { defineConfig } from 'astro/config';

// Static site. Set `site` to the deployed origin once there is one, so canonical
// and Open Graph URLs resolve.
export default defineConfig({
  output: 'static',
  build: {
    inlineStylesheets: 'auto',
  },
});
