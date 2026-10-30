// Renders public/og-image.png (1200x630, the social card) and public/apple-touch-icon.png
// (180x180) in the site's style. Text is converted to outlines with opentype.js, so the
// output does not depend on installed fonts; static TTFs for the two typefaces are fetched
// once from Google Fonts into .cache/fonts/ (ignored by git).
//
//   npm run og

import { access, mkdir, readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Resvg } from '@resvg/resvg-js';

const require = createRequire(import.meta.url);
const opentype = require('opentype.js');

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const cache = join(root, '.cache', 'fonts');
const pub = join(root, 'public');

const W = 1200;
const H = 630;
const PAD = 72;

const color = {
  bg: '#0c0b09',
  line2: '#3d382e',
  paper: '#ede6d6',
  paper2: '#cfc7b4',
  mute: '#8d8676',
  accent: '#ff4f1f',
  ink: '#0c0b09',
};

async function fetchFont(family, weight) {
  const file = join(cache, `${family.toLowerCase().replace(/\s+/g, '-')}-${weight}.ttf`);
  try {
    await access(file);
    return file;
  } catch {
    // not cached yet
  }
  const query = `family=${family.replace(/\s+/g, '+')}:wght@${weight}`;
  // An old user agent makes the Google Fonts API answer with static TTF instances.
  const css = await fetch(`https://fonts.googleapis.com/css2?${query}`, {
    headers: { 'user-agent': 'Mozilla/4.0' },
  }).then((r) => r.text());
  const url = css.match(/url\((https:[^)]+\.ttf)\)/)?.[1];
  if (!url) throw new Error(`No TTF for ${family} ${weight}`);
  const bytes = Buffer.from(await fetch(url).then((r) => r.arrayBuffer()));
  await mkdir(cache, { recursive: true });
  await writeFile(file, bytes);
  return file;
}

const COORDS = ['x', 'y', 'x1', 'y1', 'x2', 'y2'];

// Loads a static font and wraps it with a glyph cache. Each glyph's outline and advance are
// copied on first access: opentype.js's lazily loaded glyphs turned non-finite part-way
// through longer strings, and a snapshot taken before any other access is always clean.
async function loadFont(family, weight) {
  const file = await fetchFont(family, weight);
  const buf = await readFile(file);
  const font = opentype.parse(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const weightClass = font.tables.os2?.usWeightClass;
  if (font.tables.fvar || (weightClass && weightClass !== weight)) {
    throw new Error(`${file} is not a static ${weight} instance (usWeightClass ${weightClass})`);
  }
  const glyphs = new Map();
  const glyph = (ch) => {
    let g = glyphs.get(ch);
    if (!g) {
      const src = font.charToGlyph(ch);
      const commands = src.path.commands.map((c) => ({ ...c }));
      for (const c of commands) {
        for (const k of COORDS) {
          if (k in c && !Number.isFinite(c[k])) throw new Error(`Non-finite outline for "${ch}" in ${family} ${weight}`);
        }
      }
      if (!Number.isFinite(src.advanceWidth)) throw new Error(`No advance for "${ch}" in ${family} ${weight}`);
      g = { index: src.index, advance: src.advanceWidth, commands };
      glyphs.set(ch, g);
    }
    return g;
  };
  return { font, upem: font.unitsPerEm, glyph };
}

// SVG path data for one cached glyph placed at (x, y) in a y-down coordinate system.
function outline(g, x, y, scale) {
  const X = (v) => (x + v * scale).toFixed(2);
  const Y = (v) => (y - v * scale).toFixed(2);
  let d = '';
  for (const c of g.commands) {
    if (c.type === 'M') d += `M${X(c.x)} ${Y(c.y)}`;
    else if (c.type === 'L') d += `L${X(c.x)} ${Y(c.y)}`;
    else if (c.type === 'Q') d += `Q${X(c.x1)} ${Y(c.y1)} ${X(c.x)} ${Y(c.y)}`;
    else if (c.type === 'C') d += `C${X(c.x1)} ${Y(c.y1)} ${X(c.x2)} ${Y(c.y2)} ${X(c.x)} ${Y(c.y)}`;
    else if (c.type === 'Z') d += 'Z';
  }
  return d;
}

// Lays out a string glyph by glyph (kerning plus tracking in em) and returns SVG path data.
function text(f, str, x, y, size, { tracking = 0, anchor = 'start' } = {}) {
  const scale = size / f.upem;
  const glyphs = [...str].map((c) => f.glyph(c));
  const advance = (i) => {
    let a = glyphs[i].advance * scale;
    if (i < glyphs.length - 1) {
      const kern = f.font.getKerningValue(glyphs[i].index, glyphs[i + 1].index);
      a += (Number.isFinite(kern) ? kern : 0) * scale + tracking * size;
    }
    return a;
  };
  let width = 0;
  for (let i = 0; i < glyphs.length; i++) width += advance(i);
  const x0 = anchor === 'end' ? x - width : anchor === 'middle' ? x - width / 2 : x;
  let cursor = x0;
  const parts = [];
  for (let i = 0; i < glyphs.length; i++) {
    const d = outline(glyphs[i], cursor, y, scale);
    if (d) parts.push(d);
    cursor += advance(i);
  }
  const d = parts.join(' ');
  if (!Number.isFinite(width) || d.includes('NaN')) throw new Error(`Bad glyph layout for "${str}"`);
  return { d, width, x: x0 };
}

function path(t, fill) {
  return `<path d="${t.d}" fill="${fill}"/>`;
}

// The stripe mark (A and F from five stripes), same geometry as Mark.astro.
function mark(x, y, scale, fill) {
  const rects = [
    [0, 0, 24, 7], [0, 11, 7, 7], [17, 11, 7, 7], [0, 22, 24, 7], [0, 33, 7, 7], [17, 33, 7, 7], [0, 44, 7, 7], [17, 44, 7, 7],
    [34, 0, 24, 7], [34, 11, 7, 7], [34, 22, 17, 7], [34, 33, 7, 7], [34, 44, 7, 7],
  ];
  const body = rects.map(([rx, ry, rw, rh]) => `<rect x="${rx}" y="${ry}" width="${rw}" height="${rh}"/>`).join('');
  return `<g transform="translate(${x} ${y}) scale(${scale})" fill="${fill}">${body}</g>`;
}

// Deterministic barcode, same widths as Barcode.astro.
function barcode(x, y, height, fill) {
  const widths = [2, 1, 1, 3, 1, 2, 1, 1, 1, 2, 3, 1, 1, 2, 1, 1, 2, 1, 3, 1, 1, 2, 1, 1, 1, 2, 1, 3, 1, 2];
  const unit = 2.6;
  let cx = x;
  const bars = widths.map((w) => {
    const r = `<rect x="${cx.toFixed(1)}" y="${y}" width="${(w * unit).toFixed(1)}" height="${height}"/>`;
    cx += (w + 1) * unit;
    return r;
  });
  return { svg: `<g fill="${fill}">${bars.join('')}</g>`, width: cx - unit - x };
}

function corners(inset, size, stroke) {
  const l = [
    [inset, inset, size, 0], [inset, inset, 0, size],
    [W - inset - size, inset, size, 0], [W - inset, inset, 0, size],
    [inset, H - inset, size, 0], [inset, H - inset - size, 0, size],
    [W - inset - size, H - inset, size, 0], [W - inset, H - inset - size, 0, size],
  ];
  return l
    .map(([x, y, w, h]) => `<rect x="${x - (w ? 0 : 1)}" y="${y - (h ? 0 : 1)}" width="${w || 2}" height="${h || 2}" fill="${stroke}"/>`)
    .join('');
}

// Picks the largest size at or below max whose longest line fits the width.
function fitSize(f, lines, maxWidth, max, min, tracking) {
  for (let size = max; size >= min; size -= 2) {
    const widest = Math.max(...lines.map((l) => text(f, l, 0, 0, size, { tracking }).width));
    if (widest <= maxWidth) return size;
  }
  return min;
}

function ogImage(display, mono, monoBold) {
  const inner = W - PAD * 2;
  const parts = [];

  parts.push(`<rect width="${W}" height="${H}" fill="${color.bg}"/>`);
  parts.push(`<defs>
    <pattern id="dots" width="28" height="28" patternUnits="userSpaceOnUse"><circle cx="14" cy="14" r="1.3" fill="${color.line2}"/></pattern>
    <linearGradient id="fade" x1="0" y1="0" x2="0" y2="1"><stop offset="0.35" stop-color="#fff"/><stop offset="1" stop-color="#000"/></linearGradient>
    <mask id="fadeMask"><rect width="${W}" height="${H}" fill="url(#fade)"/></mask>
  </defs>`);
  parts.push(`<rect width="${W}" height="${H}" fill="url(#dots)" mask="url(#fadeMask)"/>`);
  parts.push(corners(26, 16, color.mute));

  // Header row: mark, wordmark, status chip.
  parts.push(mark(PAD, 60, 1.05, color.accent));
  parts.push(path(text(display, 'AGENTFORGE', PAD + 78, 100, 30, { tracking: 0.06 }), color.paper));

  const chipLabel = 'STATUS: M0 BASELINE · 2026-10';
  const chipW = text(monoBold, chipLabel, 0, 0, 15, { tracking: 0.12 }).width + 36;
  const chipX = W - PAD - chipW;
  parts.push(`<rect x="${chipX.toFixed(1)}" y="70" width="${chipW.toFixed(1)}" height="36" fill="${color.accent}"/>`);
  parts.push(path(text(monoBold, chipLabel, chipX + 18, 94, 15, { tracking: 0.12 }), color.ink));

  // Eyebrow and accent rule.
  parts.push(`<rect x="${PAD}" y="188" width="48" height="3" fill="${color.accent}"/>`);
  parts.push(path(text(mono, 'DOC 0001 / PROJECT THESIS / ACCEPTED 2026-10-30', PAD + 66, 196, 17, { tracking: 0.12 }), color.accent));

  // Headline.
  const lines = ['Compile the agent program', 'and the model together.'];
  const size = fitSize(display, lines, inner, 86, 56, -0.02);
  const lineHeight = size * 1.02;
  let y = 196 + 40 + size * 0.88;
  for (const line of lines) {
    parts.push(path(text(display, line, PAD - size * 0.04, y, size, { tracking: -0.02 }), color.paper));
    y += lineHeight;
  }

  // Sub line.
  const subY = y - lineHeight + 58;
  parts.push(path(text(mono, 'COMPILER + RUNTIME FOR AI INFERENCE SILICON  /  RUST  /  APACHE-2.0', PAD, subY, 19, { tracking: 0.08 }), color.paper2));

  // Footer strip.
  const ruleY = H - 108;
  parts.push(`<rect x="${PAD}" y="${ruleY}" width="${inner}" height="1" fill="${color.line2}"/>`);
  const bc = barcode(PAD, ruleY + 28, 30, color.paper);
  parts.push(bc.svg);
  parts.push(path(text(mono, 'AF · 0001 / M0 / REGIME D', PAD + bc.width + 24, ruleY + 48, 15, { tracking: 0.12 }), color.mute));
  parts.push(path(text(mono, 'warpy-ai.github.io/agentforge', W - PAD, ruleY + 48, 15, { tracking: 0.08, anchor: 'end' }), color.mute));

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">${parts.join('\n')}</svg>`;
}

async function main() {
  const [display, mono, monoBold] = await Promise.all([
    loadFont('Archivo', 800),
    loadFont('JetBrains Mono', 500),
    loadFont('JetBrains Mono', 700),
  ]);

  const svg = ogImage(display, mono, monoBold);
  const og = new Resvg(svg).render().asPng();
  await writeFile(join(pub, 'og-image.png'), og);
  console.log(`public/og-image.png  ${W}x${H}  ${(og.length / 1024).toFixed(0)} kB`);

  const favicon = await readFile(join(pub, 'favicon.svg'), 'utf8');
  const icon = new Resvg(favicon, { fitTo: { mode: 'width', value: 180 } }).render().asPng();
  await writeFile(join(pub, 'apple-touch-icon.png'), icon);
  console.log(`public/apple-touch-icon.png  180x180  ${(icon.length / 1024).toFixed(0)} kB`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
