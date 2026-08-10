/**
 * Capture automation traces from a browser that the rule gate cannot see.
 *
 * Why this family did not exist
 * -----------------------------
 * `capture_playwright_bots.mjs` launches headless with no stealth, so a real
 * attempt from it reports `navigator.webdriver === true` and a headless user
 * agent. `automation_score()` in the CAPTCHA gives those +80 each against a
 * block threshold of 80, so those bots are refused before the trajectory model
 * is consulted at all. Measuring them against the trajectory model and calling
 * 20.7% a failure was measuring the wrong thing.
 *
 * The threat that matters is the one that clears that gate. Hiding
 * `navigator.webdriver` is a one-line `Object.defineProperty`, and the user
 * agent is a launch flag. Neither requires skill or money. So a bot with
 * `automation_score` of exactly 0 is the realistic adversary, and until now the
 * corpus contained no such thing.
 *
 * What this produces
 * ------------------
 * Traces from a real Chrome event loop — genuine `isTrusted` PointerEvents with
 * the browser's own timing and coalescing — from a page where the automation
 * signals read clean. Every row carries the measured signals so the claim can be
 * checked rather than trusted:
 *
 *     "automation_signals": { "webdriver": false, "headlessUA": false, ... },
 *     "automation_score": 0
 *
 * The script refuses to write if any row scores above zero. A dataset that
 * quietly failed to hide would look like a hard family and be a soft one.
 *
 * Motion profiles span what an attacker of this sophistication would actually
 * reach for, weakest first:
 *
 *   stealth_linear   the library's own interpolation. Free, and the floor.
 *   stealth_eased    minimum-jerk easing over a Bezier — humanised-cursor
 *                    libraries in one line of npm.
 *   stealth_replay   a captured human drag, replayed through the real mouse.
 *                    The strongest, and the one the trajectory model provably
 *                    cannot separate (measured AUC 0.516 on the aim surface).
 *
 * Rows are labelled `rtbot-` so they can never be confused with collected human
 * sessions. That separation is not paranoia: on 2026-08-10 one person collected
 * under three participant codes, the codes were read as three people, and a
 * cross-person false-pair rate of 0.19% was reported from data containing
 * exactly one person. Labels that cannot be untangled later must not be created.
 *
 *   node tools/capture_stealth_bots.mjs --per-family 100 --role external_holdout
 */

import crypto from 'node:crypto';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const GENERATOR_VERSION = 'stealth_v1';
const WIDTH = 500;
const HEIGHT = 334;
const FAMILIES = ['stealth_linear', 'stealth_eased', 'stealth_replay'];
const DESCRIPTIONS = {
  stealth_linear: 'rule gate evaded; playwright default interpolation',
  stealth_eased: 'rule gate evaded; minimum-jerk easing over a bezier',
  stealth_replay: 'rule gate evaded; captured human drag replayed through the real mouse',
};

function args(flag, fallback) {
  const index = process.argv.indexOf(flag);
  return index === -1 ? fallback : process.argv[index + 1];
}

const perFamily = Number(args('--per-family', '100'));
const role = args('--role', 'external_holdout');
const seed = Number(args('--seed', '20260810'));
const humanSource = args('--human', 'data/interim/collection_20260806.jsonl');
const output = args('--out', `data/interim/stealth_bots_${role}_${perFamily * FAMILIES.length}_20260810.jsonl`);

if (!Number.isInteger(perFamily) || perFamily < 1) throw new Error('--per-family must be a positive integer');
if (!['development', 'external_holdout'].includes(role)) throw new Error('--role must be development or external_holdout');

/** Deterministic RNG so a run can be repeated exactly. */
function rngFactory(value) {
  let state = value >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/**
 * Real human drags, used as the substrate for `stealth_replay`.
 * Sealed people are refused: warping their traces would put their movement into
 * the corpus through the back door and destroy the only unseen-person set there is.
 */
function loadHumanDrags(file) {
  const SEALED = new Set(['sw', 'ms']);
  const out = [];
  for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const record = JSON.parse(line);
    const person = String(record.participant_id || '?').split('-')[0];
    if (SEALED.has(person)) continue;
    if (record.quality_status !== 'valid') continue;
    const rows = (record.events || []).filter((e) => e.x_pixel != null);
    if (rows.length < 8) continue;
    const base = rows[0].client_timestamp_ms || 0;
    out.push(rows.map((r) => ({
      x: Number(r.x_pixel),
      y: Number(r.y_pixel),
      t: Number((r.client_timestamp_ms || base) - base),
    })));
  }
  if (!out.length) throw new Error(`no usable human drags in ${file}`);
  return out;
}

function trajectory(family, random, humans) {
  const start = { x: 40 + random() * 80, y: 40 + random() * 60 };
  const end = { x: WIDTH - 120 + random() * 60, y: HEIGHT - 110 + random() * 60 };
  const steps = 10 + Math.floor(random() * 8);

  if (family === 'stealth_replay') {
    // Map a real drag onto this start/end. The shape is a person's; only the
    // placement is ours.
    const source = humans[Math.floor(random() * humans.length)];
    const sx = source[0].x;
    const sy = source[0].y;
    const ex = source[source.length - 1].x;
    const ey = source[source.length - 1].y;
    const spanX = ex - sx || 1;
    const spanY = ey - sy || 1;
    return source.map((point, index) => ({
      x: start.x + ((point.x - sx) / spanX) * (end.x - start.x),
      y: start.y + ((point.y - sy) / spanY) * (end.y - start.y),
      delay: index === 0 ? 0 : Math.max(8, point.t - source[index - 1].t),
    }));
  }

  const points = [];
  const c1 = { x: start.x + (end.x - start.x) * 0.3 + (random() - 0.5) * 90,
               y: start.y + (end.y - start.y) * 0.3 + (random() - 0.5) * 70 };
  const c2 = { x: start.x + (end.x - start.x) * 0.7 + (random() - 0.5) * 90,
               y: start.y + (end.y - start.y) * 0.7 + (random() - 0.5) * 70 };
  for (let i = 0; i <= steps; i += 1) {
    const raw = i / steps;
    // Minimum-jerk easing for `stealth_eased`; linear for the floor family.
    const u = family === 'stealth_eased'
      ? 10 * raw ** 3 - 15 * raw ** 4 + 6 * raw ** 5
      : raw;
    const inv = 1 - u;
    const x = family === 'stealth_linear'
      ? start.x + (end.x - start.x) * u
      : inv ** 3 * start.x + 3 * inv ** 2 * u * c1.x + 3 * inv * u ** 2 * c2.x + u ** 3 * end.x;
    const y = family === 'stealth_linear'
      ? start.y + (end.y - start.y) * u
      : inv ** 3 * start.y + 3 * inv ** 2 * u * c1.y + 3 * inv * u ** 2 * c2.y + u ** 3 * end.y;
    points.push({ x, y, delay: i === 0 ? 0 : 12 + Math.floor(random() * 26) });
  }
  return points;
}

const humans = loadHumanDrags(humanSource);
process.stdout.write(`human drags for replay substrate: ${humans.length}\n`);

const browser = await chromium.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
  // Drop the automation banner and the `--headless` marker the UA otherwise carries.
  args: ['--disable-blink-features=AutomationControlled'],
});
const context = await browser.newContext({
  viewport: { width: 700, height: 460 },
  // A perfectly ordinary desktop Chrome string. `headlessUA` keys off /headless/i.
  userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
    + '(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
  locale: 'ko-KR',
});
// Runs before any page script, so nothing ever observes the true value.
await context.addInitScript(() => {
  Object.defineProperty(navigator, 'webdriver', { get: () => false, configurable: true });
});
const page = await context.newPage();

await page.setContent(`
  <!doctype html>
  <style>body{margin:0;padding:40px}#capture{width:${WIDTH}px;height:${HEIGHT}px;background:#fff;border:1px solid #000}</style>
  <div id="capture"></div>
  <script>
    const capture = document.getElementById('capture');
    window.trace = []; let active = false; let start = 0;
    function append(event) {
      const rect = capture.getBoundingClientRect();
      const x = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
      const y = Math.max(0, Math.min(rect.height, event.clientY - rect.top));
      window.trace.push({
        event_type: event.type, t_ms: performance.now() - start,
        x, y, x_normalized: x / rect.width, y_normalized: y / rect.height,
        is_trusted: event.isTrusted, pointer_type: event.pointerType,
        coalesced_count: event.getCoalescedEvents ? event.getCoalescedEvents().length : null,
      });
    }
    capture.addEventListener('pointerdown', (e) => {
      window.trace = []; start = performance.now(); active = true;
      capture.setPointerCapture(e.pointerId); append(e);
    });
    capture.addEventListener('pointermove', (e) => { if (active) append(e); });
    capture.addEventListener('pointerup', (e) => { append(e); active = false; });
  </script>
`);

/** Exactly the CAPTCHA's own signal shape and scoring, so the claim is checkable. */
const signals = await page.evaluate(() => ({
  webdriver: navigator.webdriver === true,
  headlessUA: /headless/i.test(navigator.userAgent || ''),
  languages: (navigator.languages || []).length,
  cores: navigator.hardwareConcurrency || 0,
}));
const automationScore = (s) => (s.webdriver ? 80 : 0) + (s.headlessUA ? 80 : 0)
  + (s.languages === 0 ? 15 : 0) + (s.cores === 0 ? 10 : 0);
const score = automationScore(signals);
process.stdout.write(`automation signals ${JSON.stringify(signals)} -> score ${score}\n`);
if (score !== 0) {
  await browser.close();
  throw new Error(`stealth failed: automation_score ${score} — 규칙 게이트를 통과하지 못하는 봇은 이 계열의 목적이 아니다`);
}

const box = await page.locator('#capture').boundingBox();
if (!box) throw new Error('capture box not found');

const rows = [];
for (let familyIndex = 0; familyIndex < FAMILIES.length; familyIndex += 1) {
  const family = FAMILIES[familyIndex];
  for (let index = 0; index < perFamily; index += 1) {
    const random = rngFactory(seed + familyIndex * 100003 + index);
    const points = trajectory(family, random, humans);
    await page.mouse.move(box.x + points[0].x, box.y + points[0].y);
    await page.mouse.down();
    for (const point of points.slice(1)) {
      await page.waitForTimeout(point.delay);
      await page.mouse.move(box.x + point.x, box.y + point.y);
    }
    await page.mouse.up();
    const captured = await page.evaluate(() => window.trace);
    if (captured.length < 5) continue;
    rows.push({
      attempt_id: `rtbot-stealth-${family}-${String(index).padStart(4, '0')}`,
      session_id: `rtbot-stealth-${family}-${String(index).padStart(4, '0')}`,
      challenge_id: 'rtbot_stealth_challenge',
      collection: {
        label: 'bot',
        label_source: 'stealth_automation_capture',
        bot_family: family,
        generator_version: `${GENERATOR_VERSION}_${family}`,
        generator_version_base: GENERATOR_VERSION,
      },
      automation_signals: signals,
      automation_score: score,
      captcha: { width: WIDTH, height: HEIGHT },
      events: captured.map((event, seq) => ({
        seq,
        event_type: event.event_type,
        t_ms: Math.round(event.t_ms * 1000) / 1000,
        x: Math.round(event.x * 1000) / 1000,
        y: Math.round(event.y * 1000) / 1000,
        x_normalized: Math.round(event.x_normalized * 1e6) / 1e6,
        y_normalized: Math.round(event.y_normalized * 1e6) / 1e6,
        is_trusted: event.is_trusted,
        pointer_type: event.pointer_type,
        coalesced_count: event.coalesced_count,
        target_role: 'captcha_area',
      })),
    });
  }
  process.stdout.write(`  ${family}: ${rows.length} rows so far\n`);
}

await browser.close();

fs.mkdirSync(path.dirname(output), { recursive: true });
const body = `${rows.map((row) => JSON.stringify(row)).join('\n')}\n`;
fs.writeFileSync(output, body, 'utf8');
const manifest = {
  dataset_name: path.basename(output, '.jsonl'),
  generator_version: GENERATOR_VERSION,
  seed,
  browser: 'Google Chrome via Playwright, webdriver hidden, desktop UA',
  automation_signals: signals,
  automation_score: score,
  note: 'automation_score 0 — 민서의 규칙 게이트를 통과한다. 궤적층이 유일한 방어선인 계열.',
  total_rows: rows.length,
  family_count: FAMILIES.length,
  scenario_descriptions: DESCRIPTIONS,
  per_family: Object.fromEntries(FAMILIES.map((f) => [f, perFamily])),
  role,
  training_usage: role === 'development' ? 'development_only' : 'external_holdout_only',
  human_substrate: { source: humanSource, drags: humans.length, sealed_people_refused: ['sw', 'ms'] },
  database_modified: false,
  output: {
    path: output,
    bytes: Buffer.byteLength(body),
    sha256: crypto.createHash('sha256').update(body).digest('hex'),
  },
};
fs.writeFileSync(`${output}.manifest.json`, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
process.stdout.write(`${JSON.stringify(manifest, null, 2)}\n`);
