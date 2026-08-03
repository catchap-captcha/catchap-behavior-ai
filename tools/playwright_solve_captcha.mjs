/**
 * Solve the live main CAPTCHA with a real browser, the way an attacker would.
 *
 * Why this exists
 * ---------------
 * `tools/redteam_main_captcha.py` speaks HTTP directly. It never opens a
 * browser, so every PointerEvent origin signal comes out null:
 *
 *     is_trusted        null      (a real event would be true)
 *     pointer_type      null
 *     pressure          null
 *     coalesced_count   null
 *
 * Validating the new signals against that tool would show perfect separation
 * and mean nothing — it would only prove we can tell "did not use a browser".
 * A real attacker runs Playwright or Selenium, gets `is_trusted: true` for
 * free, and lands squarely in the human range.
 *
 * This driver is that adversary. Everything below is produced by Chromium's
 * own event loop, so a run here is directly comparable to a human session.
 *
 * The open question it answers
 * ----------------------------
 * Does a Playwright-driven mouse produce coalesced events? A physical mouse
 * polls faster than the display refreshes, so the browser hands several raw
 * samples to each `pointermove`. `mouse.move()` dispatches one at a time, so
 * the count should stay at 1 — but that is a guess until measured, and the
 * whole value of `coalesced_count` as a feature rests on it.
 *
 * Usage (nothing runs without an allowed target and --confirm):
 *
 *   node tools/playwright_solve_captcha.mjs \
 *     --base http://localhost:18000 --style bezier --count 20 --confirm
 *
 * Motion styles
 *   linear   straight interpolation, uniform steps
 *   bezier   cubic Bezier with jitter and eased timing
 *   human    bezier plus overshoot-and-correct and a pre-grab hesitation
 */

import { createRequire } from 'node:module';
import fs from 'node:fs';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

// The captcha is our own service and is not meant to be driven anywhere else.
// 61.109.239.231 is our GPU box serving the captcha on :8000 — reachable without the
// tunnel, which is the only way to exercise the insecure-context path teammates hit.
const ALLOWED_HOSTS = new Set(['localhost', '127.0.0.1', '::1', '61.109.239.231']);

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}
const has = (name) => process.argv.includes(name);

const BASE = arg('--base', 'http://localhost:18000');
const STYLE = arg('--style', 'bezier');
const COUNT = Number(arg('--count', '20'));
const DRAG = arg('--drag', 'one');
const REST_MS = Number(arg('--rest', '3000'));
const HEADLESS = !has('--headed');
const OUT = arg('--out', 'reports/playwright_solve.jsonl');

// ---------------------------------------------------------------- motion

function bezierPath(from, to, rng, steps) {
  const span = Math.hypot(to.x - from.x, to.y - from.y) || 1e-6;
  const nx = (to.y - from.y) / span;
  const ny = -(to.x - from.x) / span;
  const control = (r, bow) => ({
    x: from.x + (to.x - from.x) * r + nx * span * bow,
    y: from.y + (to.y - from.y) * r + ny * span * bow,
  });
  const c1 = control(0.3, (rng() - 0.5) * 0.44);
  const c2 = control(0.7, (rng() - 0.5) * 0.44);
  const points = [];
  for (let i = 1; i <= steps; i += 1) {
    const t = i / (steps + 1);
    const u = 1 - t;
    points.push({
      x: u ** 3 * from.x + 3 * u ** 2 * t * c1.x + 3 * u * t ** 2 * c2.x + t ** 3 * to.x
         + (rng() - 0.5) * 1.6,
      y: u ** 3 * from.y + 3 * u ** 2 * t * c1.y + 3 * u * t ** 2 * c2.y + t ** 3 * to.y
         + (rng() - 0.5) * 1.6,
      // ease-in-out: humans accelerate off the grab point and brake into the drop
      gap: Math.max(6, 30 / (Math.sin(Math.PI * t) + 0.25) + (rng() - 0.5) * 10),
    });
  }
  return points;
}

function pathFor(style, from, to, rng) {
  if (style === 'linear') {
    return Array.from({ length: 8 }, (_, i) => {
      const t = (i + 1) / 9;
      return { x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t, gap: 45 };
    });
  }
  const points = bezierPath(from, to, rng, style === 'human' ? 26 : 20);
  if (style === 'human' && rng() < 0.4) {
    // overshoot then settle back — the correction humans make near the target
    const last = points[points.length - 1];
    points.push({ x: to.x + (to.x - last.x) * 0.5, y: to.y + (to.y - last.y) * 0.5, gap: 40 });
    points.push({ x: to.x, y: to.y, gap: 60 });
  }
  return points;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------- one solve

async function solveOnce(page, style, rng, index) {
  const participant = `pwbot-${style}-${DRAG}-${index}`;
  await page.goto(`${BASE}/?participant=${participant}`, { waitUntil: 'domcontentloaded' });

  // The app fetches /api/config then posts a challenge; wait for objects to exist.
  await page.waitForSelector('button.hit-object', { timeout: 15000 });
  const objects = await page.$$('button.hit-object');
  const zone = await page.$('.cc-zone');
  if (!objects.length || !zone) return { participant, outcome: 'no_objects' };

  // Read the coalesced counts off the native events ourselves. The captcha's own
  // handler reports null for these (it reads a React SyntheticEvent, which copies
  // properties but not methods), so the wire value cannot answer the question.
  await page.evaluate(() => {
    window.__coalesced = [];
    window.addEventListener('pointermove', (e) => {
      window.__coalesced.push(e.getCoalescedEvents ? e.getCoalescedEvents().length : null);
    }, true);
  });

  const zoneBox = await zone.boundingBox();
  const drop = { x: zoneBox.x + zoneBox.width / 2, y: zoneBox.y + zoneBox.height / 2 };

  // `--drag one` drags a single object; `all` drags every one. This is not a
  // style choice — dragging everything reliably lands on a honeypot, and the
  // deployed captcha then blocks before it ever calls the model, so those runs
  // produce no score at all. One object is also what a real attacker does.
  const chosen = DRAG === 'one' ? objects.slice(0, 1) : objects;

  let dragged = 0;
  for (const obj of chosen) {
    const box = await obj.boundingBox();
    if (!box) continue;
    const from = { x: box.x + box.width / 2, y: box.y + box.height / 2 };

    await page.mouse.move(from.x, from.y);
    await sleep(120 + rng() * 260);            // reading pause before the grab
    await page.mouse.down();
    for (const p of pathFor(style, from, drop, rng)) {
      await page.mouse.move(p.x, p.y);
      await sleep(p.gap);
    }
    await page.mouse.up();
    await sleep(40 + rng() * 120);
    dragged += 1;
  }

  // What Chromium actually reported for the events this run produced.
  const observed = await page.evaluate(() => {
    const all = window.__coalesced || [];
    // Drop the missing ones before aggregating. Math.max coerces null to 0, so
    // leaving them in reports "every move coalesced 0" on an origin where the API
    // does not exist at all — the two cases must not look alike.
    const c = all.filter((v) => typeof v === 'number');
    return {
      hasCoalesced: typeof PointerEvent.prototype.getCoalescedEvents === 'function',
      coalescedMax: c.length ? Math.max(...c) : null,
      coalescedMean: c.length ? Number((c.reduce((a, b) => a + b, 0) / c.length).toFixed(2)) : null,
      moves: all.length,
      movesWithCoalesced: c.length,
      secureContext: window.isSecureContext,
    };
  });

  const confirm = await page.$('button.cc-verify');
  if (!confirm) return { participant, outcome: 'no_verify_button', dragged, observed };
  await confirm.click();
  await page.waitForTimeout(2000);

  return { participant, outcome: 'submitted', dragged, observed };
}

// ---------------------------------------------------------------- main

async function main() {
  const host = new URL(BASE).hostname;
  if (!ALLOWED_HOSTS.has(host)) {
    console.error(`refusing target '${host}'. This drives a live CAPTCHA and is meant for our `
      + `own service over the tunnel or loopback. Allowed: ${[...ALLOWED_HOSTS].join(', ')}`);
    process.exit(1);
  }
  if (!has('--confirm')) {
    console.error('pass --confirm; this sends real solve attempts to the live captcha');
    process.exit(1);
  }

  const browser = await chromium.launch({ headless: HEADLESS });
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();

  let state = 20260731;
  const rng = () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };

  const rows = [];
  for (let i = 0; i < COUNT; i += 1) {
    try {
      const row = await solveOnce(page, STYLE, rng, i);
      rows.push(row);
      console.log(`[${i + 1}/${COUNT}] ${row.outcome} dragged=${row.dragged ?? '-'} `
        + `moves=${row.observed?.moves ?? '-'} `
        + `coalesced ${row.observed?.movesWithCoalesced ?? '-'}/${row.observed?.moves ?? '-'} `
        + `max=${row.observed?.coalescedMax ?? 'n/a'} mean=${row.observed?.coalescedMean ?? 'n/a'} `
        + `${row.participant}`);
    } catch (error) {
      rows.push({ index: i, outcome: 'error', detail: String(error).slice(0, 160) });
      console.log(`[${i + 1}/${COUNT}] error ${String(error).slice(0, 120)}`);
    }
    // Through the tunnel every request reaches the captcha as 127.0.0.1, so this
    // run shares the per-IP budget with anyone collecting human data right now.
    if (i + 1 < COUNT) await sleep(REST_MS);
  }

  await browser.close();
  fs.mkdirSync(OUT.replace(/\/[^/]+$/, ''), { recursive: true });
  fs.appendFileSync(OUT, rows.map((r) => JSON.stringify(r)).join('\n') + '\n');

  console.log(`\nappended ${rows.length} rows to ${OUT}`);
  console.log('Scores are read from the model side, not here. Join on participant_id:');
  console.log("  SELECT a.participant_id, p.human_probability, p.recommended_action");
  console.log("  FROM ai_behavior_attempts a JOIN ai_model_predictions p ON p.attempt_id = a.id");
  console.log("  WHERE a.participant_id LIKE 'pwbot-%';");
}

main().catch((error) => { console.error(error); process.exit(1); });
