/** Capture defensive bot traces from a real local Chrome event loop. */

import crypto from 'node:crypto';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const WIDTH = 540;
const HEIGHT = 280;
const DEFAULT_FAMILIES = [
  'playwright_linear',
  'playwright_bezier',
  'playwright_stop_go',
  'playwright_frame_quantized',
  'playwright_waypoint',
  'playwright_overshoot_correct',
  'playwright_random_timing',
  'playwright_ease_burst',
  'playwright_micro_jitter',
  'playwright_composite',
];
const EASE_BURST_FAMILIES = [
  'playwright_ease_burst_development',
  'playwright_ease_burst_external',
];
const ALL_FAMILIES = [...DEFAULT_FAMILIES, ...EASE_BURST_FAMILIES];
const SCENARIO_DESCRIPTIONS = {
  playwright_linear: 'constant-shape direct movement',
  playwright_bezier: 'single smooth curved movement',
  playwright_stop_go: 'smooth movement with scheduled pauses',
  playwright_frame_quantized: 'fixed-frame stepping with quantized coordinates',
  playwright_waypoint: 'piecewise movement through deterministic intermediate points',
  playwright_overshoot_correct: 'destination overshoot followed by a correction pass',
  playwright_random_timing: 'direct geometry with randomized event timing',
  playwright_ease_burst: 'alternating slow and burst velocity segments',
  playwright_ease_burst_development: 'short linear burst with development-only phase ranges',
  playwright_ease_burst_external: 'short linear burst with parameter-separated external phase ranges',
  playwright_micro_jitter: 'smooth path with correlated micro-jitter',
  playwright_composite: 'waypoints, pauses, curve drift, timing jitter, and correction',
};
const GENERATOR_VERSION = 'playwright_chrome_v3';

function argument(name, fallback) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

function rngFactory(seed) {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function smoothstep(value) {
  return value * value * (3 - 2 * value);
}

function clamp(value, lower, upper) {
  return Math.max(lower, Math.min(upper, value));
}

function lerp(start, end, ratio) {
  return start + (end - start) * ratio;
}

function pointOnSegment(start, end, ratio) {
  const u = smoothstep(ratio);
  return { x: lerp(start.x, end.x, u), y: lerp(start.y, end.y, u) };
}

function segmentPoint(points, ratio) {
  const scaled = ratio * (points.length - 1);
  const segment = Math.min(points.length - 2, Math.floor(scaled));
  return pointOnSegment(points[segment], points[segment + 1], scaled - segment);
}

function buildPoints(count, pointAt, delayAt) {
  return Array.from({ length: count }, (_, index) => {
    const ratio = index / (count - 1);
    const point = pointAt(ratio, index);
    return { ...point, delay: delayAt(ratio, index) };
  });
}

function ranged(random, lower, upper) {
  return lower + (upper - lower) * random();
}

function easeBurstTrajectory(start, end, random, profile) {
  const count = Math.floor(ranged(random, profile.count[0], profile.count[1] + 1));
  const burstStart = ranged(random, profile.burstStart[0], profile.burstStart[1]);
  const burstEnd = ranged(random, profile.burstEnd[0], profile.burstEnd[1]);
  const startDistance = ranged(random, profile.startDistance[0], profile.startDistance[1]);
  const endDistance = ranged(random, profile.endDistance[0], profile.endDistance[1]);
  const burstDistance = 1 - startDistance - endDistance;

  return buildPoints(count, (ratio) => {
    let progress;
    if (ratio < burstStart) progress = startDistance * ratio / burstStart;
    else if (ratio < burstEnd) progress = startDistance + burstDistance * (ratio - burstStart) / (burstEnd - burstStart);
    else progress = startDistance + burstDistance + endDistance * (ratio - burstEnd) / (1 - burstEnd);
    return { x: lerp(start.x, end.x, progress), y: lerp(start.y, end.y, progress) };
  }, (ratio) => {
    const fast = ratio >= burstStart && ratio <= burstEnd;
    const bounds = fast ? profile.fastDelay : profile.slowDelay;
    return Math.max(1, Math.floor(ranged(random, bounds[0], bounds[1] + 1)));
  });
}

function trajectory(family, random) {
  const count = 40 + Math.floor(random() * 24);
  const x0 = 35 + random() * 30;
  const y0 = 70 + random() * 130;
  const x1 = 430 + random() * 65;
  const y1 = 70 + random() * 130;
  const start = { x: x0, y: y0 };
  const end = { x: x1, y: y1 };
  const directPoint = (ratio) => pointOnSegment(start, end, ratio);
  const shortDelay = () => 3 + Math.floor(random() * 5);

  if (family === 'playwright_ease_burst_development') {
    return easeBurstTrajectory(start, end, random, {
      count: [38, 60],
      burstStart: [0.14, 0.28],
      burstEnd: [0.56, 0.69],
      startDistance: [0.08, 0.18],
      endDistance: [0.08, 0.18],
      slowDelay: [9, 18],
      fastDelay: [1, 4],
    });
  }

  if (family === 'playwright_ease_burst_external') {
    return easeBurstTrajectory(start, end, random, {
      count: [48, 72],
      burstStart: [0.34, 0.46],
      burstEnd: [0.72, 0.84],
      startDistance: [0.12, 0.22],
      endDistance: [0.12, 0.22],
      slowDelay: [14, 28],
      fastDelay: [1, 3],
    });
  }

  if (family === 'playwright_linear') {
    return buildPoints(count, directPoint, shortDelay);
  }

  if (family === 'playwright_bezier') {
    const bend = (random() > 0.5 ? 1 : -1) * (35 + random() * 35);
    return buildPoints(count, (ratio) => {
      const point = directPoint(ratio);
      return { x: point.x, y: point.y + Math.sin(Math.PI * ratio) * bend };
    }, shortDelay);
  }

  if (family === 'playwright_stop_go') {
    return buildPoints(count, directPoint, (_, index) => {
      const pause = index > 5 && index % (9 + Math.floor(random() * 5)) === 0;
      return shortDelay() + (pause ? 35 + Math.floor(random() * 55) : 0);
    });
  }

  if (family === 'playwright_frame_quantized') {
    return buildPoints(count, (ratio) => {
      const point = directPoint(ratio);
      return { x: Math.round(point.x / 5) * 5, y: Math.round(point.y / 5) * 5 };
    }, () => 16);
  }

  if (family === 'playwright_waypoint') {
    const middleX = (x0 + x1) / 2;
    const offset = (random() > 0.5 ? 1 : -1) * (28 + random() * 45);
    const waypoints = [
      start,
      { x: middleX - 40 + random() * 20, y: clamp(y0 + offset, 20, HEIGHT - 20) },
      { x: middleX + 25 + random() * 25, y: clamp(y1 - offset * 0.55, 20, HEIGHT - 20) },
      end,
    ];
    return buildPoints(count, (ratio) => segmentPoint(waypoints, ratio), shortDelay);
  }

  if (family === 'playwright_overshoot_correct') {
    const overshoot = {
      x: clamp(x1 + 18 + random() * 42, 10, WIDTH - 10),
      y: clamp(y1 + (random() - 0.5) * 42, 10, HEIGHT - 10),
    };
    return buildPoints(count + 16, (ratio) => {
      if (ratio < 0.82) return pointOnSegment(start, overshoot, ratio / 0.82);
      return pointOnSegment(overshoot, end, (ratio - 0.82) / 0.18);
    }, (_, index) => shortDelay() + (index === Math.floor((count + 16) * 0.82) ? 25 : 0));
  }

  if (family === 'playwright_random_timing') {
    return buildPoints(count, directPoint, () => {
      const delayPool = [2, 3, 4, 6, 9, 15, 25, 40];
      return delayPool[Math.floor(random() * delayPool.length)];
    });
  }

  if (family === 'playwright_ease_burst') {
    return buildPoints(count, (ratio) => {
      const burstRatio = ratio < 0.28
        ? ratio * 0.55
        : ratio < 0.62
          ? 0.154 + (ratio - 0.28) * 1.92
          : 0.807 + (ratio - 0.62) * 0.51;
      return pointOnSegment(start, end, clamp(burstRatio, 0, 1));
    }, (_, index) => {
      const phase = index / (count - 1);
      return phase > 0.28 && phase < 0.62 ? 2 + Math.floor(random() * 3) : 11 + Math.floor(random() * 8);
    });
  }

  if (family === 'playwright_micro_jitter') {
    const phase = random() * Math.PI * 2;
    return buildPoints(count, (ratio) => {
      const point = directPoint(ratio);
      const envelope = Math.sin(Math.PI * ratio);
      return {
        x: point.x + Math.sin(ratio * Math.PI * 9 + phase) * envelope * 1.8,
        y: point.y + Math.cos(ratio * Math.PI * 7 + phase) * envelope * 2.8,
      };
    }, shortDelay);
  }

  if (family === 'playwright_composite') {
    const offset = (random() > 0.5 ? 1 : -1) * (25 + random() * 35);
    const waypoint = { x: (x0 + x1) / 2, y: clamp((y0 + y1) / 2 + offset, 20, HEIGHT - 20) };
    const overshoot = { x: clamp(x1 + 20 + random() * 25, 10, WIDTH - 10), y: clamp(y1 - offset * 0.15, 10, HEIGHT - 10) };
    const total = count + 14;
    return buildPoints(total, (ratio) => {
      let point;
      if (ratio < 0.48) point = pointOnSegment(start, waypoint, ratio / 0.48);
      else if (ratio < 0.84) point = pointOnSegment(waypoint, overshoot, (ratio - 0.48) / 0.36);
      else point = pointOnSegment(overshoot, end, (ratio - 0.84) / 0.16);
      const drift = Math.sin(Math.PI * ratio) * Math.sin(ratio * Math.PI * 5) * 3;
      return { x: point.x, y: point.y + drift };
    }, (_, index) => {
      const pause = index > 8 && index % (12 + Math.floor(random() * 5)) === 0;
      return 3 + Math.floor(random() * 12) + (pause ? 20 + Math.floor(random() * 35) : 0);
    });
  }

  throw new Error(`Unsupported Playwright bot family: ${family}`);
}

function payload(attemptId, family, events, role) {
  return {
    schema_version: '1.0',
    attempt_id: attemptId,
    challenge_id: 'local_playwright_capture',
    session_id: `local_${family}`,
    anonymous_participant_id: null,
    captcha: { width: WIDTH, height: HEIGHT },
    timing: { presented_at: null, submitted_at: null },
    events,
    interaction: {
      regrab_count: 0,
      retry_count: 0,
      pointercancel_count: 0,
      empty_click_count: 0,
      failed_drop_count: 0,
    },
    collection: {
      label: 'bot',
      label_source: 'playwright',
      bot_family: family,
      generator_version: GENERATOR_VERSION,
      training_usage: role === 'development' ? 'development_only' : 'external_holdout_only',
      age_group: 'unknown',
    },
    position_correct: true,
    interaction_success: true,
    final_drop_error: 0.0,
  };
}

const output = argument('--out', 'data/interim/playwright_bots_300_20260722.jsonl');
const perFamily = Number(argument('--per-family', '30'));
const seed = Number(argument('--seed', '20260721'));
const role = argument('--role', 'external_holdout');
const requestedFamilies = argument('--families', null);
const families = requestedFamilies
  ? requestedFamilies.split(',').map((family) => family.trim()).filter(Boolean)
  : DEFAULT_FAMILIES;
if (!Number.isInteger(perFamily) || perFamily < 1) {
  throw new Error('--per-family must be a positive integer');
}
if (!['development', 'external_holdout'].includes(role)) {
  throw new Error('--role must be development or external_holdout');
}
if (!families.length || families.some((family) => !ALL_FAMILIES.includes(family))) {
  throw new Error(`--families must use known values: ${ALL_FAMILIES.join(', ')}`);
}

const executablePath = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const browser = await chromium.launch({ executablePath, headless: true });
const page = await browser.newPage({ viewport: { width: 640, height: 380 } });
await page.setContent(`
  <!doctype html>
  <style>
    body { margin: 0; padding: 40px; }
    #capture { width: ${WIDTH}px; height: ${HEIGHT}px; background: #fff; border: 1px solid #000; }
  </style>
  <div id="capture"></div>
  <script>
    const capture = document.getElementById('capture');
    window.trace = [];
    let active = false;
    let start = 0;
    function append(event) {
      const rect = capture.getBoundingClientRect();
      const x = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
      const y = Math.max(0, Math.min(rect.height, event.clientY - rect.top));
      window.trace.push({
        event_type: event.type,
        t_ms: performance.now() - start,
        x,
        y,
        x_normalized: x / rect.width,
        y_normalized: y / rect.height,
      });
    }
    capture.addEventListener('pointerdown', (event) => {
      window.trace = [];
      start = performance.now();
      active = true;
      capture.setPointerCapture(event.pointerId);
      append(event);
    });
    capture.addEventListener('pointermove', (event) => { if (active) append(event); });
    capture.addEventListener('pointerup', (event) => { append(event); active = false; });
  </script>
`);

const box = await page.locator('#capture').boundingBox();
if (!box) throw new Error('capture box not found');

const rows = [];
for (let familyIndex = 0; familyIndex < families.length; familyIndex += 1) {
  const family = families[familyIndex];
  for (let index = 0; index < perFamily; index += 1) {
    const random = rngFactory(seed + familyIndex * 100003 + index);
    const points = trajectory(family, random);
    const first = points[0];
    await page.mouse.move(box.x + first.x, box.y + first.y);
    await page.mouse.down();
    for (const point of points.slice(1)) {
      await page.waitForTimeout(point.delay);
      await page.mouse.move(box.x + point.x, box.y + point.y);
    }
    await page.mouse.up();
    const captured = await page.evaluate(() => window.trace);
    const events = captured.map((event, seq) => ({
      seq,
      event_type: event.event_type,
      t_ms: Math.round(event.t_ms * 1000) / 1000,
      x: Math.round(event.x * 1000) / 1000,
      y: Math.round(event.y * 1000) / 1000,
      x_normalized: Math.round(event.x_normalized * 1e6) / 1e6,
      y_normalized: Math.round(event.y_normalized * 1e6) / 1e6,
      target_role: 'captcha_area',
    }));
    rows.push(payload(`playwright_${family}_${String(index).padStart(4, '0')}`, family, events, role));
  }
}

await browser.close();
fs.mkdirSync(path.dirname(output), { recursive: true });
const body = `${rows.map((row) => JSON.stringify(row)).join('\n')}\n`;
fs.writeFileSync(output, body, 'utf8');
const counts = Object.fromEntries(families.map((family) => [family, perFamily]));
const manifest = {
  dataset_name: path.basename(output, '.jsonl'),
  generator_version: GENERATOR_VERSION,
  seed,
  browser: 'Google Chrome via Playwright',
  total_rows: rows.length,
  family_count: families.length,
  scenario_descriptions: Object.fromEntries(families.map((family) => [family, SCENARIO_DESCRIPTIONS[family]])),
  per_family: counts,
  role,
  training_usage: role === 'development' ? 'development_only' : 'external_holdout_only',
  database_modified: false,
  output: {
    path: output,
    bytes: Buffer.byteLength(body),
    sha256: crypto.createHash('sha256').update(body).digest('hex'),
  },
};
fs.writeFileSync(`${output}.manifest.json`, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
process.stdout.write(`${JSON.stringify(manifest, null, 2)}\n`);
