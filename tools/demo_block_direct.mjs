/**
 * 시연용 — 캡차 서버를 **직접** 두드려 「차단」 상태를 만든다.
 *
 * 왜 화면을 안 태우나
 * -------------------
 * `demo_blocked_bot.mjs` 는 실제 화면을 몰아서 차단까지 갔다. 그게 맞는 그림이지만
 * 한 판에 3~10분이 든다 — 로그인 6회 실패(캡차는 6번째에 뜬다)에 30초, 거기에 우리가
 * 건 대기 사다리(오답 4·5·6번째에 5·20·60초)를 봇이 그대로 앉아서 기다린다. 게다가
 * 화면 조작이 자주 새서(0813 실측: 객체 7개 중 4개만 선택) 라운드가 통째로 헛돈다.
 *
 * 차단은 **서버가 세션 단위로** 판정한다. 화면은 그 결과를 그릴 뿐이다. 그래서 여기서는
 * 서버에만 오답을 던져 세션을 차단 상태로 만든다. 걸리는 시간은 대기 사다리뿐이다.
 *
 * 시연영상은 이렇게 만든다
 * ------------------------
 *   ① 이 도구로 세션 하나를 차단 상태로 만든다 (약 1분 반)
 *   ② 찍힌 세션 번호를 브라우저 sessionStorage 에 심고 로그인 화면을 연다
 *   ③ 위젯이 문항을 받으려다 429(blocked)를 맞고 **차단 화면을 그린다** — 이걸 녹화한다
 *
 * ②③ 은 `--open` 을 주면 이 도구가 이어서 한다.
 *
 *   node tools/demo_block_direct.mjs
 *   node tools/demo_block_direct.mjs --open --video ./demo_video
 *
 * ⚠️실서비스를 두드린다. 만들어 쓰는 세션은 이 도구가 만든 것뿐이고 실제 사용자 세션은
 *   건드리지 않는다.
 */
import { createHash } from 'node:crypto';

const argument = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
};

const GUARD = argument('--guard', 'https://captcha.catchap5.com');
const SITE = argument('--site', 'https://www.catchap5.com');
/** 공개값이다 — 화면 코드(`catchapGuard.ts`)에 그대로 박혀 있는 것과 같다. */
const SITE_KEY = argument('--site-key', 'site_ENd3JivVHLliFXYaEG3bAiDx3eFd2PNd');
const ROUNDS = Number(argument('--rounds', '9'));
const OPEN = process.argv.includes('--open');
const VIDEO_DIR = argument('--video', null);

// ★`guard-test-` 로 시작하게 둔다 — 시험 트래픽을 통계에서 뺄 수 있는 유일한 표시다.
// 실제 사이트는 이 모양을 절대 만들지 않는다(위젯은 `guard-<시각>-<무작위>` 를 만든다).
const sessionId = argument('--session', `guard-test-${Date.now()}-direct`);

/** sha256(seed + ":" + nonce) 의 선행 0비트가 `bits` 이상인 nonce. 화면 쪽과 같은 규칙. */
function solvePow(pow) {
  if (!pow?.seed) return null;
  const { seed, bits } = pow;
  for (let nonce = 0; nonce < 20_000_000; nonce += 1) {
    const digest = createHash('sha256').update(`${seed}:${nonce}`).digest();
    let zeros = 0;
    for (const byte of digest) {
      if (byte === 0) { zeros += 8; continue; }
      zeros += Math.clz32(byte) - 24;
      break;
    }
    if (zeros >= bits) return String(nonce);
  }
  return null;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * 429 를 예외로 던지지 않고 그대로 돌려준다 — 차단이 이 도구의 **목표**라서,
 * 실패로 처리하면 정작 원하는 결과를 못 읽는다.
 */
async function call(path, body) {
  const res = await fetch(GUARD + path, {
    method: 'POST',
    // 오리진을 안 붙이면 403 이다 — 캡차 서버가 site key 에 등록된 도메인만 받는다
    // (iframe 임베드가 막히는 것과 같은 검사다). 화면이 www 에서 부르는 걸 그대로 흉내낸다.
    headers: {
      'Content-Type': 'application/json',
      'X-Captcha-Site-Key': SITE_KEY,
      Origin: SITE,
      Referer: `${SITE}/login`,
    },
    body: JSON.stringify(body),
  });
  if (res.status === 429) {
    const after = Number(res.headers.get('Retry-After'));
    // Retry-After 가 없는 429 는 대기가 아니라 **상한에 걸린 것**이다(분당 문항 수,
    // 채점 실패 누적). 이걸 "5초 대기" 로 적으면 로그가 거짓말을 한다 — 0813 에
    // 이것 때문에 상한을 대기로 착각하고 아홉 판을 헛돌렸다.
    return {
      retry: true,
      seconds: Number.isFinite(after) && after > 0 ? Math.ceil(after) : 0,
      reason: res.headers.get('X-Captcha-Retry-Reason') === 'blocked' ? 'blocked'
        : (Number.isFinite(after) && after > 0 ? 'wait' : 'capped'),
      detail: (await res.text()).slice(0, 80),
    };
  }
  // 오류 응답은 JSON 이 아닐 수 있다 — 500 이면 게이트웨이가 평문 "Internal Server Error"
  // 를 준다. 그대로 파싱하면 도구가 죽어서 정작 무엇이 잘못됐는지 못 본다.
  const text = await res.text();
  try {
    return { status: res.status, body: text ? JSON.parse(text) : null };
  } catch {
    return { status: res.status, body: null, text: text.slice(0, 120) };
  }
}

/**
 * 기계 같은 궤적을 만들어 보낸다 — **모양이 매 판 똑같다.**
 *
 * 왜 보내야 하나: 궤적을 안 보내면 행동 AI 가 채점을 못 하고(`behavior_batches_missing`),
 * 채점이 없으면 위험도가 안 붙는다. 그런데 차단 조건은 "**의심**이 붙은 채로 5회 오답"
 * 이라, 궤적 없는 봇은 영원히 대기 사다리만 돌고 차단까지 안 간다(0813 실측: 9라운드
 * 전부 대기).
 *
 * 왜 모양을 고정하나: 집는 자리는 문항마다 다르지만 **거기서부터의 이동은 매번 같게**
 * 둔다. 우리 재생 탐지가 궤적의 모양을 보므로 두세 번이면 `dtw_similar_trace`(중간),
 * 이어서 `exact_trace_fingerprint`(높음)가 붙는다 — 0813 에 브라우저 봇으로 확인한
 * 경로 그대로다. 등속·직선·일정 간격이라 그 자체로도 기계 신호다.
 */
function mechanicalTrace(objectId, cx, cy, startedAt) {
  const events = [];
  const at = (offset, type, extra = {}) =>
    events.push({ type, timestamp_ms: startedAt + offset, ...extra });
  // 좌표는 0~1 을 벗어나면 서버가 배치를 통째로 반려한다(422). 집는 자리가 오른쪽·
  // 아래쪽이면 반대로 끌어 화면 안에 머무르게 한다 — 모양은 그대로(직선·등속)다.
  const dx = (cx > 0.5 ? -1 : 1) * 0.02;
  const dy = (cy > 0.5 ? -1 : 1) * 0.01;
  const clamp = (value) => Math.max(0, Math.min(1, Number(value.toFixed(4))));

  at(0, 'challenge_loaded');
  at(300, 'pointer_down', { object_id: objectId, x: clamp(cx), y: clamp(cy) });
  at(320, 'drag_start', { object_id: objectId, x: clamp(cx), y: clamp(cy) });
  // 40ms 간격 · 같은 보폭 — 사람 손은 이렇게 못 움직인다(간격 불규칙 AUC 0.86).
  for (let step = 1; step <= 12; step += 1) {
    at(320 + step * 40, 'pointer_move',
      { object_id: objectId, x: clamp(cx + step * dx), y: clamp(cy + step * dy) });
  }
  at(840, 'drop', { object_id: objectId, x: clamp(cx + 12 * dx), y: clamp(cy + 12 * dy) });
  at(860, 'selection_add', { object_id: objectId, x: clamp(cx + 12 * dx), y: clamp(cy + 12 * dy) });
  at(900, 'submit');
  return events.map((event, index) => ({ seq: index, ...event }));
}

/**
 * 함정을 걸러낸다 — 미리보기 이미지가 404 인 것이 함정이다.
 *
 * 시도를 쓰지 않는다. 사진 조각을 받아보는 GET 뿐이라 캡차 판정에 안 잡힌다.
 */
async function withoutHoneypots(objects) {
  const checked = await Promise.all(objects.map(async (object) => {
    const asset = await fetch(GUARD + object.preview_url,
      { headers: { Origin: SITE, Referer: `${SITE}/login` } }).catch(() => null);
    return asset && asset.ok ? object : null;
  }));
  return checked.filter(Boolean);
}

/** 배치는 영수증을 이어 붙여야 받아준다. 한 번 끊기면 그 뒤가 전부 거부된다. */
async function sendTrace(challengeId, nonce, events) {
  const sent = await call(`/api/captcha/challenges/${challengeId}/behavior-batches`, {
    session_id: sessionId, nonce, batch_seq: 0, previous_receipt: null, events,
  });
  if (!sent.body?.accepted)
    console.log(`      배치 거부: ${sent.status} ${sent.text ?? JSON.stringify(sent.body).slice(0, 140)}`);
  return Boolean(sent.body?.accepted);
}

async function driveToBlock() {
  console.log(`  ${GUARD}\n  세션 ${sessionId}\n`);

  for (let round = 1; round <= ROUNDS; round += 1) {
    const made = await call('/api/captcha/challenges', { purpose: 'login', session_id: sessionId });
    if (made.retry) {
      if (made.reason === 'blocked') {
        console.log(`\n  ★차단됨 — 문항 발급이 막혔습니다 (${made.seconds}초).`);
        return true;
      }
      if (made.reason === 'capped') {
        console.log(`   ${round}회: 상한에 걸렸습니다 — ${made.detail}`);
        return false;
      }
      console.log(`   ${round}회: 대기 ${made.seconds}초`);
      await sleep(made.seconds * 1000 + 2000);
      continue;
    }
    if (made.status !== 200 && made.status !== 201) {
      console.log(`   ${round}회: 문항 발급 실패 ${made.status} ${made.text ?? JSON.stringify(made.body ?? '')}`);
      await sleep(1500);
      continue;
    }

    const challenge = made.body;
    // 함정을 **피해서** 하나만 고른다.
    //
    // 함정을 밟으면 그 자리에서 봇 확정으로 끝나 궤적이 채점되지 않는다
    // (`behavior_action_unknown_object`). 채점이 없으면 의심이 안 붙고, 의심이 없으면
    // 차단까지 못 간다. 게다가 채점 실패가 3번 쌓이면 10분간 문항을 아예 못 받는다
    // (`max_telemetry_failures_10m`). 그래서 함정을 반드시 피해야 한다.
    //
    // ⚠️함정은 미리보기 이미지가 **404** 라 시도를 쓰지 않고 걸러진다(0813 실측).
    //   실제 봇도 이렇게 한다 — 별도로 보고했다.
    const real = await withoutHoneypots(challenge.objects);
    if (!real.length) { console.log(`   ${round}회: 고를 객체가 없습니다`); continue; }
    const target = real[0];
    const selected = [target.object_id];
    // 화면(`frac`)과 같은 규칙으로 0~1 을 만든다 — 문항에 따라 히트영역이 이미 0~1 이라
    // 무조건 나누면 좌표가 왼쪽 위로 쏠려 "집은 자리가 객체 밖" 으로 반려된다(0813 실측).
    const unit = (value, dimension) => (value > 1 && dimension > 0 ? value / dimension : value);
    const cx = unit(target.hit_region[0], challenge.width)
      + unit(target.hit_region[2], challenge.width) / 2;
    const cy = unit(target.hit_region[1], challenge.height)
      + unit(target.hit_region[3], challenge.height) / 2;
    const accepted = await sendTrace(challenge.challenge_id, challenge.behavior_nonce,
      mechanicalTrace(target.object_id, cx, cy, Date.now()));

    const verified = await call(`/api/captcha/challenges/${challenge.challenge_id}/verify`, {
      selected_object_ids: selected,
      session_id: sessionId,
      duration_ms: 800,
      pow_nonce: solvePow(challenge.pow),
      client_signals: { webdriver: true, headlessUA: true, languages: 1, cores: 8 },
    });

    if (verified.retry) {
      if (verified.reason === 'blocked') {
        console.log(`\n  ★${round}회째에 차단됐습니다 (${verified.seconds}초).`);
        return true;
      }
      console.log(`   ${round}회: 대기 ${verified.seconds}초`);
      await sleep(verified.seconds * 1000 + 2000);
      continue;
    }
    console.log(`   ${round}회: 궤적 ${accepted ? '접수' : '거부'} · ` +
      `${verified.status} ${verified.body?.success ? '정답' : '오답'}` +
      `${verified.body?.risk_level ? ` · 위험도 ${verified.body.risk_level}` : ''}` +
      `${verified.text ? ` ${verified.text}` : ''}`);
    await sleep(600);
  }

  console.log(`\n  ${ROUNDS}회 동안 차단이 안 됐습니다.`);
  return false;
}

/** 차단된 세션을 브라우저에 심고 로그인 화면을 연다 — 위젯이 차단 화면을 그린다. */
async function openBlockedScreen() {
  const { createRequire } = await import('node:module');
  const { chromium } = createRequire(import.meta.url)('playwright');
  const browser = await chromium.launch({ headless: !process.argv.includes('--headed') });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    ...(VIDEO_DIR ? { recordVideo: { dir: VIDEO_DIR, size: { width: 1280, height: 900 } } } : {}),
  });
  const page = await context.newPage();
  // 화면이 뜨기 **전에** 심어야 한다. 위젯은 첫 렌더에서 세션을 읽고, 없으면 새로 만든다.
  await page.addInitScript(
    (id) => sessionStorage.setItem('catchap-guard-session', id), sessionId);

  // 화면이 실제로 받는 것을 찍는다. "차단" 과 "대기" 는 429 의 헤더로만 갈리므로
  // 응답 본문만 봐서는 왜 대기 화면이 나왔는지 알 수 없다.
  page.on('response', (r) => {
    if (!/captcha\/challenges/.test(r.url())) return;
    console.log(`      ← ${r.status()} Retry-After=${r.headers()['retry-after'] ?? '-'} ` +
      `사유=${r.headers()['x-captcha-retry-reason'] ?? '-'}`);
  });
  await page.goto(`${SITE}/login`, { waitUntil: 'domcontentloaded' });
  const used = await page.evaluate(() => sessionStorage.getItem('catchap-guard-session'))
    .catch(() => null);
  console.log(`   화면이 쓰는 세션: ${used ?? '(없음)'}`);
  await page.waitForSelector('input[type="password"]', { timeout: 20_000 }).catch(() => {});
  for (let i = 0; i < 6; i += 1) {
    await page.fill('input[name="student_login_id"], input[type="text"]', 'demo-bot',
      { timeout: 4000 }).catch(() => {});
    await page.fill('input[type="password"]', 'wrong-password', { timeout: 4000 }).catch(() => {});
    await page.keyboard.press('Enter');
    await page.waitForTimeout(1800);
  }

  await page.locator('.fc-cooldown').first().waitFor({ timeout: 20_000 }).catch(() => {});
  const shown = (await page.locator('.fc-cooldown').innerText().catch(() => ''))
    .replace(/\s+/g, ' ');
  console.log(`   화면: ${shown || '(차단 화면이 안 보입니다)'}`);
  await page.waitForTimeout(9000);   // 영상에 충분히 남게
  await context.close();
  await browser.close();
  if (VIDEO_DIR) console.log(`   영상: ${VIDEO_DIR}`);
  return shown.includes('차단');
}

const blocked = await driveToBlock();
if (blocked && OPEN) await openBlockedScreen();
console.log(`\n  세션: ${sessionId}`);
process.exit(blocked ? 0 : 1);
