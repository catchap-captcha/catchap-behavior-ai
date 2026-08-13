/**
 * 시연영상용 — **화면을 실제로 조작하는** 봇이 캡차를 풀다 차단당하는 장면을 녹화한다.
 *
 * `demo_block_direct.mjs` 와 무엇이 다른가
 * ----------------------------------------
 * 그쪽은 서버에 직접 요청을 던진다. 빠르고 확실하지만 **화면에 아무것도 안 나온다** —
 * 영상에는 차단된 뒤 화면만 담긴다. 이 도구는 진짜 마우스로 끌어다 놓아서
 * "봇이 푸는 모습 → 계속 틀림 → 차단" 이 한 화면에 이어진다.
 *
 * 어떻게 잡히나
 * -------------
 * 끌 때 **중간 경로를 안 남긴다**(집고 바로 놓는다). 매번 같은 모양이라 재생 탐지가
 * 잡는다 — 0813 실측으로 `dtw_similar_trace`(중간) → `exact_trace_fingerprint`(높음)
 * 순서로 붙었다. 등속 직선으로 곱게 끄는 봇은 **안 잡힌다**(사람 점수 0.9999). 그건
 * `client_signals` 의 webdriver 신호를 살려야 잡히는데 아직 안 쓰고 있다.
 *
 * ⚠️함정을 DB 로 피한다 — 이건 봇이 못 하는 일이다
 * ------------------------------------------------
 * 함정을 밟으면 궤적이 객체에 안 묶여 채점 자체가 안 되고(`behavior_action_unknown_object`),
 * 그게 3번 쌓이면 10분간 문항을 못 받는다(`max_telemetry_failures_10m`). 그러면 차단까지
 * 못 간다.
 *
 * 예전에는 함정 미리보기가 404 라 밖에서도 걸러졌지만, 그 구멍을 오늘 막았다
 * (catchap-captcha#39). 그래서 **이 도구만** 우리 DB 를 읽어 함정을 피한다.
 * 실제 공격자는 못 하는 일이고, 영상에 담기는 화면·판정은 전부 진짜다.
 *
 *   node tools/demo_video_bot.mjs --video ./demo_video --pw <비밀번호 파일>
 */
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const argument = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
};

const SITE = argument('--site', 'https://www.catchap5.com');
const LOGIN_ID = argument('--id', `demo-bot-${Date.now()}`);
const ROUNDS = Number(argument('--rounds', '14'));
const VIDEO_DIR = argument('--video', null);
const PASSWORD_FILE = argument('--pw', null);
const HEADED = process.argv.includes('--headed');
/** `teleport` 는 집고 바로 놓는다. `uniform` 은 일정 간격·같은 보폭으로 곧게 끈다. */
const DRAG = argument('--drag', 'uniform');

/**
 * 이 문항에서 **오답이 확실하면서 함정도 아닌** 객체 하나. DB 를 읽는다 — 위 ⚠️ 참고.
 *
 * 왜 아무 객체나 집으면 안 되나
 *   · 함정을 집으면 궤적이 채점 자체가 안 되고, 3번이면 10분 잠긴다
 *   · 정답을 집으면 그 판은 오답으로 안 세어져 차단 조건(오답 5회)이 안 채워진다
 *     (0813 실측: 10판 돌려 오답이 2번뿐이라 차단까지 못 갔다)
 *
 * `decoy` 는 사진에 실제로 그려진 물건이지만 정답이 아닌 것이다. 궤적은 정상으로
 * 채점되고 답은 반드시 틀린다 — 시연에 필요한 조합이 정확히 이것이다.
 */
function wrongAnswerObject(challengeId) {
  if (!PASSWORD_FILE) return null;
  try {
    const out = execFileSync('.venv/bin/python', [
      'tools/q.py', '--db', 'catchap_captcha', '--password-file', PASSWORD_FILE, '--raw',
      '--sql', `SELECT m.temporary_object_id FROM captcha_challenge_objects m
                JOIN captcha_objects o ON o.id=m.object_id
                WHERE m.challenge_id='${challengeId}' AND o.role='decoy' LIMIT 1;`,
    ], { encoding: 'utf8' });
    return (out.match(/tmp_[A-Za-z0-9_-]+/g) ?? [])[0] ?? null;
  } catch {
    return null;
  }
}

/**
 * 순간이동으로 끈다 — 집고, 바로 놓는다. 중간 점이 없다.
 *
 * 사람 손은 끄는 동안 계속 미세하게 꺾이고 멈칫한다(방향 전환 AUC 0.87 · 간격 불규칙
 * 0.86). 여기서는 그게 통째로 없고, 매번 모양이 같아서 재생으로도 걸린다.
 */
async function teleportDrag(page, from, to) {
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  if (DRAG === 'uniform') {
    // 같은 보폭으로 곧게 끈다 — 사람 손이 못 내는 모양이다(방향 전환 AUC 0.87 ·
    // 등속 0.86). 순간이동은 점이 하나뿐이라 모델이 볼 것이 없어 오히려 사람처럼
    // 보인다(0813 실측: 사람 점수 0.9999, 위험도 35.0 으로 통과).
    await page.waitForTimeout(80);
    await page.mouse.move(to.x, to.y, { steps: 24 });
    await page.waitForTimeout(80);
    await page.mouse.up();
    return;
  }
  // 집고 놓는 사이에 아주 짧게 쉰다. 한 틱 만에 끝내면 화면이 집은 것을 기록하기 전에
  // 놓아버려서 아무것도 안 담긴다(0813 실측: 세 판에 한 판꼴로 "고름 0개").
  // 경로는 여전히 없다 — 중간 점을 안 찍는다.
  await page.waitForTimeout(80);
  await page.mouse.move(to.x, to.y);
  await page.waitForTimeout(80);
  await page.mouse.up();
}

const main = async () => {
  const browser = await chromium.launch({ headless: !HEADED });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    ...(VIDEO_DIR ? { recordVideo: { dir: VIDEO_DIR, size: { width: 1280, height: 900 } } } : {}),
  });
  const page = await context.newPage();

  // ★세션 번호에 `test` 를 박아 둔다 — 안 그러면 이 봇의 기록을 **실사용자와 구분할 수
  // 없다.** 위젯이 만드는 번호(`guard-<시각>-<무작위>`)와 글자 모양이 똑같기 때문이다.
  //
  // 그러면 나중에 오탐율이나 재시도 비율을 잴 때 우리 봇이 통계에 섞여 들어간다.
  // 0813 에 실제로 그랬다 — 로그인 캡차 채점 99건 중 대부분이 이 봇인데 표에는
  // "거절률 27.3%" 로만 보였다. 사용자가 적을수록 몇 판이 통계를 통째로 흔든다.
  //
  // 화면이 뜨기 전에 심어야 한다. 위젯은 첫 렌더에서 읽고, 없으면 새로 만든다.
  await page.addInitScript((id) => {
    try { sessionStorage.setItem('catchap-guard-session', id); } catch { /* 무시 */ }
  }, `guard-test-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`);

  // 문항이 올 때마다 받아 둔다 — 화면에는 challenge_id 가 안 적혀 있어서, 어느 객체가
  // 함정인지 물어보려면 응답을 가로채는 수밖에 없다.
  let current = null;
  let acted = null;   // 이미 손댄 문항 — 같은 것을 두 번 풀지 않게
  page.on('response', async (response) => {
    if (!/\/api\/captcha\/challenges$/.test(response.url())) return;
    const body = await response.json().catch(() => null);
    if (body?.challenge_id) current = body;
  });

  const seen = new Set();
  const report = async () => {
    const text = await page.locator('.fc-instruction, .fc-cooldown').allTextContents()
      .catch(() => []);
    for (const line of text.map((s) => s.trim()).filter(Boolean)) {
      if (!seen.has(line)) { seen.add(line); console.log(`   화면: ${line}`); }
    }
    return text.join(' ');
  };

  console.log(`  ${SITE} · 아이디 ${LOGIN_ID}\n`);
  await page.goto(`${SITE}/login`, { waitUntil: 'domcontentloaded' });
  // 로그인 화면은 늦게 그려진다. 안 기다리면 첫 판이 비밀번호 칸을 못 찾아 통째로 샌다.
  await page.waitForSelector('input[type="password"]', { timeout: 20_000 }).catch(() => {});

  for (let round = 1; round <= ROUNDS; round += 1) {
    const cooling = page.locator('.fc-cooldown');
    if (await cooling.count()) {
      const text = (await cooling.innerText().catch(() => '')).replace(/\s+/g, ' ');
      if (text.includes('차단')) {
        console.log(`\n  ★${round}회째에 차단 화면이 떴습니다 — ${text}`);
        await page.waitForTimeout(10_000);   // 영상에 충분히 남게
        await context.close(); await browser.close();
        if (VIDEO_DIR) console.log(`   영상: ${VIDEO_DIR}`);
        return 0;
      }
      console.log(`   대기: ${text}`);
      await page.locator('.fc-object').first().waitFor({ timeout: 90_000 }).catch(() => {});
      continue;
    }

    await page.locator('.fc-object').first().waitFor({ timeout: 5000 }).catch(() => {});
    const objects = page.locator('.fc-object');
    const count = await objects.count();
    if (!count) {
      // 캡차는 로그인 **6번째** 실패에서 뜬다. 없는 계정을 쓰므로 실제 사용자 계정은
      // 안 건드린다. 채우기 제한을 짧게 둔다 — 기본 30초라 한 번 놓치면 판이 날아간다.
      await page.fill('input[name="student_login_id"], input[type="text"]', LOGIN_ID,
        { timeout: 4000 }).catch(() => {});
      await page.fill('input[type="password"]', 'wrong-password', { timeout: 4000 })
        .catch(() => {});
      await page.keyboard.press('Enter');
      await page.waitForTimeout(1800);
      continue;
    }

    // ★새 문항이 도착할 때까지 기다린다.
    //
    // 가로챈 문항이 화면보다 뒤처지면 지난 판의 객체 id 로 자리를 찾게 되고, 그 id 는
    // 이번 문항에 없으니 엉뚱한 자리를 집는다 — 0813 에 그래서 함정을 밟았고 그 뒤
    // 판이 통째로 헛돌았다. 자리를 못 찾았을 때 조용히 0번을 집던 것이 원인이었다.
    for (let waited = 0; waited < 40 && (!current || current.challenge_id === acted); waited += 1)
      await page.waitForTimeout(200);
    if (!current || current.challenge_id === acted) {
      console.log(`   ${round}회: 새 문항이 안 옵니다 — 건너뜁니다`);
      await page.waitForTimeout(1500);
      continue;
    }
    acted = current.challenge_id;

    // ★화면 순서는 응답 순서와 **다르다.** 위젯은 큰 객체부터 그린다 — 작은 것이 큰 것에
    // 가려 못 집히는 일을 막으려고 넓이 내림차순으로 정렬한다(`stageObjects`).
    // 응답 순서로 세면 엉뚱한 것을 집는다(0813 실측: 그래서 함정을 밟았다).
    const target = wrongAnswerObject(current.challenge_id);
    const drawn = current.objects.slice()
      .sort((a, b) => b.hit_region[2] * b.hit_region[3] - a.hit_region[2] * a.hit_region[3]);
    const pickAt = drawn.findIndex((o) => o.object_id === target);
    if (pickAt < 0) {
      // 아무 데나 집으면 함정을 밟는다. 그러면 궤적이 채점 자체가 안 되고 세 번이면
      // 10분 잠긴다 — 차라리 이 판을 버린다.
      console.log(`   ${round}회: 오답용 객체를 못 찾았습니다 — 이 판은 건너뜁니다`);
      await page.waitForTimeout(1500);
      continue;
    }

    // 안 붙으면 다시 끈다. 한 판이 헛돌면 오답이 안 쌓여 차단까지 못 간다.
    let picked = '0개';
    for (let attempt = 1; attempt <= 3 && picked.startsWith('0'); attempt += 1) {
      const pick = await objects.nth(pickAt).boundingBox();
      const drop = await page.locator('.fc-drop').boundingBox();
      if (!pick || !drop) break;
      await teleportDrag(page,
        { x: pick.x + pick.width / 2, y: pick.y + pick.height / 2 },
        { x: drop.x + drop.width / 2, y: drop.y + drop.height / 2 });
      await page.waitForTimeout(400);
      picked = await page.locator('.fc-count strong').innerText().catch(() => '0개');
    }
    console.log(`   ${round}회: 객체 ${count}개 중 ${pickAt + 1}번째(오답) · 고름 ${picked}`);
    if (!picked.startsWith('0'))
      await page.locator('button:has-text("선택 완료")').click({ timeout: 3000 }).catch(() => {});

    await page.waitForTimeout(2600);
    await report();
  }

  console.log(`\n  ${ROUNDS}회 동안 차단 화면이 안 떴습니다.`);
  await context.close(); await browser.close();
  if (VIDEO_DIR) console.log(`   영상: ${VIDEO_DIR}`);
  return 1;
};

main().then((code) => process.exit(code));
