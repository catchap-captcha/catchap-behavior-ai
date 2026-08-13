/**
 * 시연용 — 봇처럼 캡차를 풀어 「차단」 화면까지 몰고 간다.
 *
 * 왜 이 도구가 필요한가
 * ---------------------
 * 차단 조건은 `의심(medium/high) + 5회 오답` 이다. 사람이 일부러 틀려도 차단 화면이
 * 안 나온다 — 실사용 채점이 평균 0.9549 라 의심이 안 붙고, 대기(60초)까지만 간다.
 * 그림으로도 "사람이 일부러 틀리는" 것보다 "봇이 잡히는" 쪽이 맞다.
 *
 * 무엇을 하나
 * -----------
 *   ① 로그인을 5회 실패시켜 캡차를 부른다 (없는 계정을 쓴다)
 *   ② 캡차가 뜨면 기계적으로 끌어다 놓고 제출한다 — 등속 직선, 일정 간격
 *   ③ 오답이 쌓이고 의심이 붙으면 차단 화면이 뜬다
 *
 * ②의 움직임을 일부러 기계처럼 만든다. 사람 손은 계속 미세하게 꺾이고 멈칫하는데
 * (0813 실측: 방향 전환 AUC 0.87 · 간격 불규칙 0.86), 여기서는 그걸 안 한다.
 *
 *   node tools/demo_blocked_bot.mjs --id demo-bot-1 --headed
 *
 * ⚠️실서비스를 두드린다. 없는 계정을 쓰므로 실제 사용자 계정은 건드리지 않지만,
 *   그 아이디의 실패 카운터는 남는다. 시연 전에 한 번 돌려보는 용도다.
 */
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const argument = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
};

const SITE = argument('--site', 'https://www.catchap5.com');
const LOGIN_ID = argument('--id', `demo-bot-${Date.now()}`);
const ROUNDS = Number(argument('--rounds', '12'));
const HEADED = process.argv.includes('--headed');
/** 녹화 폴더. 주면 Playwright 가 화면을 webm 으로 남긴다 — 시연 영상이 이것이다. */
const VIDEO_DIR = argument('--video', null);

/**
 * 순간이동으로 끈다 — 집고, 바로 놓는다. 중간 경로가 없다.
 *
 * 왜 이렇게 하나: 등속 직선으로 끌어봤더니 우리 모델이 **사람 점수 0.9999** 를 줬다
 * (2026-08-13 실측). Playwright 의 `mouse.move` 는 브라우저가 진짜 포인터 이벤트로
 * 만들어 주고, 우리 모델은 좌표·간격만 보기 때문이다. 그런 봇을 잡으려면
 * `client_signals` 의 webdriver 신호를 살려야 하는데 지금은 그게 비어 있다.
 *
 * 순간이동은 궤적을 거의 안 남긴다. 품질 검사가 "움직임이 없는 제출" 로 보고 위험도를
 * 올린다. 실제로 흔한 조잡한 봇의 모습이기도 하다.
 */
async function teleportDrag(page, from, to) {
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  await page.mouse.move(to.x, to.y);   // 중간 점 없음
  await page.mouse.up();
}

const main = async () => {
  const browser = await chromium.launch({ headless: !HEADED });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    ...(VIDEO_DIR ? { recordVideo: { dir: VIDEO_DIR, size: { width: 1280, height: 900 } } } : {}),
  });
  const page = await context.newPage();
  const seen = new Set();

  // 캡차 서버 응답을 그대로 찍는다. 화면의 "확인 처리 중 오류" 만 보면 무엇이 거부됐는지
  // 알 수 없다 — 문항 발급(201)과 채점(200/4xx)을 구분해야 어디서 막혔는지 보인다.
  page.on('response', (r) => {
    const url = r.url();
    if (!/captcha/.test(url)) return;
    if (r.status() >= 400 || /verify|attempt/.test(url))
      console.log(`      ← ${r.status()} ${url.replace(/^https?:\/\/[^/]+/, '')}`);
  });

  // 화면에 뜨는 문구를 그대로 찍는다 — 시연에서 무엇이 보이는지가 이 도구의 결과다.
  const report = async () => {
    const text = await page.locator('.fc-instruction, .fc-cooldown').allTextContents()
      .catch(() => []);
    for (const t of text.map((s) => s.trim()).filter(Boolean)) {
      if (!seen.has(t)) { seen.add(t); console.log(`   화면: ${t}`); }
    }
    return text.join(' ');
  };

  console.log(`  ${SITE} · 아이디 ${LOGIN_ID}\n`);
  await page.goto(`${SITE}/login`, { waitUntil: 'domcontentloaded' });
  // 로그인 화면은 별도 묶음으로 늦게 그려진다. 안 기다리면 첫 라운드가 비밀번호 칸을
  // 못 찾아 기본 시간(30초)을 통째로 날린다 — 0813 실측.
  await page.waitForSelector('input[type="password"]', { timeout: 20_000 })
    .catch(() => console.log('   (비밀번호 칸을 못 찾았습니다)'));

  for (let round = 1; round <= ROUNDS; round += 1) {
    // ① 대기·차단 화면이면 그것부터 처리한다. 안 그러면 이 화면에 문제가 없다고 보고
    //    로그인 칸을 찾으러 가서, 남은 라운드를 전부 헛돌린다(0813 실측: 18회 중 3회만 제출).
    const cooling = page.locator('.fc-cooldown');
    if (await cooling.count()) {
      const text = (await cooling.innerText().catch(() => '')).replace(/\s+/g, ' ');
      if (text.includes('차단')) {
        console.log(`\n  ★${round}회째에 차단 화면이 떴습니다 — ${text}`);
        await page.waitForTimeout(9000);   // 차단 화면이 영상에 충분히 남게
        await context.close(); await browser.close();
        if (VIDEO_DIR) console.log(`   영상: ${VIDEO_DIR}`);
        return 0;
      }
      // 그냥 대기다. 끝날 때까지 기다린다 — 사다리가 60초까지 올라간다.
      console.log(`   대기: ${text}`);
      await page.locator('.fc-object').first().waitFor({ timeout: 90_000 }).catch(() => {});
      continue;
    }

    await page.locator('.fc-object').first().waitFor({ timeout: 5000 }).catch(() => {});
    const objects = page.locator('.fc-object');
    const count = await objects.count();
    if (count) {
      // ② **전부** 고른다. 하나만 고르면 정답을 맞혀버린다 — 물체가 둘뿐인 문항이
      //    1,142개라 찍어도 곧잘 맞는다(0813 실측: 3회 중 2회 정답). 전부 고르면
      //    함정(honeypot)이 반드시 섞여 들어가 항상 틀린다. 오답이 쌓여야 차단까지 간다.
      //    키보드로 고른다 — 화면이 접근성용으로 열어둔 길이고(Tab 이동 후 Enter),
      //    마우스를 안 쓰니 궤적이 안 남는다. 순간이동 드래그는 잘 안 잡혔다.
      for (let i = 0; i < count; i += 1) {
        await objects.nth(i).focus().catch(() => {});
        await page.keyboard.press('Enter').catch(() => {});
        await page.waitForTimeout(120);
      }
      // 고른 개수를 그대로 찍는다 — 0 이면 제출 단추가 잠겨 있어 그 라운드는 통째로
      // 헛돈다. 어디서 막혔는지 이 숫자 없이는 알 수 없다.
      const picked = await page.locator('.fc-count strong').innerText().catch(() => '?');
      const submit = page.locator('button:has-text("선택 완료")');
      const locked = await submit.isDisabled().catch(() => true);
      console.log(`   ${round}회: 객체 ${count}개 · 고름 ${picked} · 제출단추 ${locked ? '잠김' : '열림'}`);
      await submit.click({ timeout: 3000 }).catch(() => {});
    } else {
      // 캡차는 **6번째** 로그인 실패에서 뜬다(0813 실측). 없는 계정을 쓰므로 실제
      // 사용자 계정은 안 건드린다. 채우기 제한 시간을 짧게 둔다 — 기본 30초라
      // 한 번만 놓쳐도 라운드가 통째로 날아간다.
      await page.fill('input[name="student_login_id"], input[type="text"]', LOGIN_ID,
        { timeout: 4000 }).catch(() => {});
      await page.fill('input[type="password"]', 'wrong-password', { timeout: 4000 })
        .catch(() => {});
      await page.keyboard.press('Enter');
    }

    await page.waitForTimeout(2600);
    await report();
  }

  console.log(`\n  ${ROUNDS}회 동안 차단 화면이 안 떴습니다.`);
  console.log('  의심 점수가 안 붙었을 수 있습니다 — DB 에서 risk_level 을 확인하십시오.');
  await context.close();
  await browser.close();
  if (VIDEO_DIR) console.log(`   영상: ${VIDEO_DIR}`);
  return 1;
};

main().then((code) => process.exit(code));
