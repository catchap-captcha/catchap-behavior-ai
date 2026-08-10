# 승격 심사 현황 — 2026-08-10

조성원 · 이 문서는 승격을 주장하지 않는다. **무엇을 어떤 조건에서 쟀는지**와 **아직 재지 못한 것**을 적는다.

## 0. 하루 동안 결론이 세 번 뒤집혔다

아침에 최악 계열 100%, 낮에 23.8%, 저녁에 "전 계열이 어느 층에서든 막힘" 이 됐다.
숫자가 좋아진 것이 아니라 **내가 잘못 재고 있었던 것을 세 번 고쳤다.** 그 경위를 §5 에 남긴다.
이 문서를 읽는 사람은 §5 를 먼저 읽는 편이 낫다.

## 1. 모델

```
tools/run_formal_two_view_fusion.py  (fold ×5 → assemble → fit-lockbox-candidate)
학습 39,534행    기존 23,547 + 표면 맞는 봇 6,000 + 사람
임계값 0.012876  참가자별 오탐 1% 정책 (max_human_frr=0.01, per_participant)
밀도 거부권      학습 사람 밀도 하위 2%
```

**바뀐 것은 학습 데이터 하나다.** `main_captcha_bots` 12,000건이 2026-08-06 에 만들어져
있었는데 학습에 연결된 적이 없었다. 배포 모델의 기반은 07-22 로, 이 데이터가 생기기
전이다. 변환 도구가 없어 특징 행이 되지 않았을 뿐이다.

## 2. 계열별 결과

```
계열                          통과율   기준    판정   어느 층에서 막히나
adversarial_replay_external     0.4%   ≤5%    통과   궤적 모델
adversarial_replay_holdout      0.4%   ≤5%    통과   궤적 모델
ml_pca_gmm_external             4.0%   ≤5%    통과   궤적 모델
main_captcha_bots (표면)        1.4%   ≤5%    통과   궤적 모델   ← 넣기 전 67.8%
vae_bots_external               4.8%  ≤10%    통과   궤적 모델   ★미지 계열
playwright_bots                20.7%  ≤10%    미달   ← §3 참조   ★미지 계열
replay_warp_external           21.2%   ≤5%    미달   ← §4 참조
```

사람 오탐 (번들 임계값): `jy 3.4% · p4 0.0% · p5 0.0%`
사람 한 명씩 빼고 전부 다시 맞춘 뒤 그 사람으로 잰 값(LOPO): **최악 5.7% · 평균 2.5%**
(기존 배포 모델은 최악 12.1%.)

## 3. playwright — 궤적 모델이 아니라 자동화 신호 층의 일

`tools/capture_playwright_bots.mjs` 는 `chromium.launch({ headless: true })` 로 띄우고
스텔스 처리가 없다. 같은 실행 조건에서 브라우저가 무엇을 보고하는지 직접 쟀다:

```
webdriver    true    → automation_score +80
headlessUA   true    → automation_score +80
                       합계 160  vs  behavior_block_score 80
```

**궤적을 보기 전에 차단된다.** 캡처 스크립트가 실제 캡차를 거치지 않고 빈 페이지에서
포인터 궤적만 뽑기 때문에 데이터셋의 `client_signals` 가 비어 있고, 그래서 20.7% 라는
숫자는 **그 신호를 뗀 채 궤적만 본 값**이다. 시스템 전체 통과율이 아니다.

⚠️ 다만 `navigator.webdriver` 를 가리는 것은 한 줄이다. **위장한 자동화는 지금 데이터에
없고, 그것이 진짜 위협이다.** 없는 위협을 못 잡는다고 미달로 치는 것도, 있는 위협을
만들어 보지 않은 것도 둘 다 틀렸다.

## 4. replay_warp — 궤적 모델이 아니라 리플레이 층의 일

사람 궤적을 변형한 것이라 낱개 판정에 신호가 없다. 같은 날 조준 구간에서
학습에 **넣고 봐도** 교차검증 AUC 0.516 임을 확인했다.

그리고 기존 홀드아웃으로는 리플레이 탐지를 **시험할 수 없다** — 한 줄에 드래그가
하나뿐이라 비교할 짝이 없다. 0% 는 실패가 아니라 시험이 성립하지 않은 것이었다.

`tools/score_replay_sessions.py` 로 세션 형태를 만들어 다시 쟀다 (사람 오탐 3% 문턱):

```
공격                    회전 불변    DTW (현재 배포)
궤적 1개 재사용            100.0%          0.0%
변형 0.01                   99.8%          0.0%
변형 0.02                   99.6%          0.0%
라이브러리 5개               80.2%          0.0%
```

**현재 배포된 DTW 비교기는 신호가 뒤집혀 있다** — 무고한 세션 0.917, 리플레이 세션
0.834~0.874. 어떤 문턱으로도 못 쓴다.

세 가지 단서가 붙는다.

- **오탐 3% 가 필요하다.** 1% 에서는 문턱이 1.0000 에 붙어 변형된 것은 하나도 안 잡힌다.
- **라이브러리 80.2% 는 방어의 성질이 아니다.** 네 번 뽑아 다섯 중 겹칠 확률 80.8% 와
  거의 같다. 라이브러리 100개면 상한이 5.9% 다. 세션을 넘는 이력이 있어야 오른다.
- 위 숫자는 실제 홀드아웃 파일이 아니라 **사람 궤적으로 재구성한 공격**이다.

## 5. 오늘 내가 틀렸던 것

전부 **방어에 유리한 쪽으로** 틀렸다. 이것이 봉인을 남겨두는 이유다.

| 틀린 보고 | 실제 | 원인 |
|---|---|---|
| 다른 사람끼리 지문 오탐 0.19% | 전부 같은 사람 | 한 사람이 코드 3개를 씀 |
| 강의 완주 기대 시도수 170만 회 | 지표 자체가 무의미 | 강의 캡차는 판정을 쓰지 않고 수집만 한다 |
| 재출제 3회면 봇 통과 1.1% | 52.5% | 봇은 한 번만 뚫으면 된다. `ASR^k` 가 아니라 `1-(1-ASR)^k` |
| 최악 계열 23.8% | 배포 임계값과 무관 | 배포된 0.99995 가 아니라 내가 다시 잡은 동작점에서 잼 |
| 사람 오탐 2.3% | LOPO 로 최악 12.1% | 예산 숫자였지 측정값이 아니었다 |
| 학습 파이프라인이 없다 | `tools/run_formal_two_view_fusion.py` | `training/` 만 보고 `tools/` 를 안 봄 |

마지막 것이 제일 컸다. 도구를 못 찾아 축소판을 직접 만들었고, 그 축소판으로 재서
**"재학습하면 나빠진다"** 는 틀린 결론을 냈다. 정식 도구에는 내 축소판에 없던
`StratifiedGroupKFold`, 참가자별 문턱 정책, 봉인 자동 거부가 다 있었다.

## 6. 아직 재지 못한 것

```
봉인 (sw·ms 190세션)    안 엶. 위 숫자는 전부 봉인 전이다
실사용자 오탐            그림자 모드로만 돌았다. 진짜 오탐은 아무도 모른다
위장한 자동화            webdriver 를 가린 봇. 데이터에 없다
세션을 넘는 리플레이     이력이 세션·60초로 제한돼 있어 시험할 수 없다
```

사람이 5명뿐이고 그중 3명만 오염되지 않은 채 쓸 수 있다. `p5` 한 사람이 LOPO 오탐의
대부분을 만든다.

## 7. 승격에 대한 입장

**이 문서로 승격을 주장하지 않는다.** 두 가지가 걸린다.

**① 기준이 궤적 모델 단독을 재는데 실제 방어는 3층이다.** playwright 는 자동화 신호
층에서, replay_warp 은 리플레이 층에서 막힌다. "전 계열 통과" 라고 쓰려면 기준이
3층 합산을 재는 것이어야 하고, 그 해석은 팀이 정할 일이다.

**② 오탐이 아직 측정되지 않았다.** 근거가 한 사람의 88세션이고 95% 구간이
[0.7%, 9.7%] 다. 다만 승급 캡차 구조(`docs/HANDOFF_BACKEND_ESCALATION_20260729.md`)에서는
오탐의 대가가 "차단" 이 아니라 "캡차 한 번 더" 라, 이 항목은 다른 항목만큼 무겁지 않다.

## 8. 재현

```bash
# 표면 맞는 봇을 특징 행으로
.venv/bin/python tools/build_main_captcha_bot_features.py
cat data/interim/bot_features_v23corr_20260722.jsonl \
    data/interim/bot_features_main_captcha_20260810.jsonl \
    > data/interim/bot_features_plus_surface_20260810.jsonl

# 정식 학습 (fold 5개 → assemble → fit)
PYTHONPATH=. .venv/bin/python tools/run_formal_two_view_fusion.py fold \
  --human-features data/interim/human_features_v23corr_plus_collection_20260806.jsonl \
  --bot-features data/interim/bot_features_plus_surface_20260810.jsonl \
  --work-dir reports/surface_aware_formal_20260810/work --fold 0 \
  --report reports/surface_aware_formal_20260810/work/fold_0.json --all-input-development
# ... fold 1~4 동일 ...
PYTHONPATH=. .venv/bin/python tools/run_formal_two_view_fusion.py assemble \
  --human-features ... --bot-features ... \
  --work-dir reports/surface_aware_formal_20260810/work \
  --report reports/surface_aware_formal_20260810/group_threshold_calibration.json \
  --max-human-frr 0.01 --human-frr-policy per_participant --all-input-development
PYTHONPATH=. .venv/bin/python tools/run_formal_two_view_fusion.py fit-lockbox-candidate \
  --human-features ... --bot-features ... \
  --model-output models/candidate/surface_aware_formal_20260810/two_view_fusion.joblib \
  --model-version surface_aware_formal_20260810 \
  --calibration reports/surface_aware_formal_20260810/group_threshold_calibration.json \
  --report reports/surface_aware_formal_20260810/fit_candidate.json \
  --max-human-frr 0.01 --human-frr-policy per_participant --all-input-development

# 밀도 거부권
.venv/bin/python tools/train_density_veto.py \
  --base models/candidate/surface_aware_formal_20260810 --version surface_aware_veto_20260810

# 측정
.venv/bin/python tools/score_veto_holdouts.py --model models/candidate/surface_aware_veto_20260810
.venv/bin/python tools/score_replay_sessions.py
```

모델·특징 파일·리포트는 저장소에서 제외된다(`.gitignore`). 위 순서로 다시 만들 수 있다.
