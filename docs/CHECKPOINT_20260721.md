# CatChap 행동 데이터 AI 점검 결과 - 2026-07-21

## 종합 판정

`실패 / 운영 모델로 적용하지 않음`

로컬 환경에서 데이터 정제, 모델 학습, 실제 브라우저 봇 검증,
사용자 그룹 5-fold 임계값 보정, 재생 공격 결합 판정, Feature v2 실험을
완료했다. 모든 오탐률과 봇 통과율 기준을 만족한 모델은 없었으며,
유효한 행동 데이터가 있는 사용자도 46명뿐이다. 따라서 운영 모델은
변경하지 않았다.

## 행동 데이터 현황

MySQL 원본 데이터는 `읽기 전용 트랜잭션`으로 조회했으며 마지막에
`ROLLBACK` 처리했다. 데이터베이스 식별자는 스냅샷 전용 HMAC 가명값으로
변환했고, HMAC 키는 저장하지 않았다.

| 항목 | 개수 |
|---|---:|
| 원본 행동 요약 데이터 | 21,678 |
| 사용 가능한 포인터 행동 궤적 | 20,067 |
| 확정된 정상 사용자 학습 데이터 | 20,066 |
| 사용자와 연결된 행동 데이터 | 19,177 |
| 익명 행동 데이터 | 889 |
| 유효한 행동 데이터가 있는 사용자 | 46명 |
| 제외된 데이터 | 1,612 |
| 행동 궤적 누락 또는 형식 오류 | 94 |
| 포인터 좌표가 4개 미만인 데이터 | 1,517 |
| 원본에서 명시적으로 제외된 데이터 | 1 |

운영 적용 기준인 사용자 100명을 충족하지 못했다. 현재 원본 데이터의
`actor_band`도 모두 성인으로 기록되어 있어 연령대 다양성은 검증되지 않았다.

## 봇 데이터 구성

| 데이터 세트 | 용도 | 개수 |
|---|---|---:|
| 합성 봇 10종 | 학습·검증·테스트 및 봇 유형별 홀드아웃 | 10,000 |
| Chrome·Playwright 봇 3종 | 외부 홀드아웃 전용 | 90 |

합성 봇 유형은 `straight`, `accel`, `jitter`, `bezier_curve`, `stop_go`,
`overshoot_correct`, `waypoint`, `random_timing`, `frame_quantized`,
`replay_warp`이다. 브라우저 봇은 실제 로컬 Chrome 이벤트 루프에서
직선형, 베지어 곡선형, 정지·이동 반복형 궤적을 수집했다. 브라우저 봇
데이터는 원격 데이터베이스에 저장하지 않았다.

## 모델 및 Feature 비교

모든 실험은 동일한 사용자 그룹 분할과 시드 값을 사용했다. Feature v1은
행동 한 건에서 특징 29개를 추출한다. Feature v2는 정규화된 궤적 형태,
방향 전환, 세부 움직임, 엔트로피 특징 15개를 추가해 총 44개를 사용한다.

| Feature | 모델 | 정상 사용자 오탐률(FRR) | 기존 봇 통과율(ASR) | 재생 봇 통과율 | 브라우저 봇 최악 통과율 | 판정 |
|---|---|---:|---:|---:|---:|---|
| v1 | RandomForest | 6.32% | 8.69% | 99.9% | 0% | 실패 |
| v1 | ExtraTrees | 4.29% | 10.03% | 100% | 100% | 실패 |
| v1 | XGBoost | 4.89% | 8.69% | 100% | 0% | 실패 |
| v1 | LightGBM | 5.85% | 8.75% | 100% | 0% | 실패 |
| v2 | RandomForest | 4.45% | 9.48% | 100% | 0% | 실패 |
| v2 | ExtraTrees | 2.96% | 10.15% | 100% | 100% | 실패 |
| v2 | XGBoost | 5.40% | 8.88% | 100% | 0% | 실패 |
| v2 | LightGBM | 3.45% | 9.30% | 100% | 0% | 실패 |

Feature v2를 사용했을 때 가장 낮은 정상 사용자 오탐률은 2.96%였다.
하지만 해당 모델도 기존 봇, 미학습 재생 봇, 브라우저 자동화 봇,
홀드아웃 정상 사용자 기준을 모두 통과하지 못했다. 따라서 Feature v2는
실험 단계로 유지하며 기존 Feature를 대체하지 않는다.

## 사용자 그룹 5-fold 임계값 보정

한 번의 validation 사용자 구성에 임계값이 좌우되지 않도록 개발 데이터의
연결 사용자와 봇 생성 그룹을 5개 fold로 분리했다. 각 행은 자신이 포함되지
않은 fold 모델로 점수화했으며, 모든 fold에서 정상 사용자 오탐률이 3% 이하인
공통 임계값을 선택했다. 익명 사용자 889건은 사용자 그룹을 확인할 수 없어
임계값 보정에서는 제외했지만 최종 개발 모델 학습에는 사용했다. test 사용자
7명과 test 데이터 3,887건은 임계값 결정에 사용하지 않았다.

| 모델 | 공통 임계값 | OOF 정상 사용자 오탐률 | 최악 fold 오탐률 | Test 오탐률 | Test 기존 봇 통과율 |
|---|---:|---:|---:|---:|---:|
| RandomForest | 0.476667 | 1.32% | 2.90% | 0.27% | 9.35% |
| ExtraTrees | 0.510000 | 1.90% | 2.96% | 1.04% | 9.53% |
| XGBoost | 0.166149 | 0.98% | 2.98% | 0.00% | 9.41% |
| LightGBM | 0.186806 | 1.15% | 2.98% | 0.09% | 9.23% |

교차 검증으로 test 정상 사용자 오탐률은 낮아졌지만 모델 단독의 기존 봇
통과율은 여전히 약 9%이므로 단독 모델은 보안 기준을 통과하지 못했다.

## 재생 공격 방어

재생 공격 탐지 계층은 행동 데이터 한 건만 판별하는 ML 모델과 별도로 작동한다.

| 탐지 방법 | 결과 | 정상 사용자 오탐 대체 지표 | 판정 |
|---|---:|---:|---|
| 동일 궤적 Fingerprint | 재생 공격 통과 0/1,000 | 중복 데이터 비율 0.08% | 통과 |
| 원본과 변형 궤적의 DTW 비교 | 재생 공격 통과 0/1,000 | 무관한 궤적 쌍 오탐률 1.00% | 통과 |

DTW는 공격에 사용된 원본 궤적이 비교 기록에 존재할 때 변형 재사용을
찾을 수 있다. 이 결과가 기록에 없는 새로운 봇까지 탐지한다는 의미는 아니다.
`replay_warp` 유형을 학습에서 완전히 제외하면 1차 ML 모델의 재생 봇
통과율은 여전히 약 100%다.

### 결합 판정 재검증

최종 판정은 `ML이 봇으로 판정`, `동일 Fingerprint`, `DTW 임계값 이상`,
`60초 내 과도한 시도` 중 하나라도 발생하면 차단 또는 강화 CAPTCHA를
요구하는 OR 방식으로 구성했다. 임계값은 개발 데이터에서만 정하고 test는
최종 측정에만 사용했다.

- DTW 임계값: `0.996693`
- DTW 개발 사용자 오탐률: `1.00%`
- DTW 개발 replay 탐지율: `93.89%`
- 세션 기준: 60초 내 이전 시도 55건 이상
- 세션 기준 개발 사용자 오탐률: `0.32%`
- DTW 비교 이력: 동일 사용자 최근 5개 궤적

처음 사용한 DTW 임계값 `0.970403`은 서로 무관한 사용자 쌍에서는 오탐률이
1%였지만, 실제 동일 사용자의 반복 행동에 적용하면 test 사용자의 약 22%를
오탐했다. 따라서 이 임계값은 폐기하고 실제 시간순 이력으로 다시 보정했다.

| 결합 검증 항목 | 결과 | 기준 | 판정 |
|---|---:|---:|---|
| Test 정상 사용자 추가 오탐 | 0건 | 전체 FRR 3% 이하 | 통과 |
| 알려진 봇 전체 ASR | 0.84% | 5% 이하 | 통과 |
| 정확히 복제한 replay ASR | 0.00% | 1% 이하 | 통과 |
| `replay_warp` 최악 ASR | 8.75% | 5% 이하 | 실패 |
| 60초 봇 버스트 ASR | 0.80% | 5% 이하 | 통과 |
| 브라우저 봇 최악 ASR | XGBoost·LightGBM 0%, 나머지 100% | 10% 이하 | 모델별 판정 |

결합 계층은 평균 봇 통과율을 크게 낮췄지만, 단발성 `replay_warp` 공격은
여전히 8.75%가 통과했다. 따라서 XGBoost와 LightGBM도 운영 모델로 올리지
않으며, DTW 후보 검색 개선과 실제 세션 Shadow Mode 검증이 추가로 필요하다.

### 추가 Replay 신호 오프라인 실험

8.75%로 남은 `replay_warp`를 줄이기 위해 아핀 잔차, 방향, 곡률,
정규화 시간, 속도 프로파일, 다중 해상도 형태, 이벤트 개수 비율을 DTW와
함께 사용하는 오프라인 LogisticRegression 메타 판정기를 추가했다.

개발 참가자 39명의 정상 시도 16,958건과 최근 이력 쌍 84,205개,
replay-source 쌍 835개를 사용했다. 참가자 그룹 5-fold OOF에서 공통 임계값
`0.857185`를 선택했으며 OOF 정상 사용자 오탐률은 0.12%, replay ASR은
0.12%, 최악 fold 정상 사용자 오탐률은 0.41%였다.

임계값 확정 후 보지 않은 test 참가자 7명의 서로 다른 원본 1,000건에 새
seed로 `replay_warp`를 적용했다.

| 독립 Replay Holdout 항목 | 결과 | 기준 | 판정 |
|---|---:|---:|---|
| XGBoost 단독 Replay ASR | 95.30% | 참고 | 실패 |
| 기존 DTW 단독 Replay ASR | 6.70% | 5% 이하 | 실패 |
| 고급 Replay 신호 ASR | 0/1,000, 0.00% | 5% 이하 | 통과 |
| 전체 결합 Replay ASR | 0/1,000, 0.00% | 5% 이하 | 통과 |
| Test 정상 사용자 전체 FRR | 0/2,219, 0.00% | 3% 이하 | 통과 |

0/1,000 Replay ASR의 95% Wilson 신뢰구간 상한은 약 0.38%다. 다만 같은
`replay_warp` 생성 방식의 새 seed를 사용한 결과이며, 시간 프로파일 신호의
영향이 가장 컸다. 비선형 시간 변형, 회전, 이벤트 삭제·보간, 국소 속도 변경,
노이즈 결합 공격까지 일반화됐다는 의미는 아니다. 이 판정기는 앱·API에
연결하지 않고 오프라인 후보로만 저장했다.

### 적대적 결합 Replay Holdout

위 제한을 직접 검증하기 위해 detector와 임계값을 동결한 뒤, 보지 않은 test
참가자 원본 1,000건에 회전(절대값 4도~13도), 이벤트 재표본화(원본의
0.68배~1.35배), 비선형 시간 변형, 국소 속도 변경을 동시에 적용했다.

| 적대적 Holdout 항목 | Replay ASR | 판정 |
|---|---:|---|
| Exact Fingerprint | 100.0% | 실패 |
| 기존 DTW | 99.9% | 실패 |
| 고급 Replay 메타 판정기 | 100.0% | 실패 |
| XGBoost 단독 | 93.8% | 실패 |
| 최종 결합 | 938/1,000, 93.8% | 실패 |

최종 ASR의 95% Wilson 신뢰구간은 92.13%~95.13%다. 즉, 현재 고급 판정기는
일정한 배율·이동형 `replay_warp`에는 통과했지만 회전·재표본화·국소 시간
변형을 결합한 공격에는 일반화되지 않는다. 운영 적용 판정은 계속 `FAIL`이며
앱·API·production에는 연결하지 않았다.

### 분리 프로필 재학습 반복

위 실패 후 non-test Human 원본 2,000건에는 작은 회전(2도~6도), 제한된
재표본화 비율, Gaussian 국소 시간 곡선만 학습용으로 생성했다. 외부 holdout은
test Human 원본 1,000건과 더 큰 회전(9도~15도), 겹치지 않는 재표본화·시간
지수, 다른 sine 시간 곡선을 사용해 분리했다.

회전 정렬 Procrustes, 호 길이 기반 곡률, chord-distance 형태, 경계 클리핑을
줄이는 보조 형태 신호를 추가하고 participant-group 5-fold OOF로 임계값을
보정했다.

| 반복 | 최종 외부 Holdout ASR | Test Human FRR | 판정 |
|---|---:|---:|---|
| Logistic 12개 신호 | 80.5% | 0.36% | 실패 |
| ExtraTrees 12개 신호 | 81.5% | 0.27% | 실패 |
| Logistic 15개 신호 | 807/1,000 = 80.7% | 4/2,219 = 0.18% | 실패 |

최신 최종 ASR의 95% Wilson 신뢰구간은 78.14%~83.03%다. Human FRR은
기준(3% 이하)을 만족했지만 ASR은 5% 이하를 크게 넘는다. 외부 holdout을
학습에 섞거나 임계값만 낮춰 통과로 만들지 않았으며, 판정은 계속 `FAIL`이다.

## 적용한 검증 기준

- 실험 단계 정상 사용자 오탐률(FRR): 3% 이하
- 운영 단계 정상 사용자 오탐률(FRR): 1% 이하
- 기존 봇 통과율(ASR): 5% 이하
- 미학습 봇 중 가장 높은 통과율: 10% 이하
- `replay_warp` 통과율: 5% 이하
- 동일 궤적 재생 공격 통과율: 1% 이하
- 실제 브라우저 봇 외부 홀드아웃 검증 필수
- 운영 적용 전 사용자 100명 이상의 행동 데이터 확보

## 다음 작업

1. 기기, 입력 방식, 브라우저, 연령대가 다양한 사용자 54명 이상을 추가한다.
2. 브라우저 봇을 화면 크기, DPR, 터치 에뮬레이션, 지연 시간, 재시도,
   독립적인 생성기 버전별로 확장한다.
3. 회전·클리핑을 명시적으로 다루는 source-aware comparator를 별도 후보로
   만들고, 공격 family와 참가자 그룹을 함께 holdout해 다시 검증한다.
4. 행동 점수와 별개로 challenge 1회 소비, nonce·세션·문제 배치 바인딩을
   시험해 replay 방어를 다층으로 검증한다.
5. 더 많은 test 참가자를 확보하고 같은 replay 실험의 신뢰구간과 참가자별
   정상 사용자 오탐률을 다시 측정한다.
6. 세션 시각과 범위가 정확히 수집되는 Shadow Mode 데이터로 요청 빈도
   55건 기준을 다시 보정한다.
7. 결합된 검증 기준을 오프라인에서 모두 통과한 후에만 운영 Shadow Mode를
   진행한다.

## 결과 파일

- `data/raw/human_db_snapshot_20260721T012239Z/manifest.json`
- `data/processed/human_confirmed_20066_20260721/manifest.json`
- `data/interim/extended_bots_10000_20260721.jsonl.manifest.json`
- `data/interim/playwright_bots_90_20260721.jsonl.manifest.json`
- `reports/local_h20066_b10000_f10_v1_20260721/training_summary.json`
- `reports/local_h20066_b10000_f10_v2_20260721/training_summary.json`
- `reports/feature_v1_vs_v2_20260721/FEATURE_VERSION_COMPARISON.md`
- `reports/replay_detection_20260721/replay_metrics.json`
- `reports/local_h20066_b10000_f10_v2_cv5_20260721/training_summary.json`
- `reports/fused_security_v2_cv5_20260721/fused_security_metrics.json`
- `data/interim/replay_warp_external_1000_20260721.jsonl.manifest.json`
- `reports/advanced_replay_v2_cv5_20260721/advanced_replay_metrics.json`
- `reports/advanced_replay_v2_cv5_20260721/ADVANCED_REPLAY_SUMMARY.md`
- `data/interim/adversarial_replay_holdout_1000_20260721.jsonl.manifest.json`
- `reports/adversarial_replay_holdout_20260721/adversarial_replay_metrics.json`
- `reports/adversarial_replay_holdout_20260721/ADVERSARIAL_REPLAY_HOLDOUT.md`
- `data/interim/adversarial_replay_development_2000_20260721.jsonl.manifest.json`
- `data/interim/adversarial_replay_external_1000_20260721.jsonl.manifest.json`
- `reports/advanced_replay_adversarial_v5_20260721/advanced_replay_metrics.json`
- `reports/adversarial_replay_external_v5_20260721/adversarial_replay_metrics.json`
- `reports/adversarial_replay_external_v5_20260721/ADVERSARIAL_REPLAY_ITERATION.md`
