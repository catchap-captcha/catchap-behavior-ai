# ML 생성 봇 - 2026-07-21

## 목적

규칙형 봇만으로는 실제 사람과 비슷한 궤적을 충분히 검증하기 어렵다. 그래서 사람의
포인터 궤적 분포를 오프라인에서 학습해, 탐지 모델 검증용 어려운 합성 봇을 생성한다.
이 도구는 우리 CatChap 데이터 파일만 읽고 JSONL을 만들며, 브라우저·CAPTCHA·네트워크를
조작하지 않는다.

## 생성 모델

`PCA + diagonal Gaussian Mixture Model(GMM)`을 사용한다.

1. 궤적을 48개 정규화된 x/y/시간 간격 벡터로 변환한다.
2. PCA가 궤적의 저차원 표현을 학습한다.
3. GMM이 그 잠재 공간에서 새 샘플을 생성한다.
4. 사람 원본의 이벤트 수 분포(5~95백분위)로 다시 복원한다.
5. 원본 궤적과의 최근접 거리가 너무 가까운 생성본은 폐기한다.

VAE보다 현재 데이터 규모(연결 참가자 46명)에서 결과를 재현하고 원본 근접성을
확인하기 쉬운 초기 ML 생성기다. 이 결과가 안전성을 보장하거나 실제 공격자를 완전히
대표한다는 뜻은 아니다.

## 데이터 분리

| 세트 | 사람 원본 | 생성 수 | 용도 |
|---|---:|---:|---|
| 개발용 | train split 12,124건 | 1,000건 | 이후 후보 모델 학습 실험에만 사용 가능 |
| 외부 holdout | test split 5,076건 | 1,000건 | 평가 전용, 학습·임계값 보정 금지 |

외부 holdout 행에는 `training_usage=external_holdout_only`가 들어간다.
`training.run_local_training`은 이 행을 학습에 넣으려 하면 오류를 발생시킨다.

## 현재 산출물

- 개발용 데이터: `data/interim/ml_pca_gmm_development_1000_20260721.jsonl`
- 외부 holdout 데이터: `data/interim/ml_pca_gmm_external_holdout_1000_20260721.jsonl`
- 개발용 생성 모델: `models/generative/ml_pca_gmm_development_20260721.joblib`
- 외부 holdout 생성 모델: `models/generative/ml_pca_gmm_external_holdout_20260721.joblib`

두 데이터 세트 모두 품질 검사 거절 0건, 사람 원본과 정확히 같은 Trace Fingerprint
0건을 확인했다. 생성된 데이터나 모델은 production에 적용하지 않았다.

## 외부 ML 봇 1차 검증

학습에 넣지 않은 외부 holdout ML 봇 1,000건을 현재 XGBoost 후보에 점수화했다.

| 항목 | 결과 | 기준 | 판정 |
|---|---:|---:|---|
| 차단된 외부 ML 봇 | 944 / 1,000 | - | 참고 |
| 외부 ML 봇 ASR | 56 / 1,000 = 5.6% | 5% 이하 | 실패 |
| 현재 후보의 사람 test FRR | 5.40% | 3% 이하 | 실패 |

따라서 이 생성 봇은 기존 후보에 충분히 어려운 외부 검증 데이터이며, 현재 후보는
production 승격 대상이 아니다. 이 결과는 생성 봇을 학습에 넣기 전의 기준선이다.

XGBoost 모델 파일은 학습 당시와 다른 라이브러리 버전에서 읽을 때 호환성 경고가 발생했다.
점수는 반환됐지만, 최종 모델 판정은 같은 학습 런타임에서 재학습한 후보로 다시 확인해야
한다.

## 재생성

```bash
python -m training.generate_ml_bots \
  --human-attempts data/processed/human_confirmed_20066_v2_20260721/human_attempts.jsonl \
  --split-manifest reports/local_h20066_b10000_f10_v2_20260721/split_manifest.json \
  --role development \
  --count 1000 \
  --out data/interim/ml_pca_gmm_development_1000_20260721.jsonl \
  --model-out models/generative/ml_pca_gmm_development_20260721.joblib
```

외부 holdout은 `--role external_holdout`으로 생성한다. 이 파일은 학습 입력이 아니라
재학습 뒤 모델 통과율(ASR)을 측정하는 용도로만 쓴다.

## 다음 검증

개발용 ML 생성 봇을 기존 합성 봇과 합친 별도 후보 모델을 만든 뒤, 기존 사람 test와
외부 ML holdout을 모두 사용해 Human FRR과 Bot ASR을 다시 측정한다. 외부 holdout 결과가
좋아도 현재 적대적 replay holdout을 통과하지 못하면 production 승격은 하지 않는다.

## Hybrid Motion 레드팀 봇 - 2026-07-22

기존 PCA-GMM 생성본만으로는 궤적의 모양, 시간, 이벤트 전달 방식을 동시에 흔들기
어렵다. 그래서 PCA-GMM으로 원본과 일정 거리 이상 떨어진 기본 궤적을 만든 뒤, 별도의
모션 정책을 적용하는 `hybrid_motion_redteam` 생성기를 추가했다.

적용하는 변형은 다음과 같다.

1. 시작점과 끝점은 유지한 비대칭 곡률 경로
2. 작은 좌표 흔들림과 후반 미세 보정
3. 회전량이 큰 구간에서 시간이 늘어나는 속도-회전 연동
4. 비선형 시간 진행과 전체 수행 시간 변화
5. 8/10ms 또는 12/16ms 화면 프레임 단위 시간 처리
6. 중간 `pointermove` 일부를 합치는 이벤트 coalescing

이는 실제 CAPTCHA를 풀거나 브라우저를 제어하는 도구가 아니다. 로컬 JSONL만 읽고
합성 JSONL을 만드는 오프라인 레드팀 데이터 생성기다. 생성 뒤에도 원본 사람 궤적과의
최근접 거리를 다시 확인하며, 사람의 원본 `attempt_id`는 출력에 포함하지 않는다.

### 분리 규칙

| 세트 | 생성 수 | 표기 | 사용 가능 범위 |
|---|---:|---|---|
| 레드팀 보정 | 500건 | `training_usage=redteam_only` | 생성기/공격 진단 확인만 가능 |
| 외부 holdout | 500건 | `training_usage=external_holdout_only` | 미래의 단 한 번 평가만 가능 |

두 세트 모두 탐지 모델 학습과 임계값 보정에 사용할 수 없다.
`training.run_local_training.build_bot_feature_rows`는 `development_only`가 아닌 봇
데이터를 기본적으로 거부한다. 이 규칙은 레드팀 봇을 학습 데이터에 섞어서 같은 공격에
과적합하는 것을 막는다.

### 현재 산출물

- 생성기: `training/generate_hybrid_redteam_bots.py`
- 보정 세트: `data/interim/hybrid_motion_redteam_calibration_500_20260722.jsonl`
- 봉인 외부 세트: `data/interim/hybrid_motion_redteam_external_holdout_500_20260722.jsonl`
- 생성 모델: `models/generative/hybrid_motion_redteam_calibration_20260722.joblib`,
  `models/generative/hybrid_motion_redteam_external_20260722.joblib`

두 세트는 사람 개발 데이터 15,802건만 사용해 만들었다. 이번 생성에서는 품질 검사
거절 0건, 생성 뒤 novelty 기준 거절 0건이었다. 아직 사람 lockbox가 이미 소비되어 있어,
현재 탐지기에 대한 통과율이나 성능 개선 수치는 보고하지 않는다. 새 lockbox가 준비된 뒤에
외부 holdout을 한 번만 점수화한다.

### 보정 세트 초기 공격 진단

학습 및 임계값 조정에 사용하지 않은 `revalidation_two_view_baseline_20260722` 후보를
고정한 채, 보정용 레드팀 500건만 점수화했다.

| 항목 | 결과 |
|---|---:|
| 차단 | 498 / 500 |
| 통과 | 2 / 500 |
| Hybrid Motion Red-team Bot ASR | **0.4%** |

이 결과는 새 하이브리드 봇군에 대한 공격 진단일 뿐, 사람 데이터가 포함되지 않았으므로
Human FRR을 0%로 해석하면 안 된다. 통과한 두 샘플은 모두 10ms 프레임 시간 처리와 중간
수준의 속도-회전 연동을 포함했다. 표본이 두 건뿐이므로 이 특성에 맞춘 feature 추가나
임계값 변경은 하지 않는다. 봉인된 외부 500건은 아직 점수화하지 않았으며, 새 사람
lockbox와 함께 한 번만 평가한다.

진단 보고서: `reports/hybrid_motion_redteam_calibration_score_20260722.json`

### 점수 유도형 약점 세트 - 2026-07-22

단일 500건 보정 세트보다 어려운 후보를 체계적으로 찾기 위해,
`tools/mine_hybrid_redteam_weaknesses.py`를 추가했다. 이 도구는 다음만 수행한다.

1. PCA-GMM 하이브리드 후보를 임시 파일에 대량 생성한다.
2. 이미 고정된 two-view detector로 `P(Human)`만 점수화한다.
3. 사람 점수가 높은 상위 후보를 `redteam_only` 약점 세트로 저장한다.

모델 fitting, feature 선택, 임계값 조정은 수행하지 않는다. 임시 후보 파일은 저장하지
않고, 선별된 약점 후보만 남긴다.

첫 실행에서는 사람 development 원본 15,802건으로 후보 3,000건을 만들고, 고정된
`revalidation_two_view_baseline_20260722` 후보로 점수화했다.

| 항목 | 결과 |
|---|---:|
| 임시 후보 | 3,000건 |
| 현재 detector threshold를 통과한 후보 | 3건 = 0.1% |
| 보관한 상위 약점 후보 | 100건 |
| 보관 후보의 Human score 범위 | 0.999846718 ~ 0.999999695 |

산출물: `data/interim/hybrid_motion_score_guided_weakset_100_20260722.jsonl`

이 파일의 모든 행은 `training_usage=redteam_only`이고, 기본 학습 변환 함수는 이를
거부한다. 100건 중 3건은 실제 현재 threshold를 넘었고, 나머지는 threshold 바로 아래의
경계 사례다. 이 결과는 사람 분모가 없는 Bot-only 공격 진단이며, 현재 고정 모델의
Human lockbox FRR 6.31% 실패를 가리지 않는다.

봉인된 hybrid external holdout 500건은 이 실행에 사용하지 않았다. 다음 단계는 약점
세트가 충분히 쌓였을 때만 공통 motion/feature 패턴을 분석하는 것이며, 특정 3건에 맞춰
threshold나 detector를 조정하지 않는다.

### 점수 유도형 약점 패턴 분석 - 2026-07-22

`tools/analyze_redteam_weaknesses.py`로 weak-set 100건을 같은 고정 detector에서
차단된 hybrid calibration 498건과 **기술 통계로만** 비교했다. 비교 과정에서 모델 fitting,
feature 선택, threshold tuning은 수행하지 않았고, 외부 holdout도 열지 않았다.

| 항목 | weak-set 100건 | 차단된 calibration 대조군 498건 |
|---|---:|---:|
| 현재 threshold 통과 | 3건 (3.0%) | 0건 |
| Human score 중앙값 | 0.999975 | 0.064112 |
| 궤적 길이 평균 | 83.03 | 247.01 |
| 선형성 평균 | 0.0917 | 0.2718 |
| 지속 시간 평균 | 5,513ms | 3,868ms |

약점 후보는 대조군보다 짧고 덜 직선적인 궤적, 낮은 속도와 더 긴 지속 시간, 높은
`y_deviation` 및 pause 위치 엔트로피를 보였다. KMeans 3개 군집은 각각 33/45/22건이었고,
각 군집에 threshold 통과 사례가 하나씩 있어 단일 모션 정책 또는 단일 feature가 원인이라고
결론 내릴 수 없다. 실제 mutation 파라미터(곡률, time power, turn slowdown, frame 간격,
event coalescing)의 차이도 작았다.

따라서 이 결과는 다음 red-team 후보 생성과 반복 분석에 사용할 가설일 뿐, 이 3건에 맞춘
feature 추가·임계값 변경·detector 재학습의 근거가 아니다. 방어 개선은 별도 개발 데이터의
OOF 검증과 새 holdout 평가를 통과할 때만 검토한다.

분석 보고서: `reports/hybrid_motion_score_guided_weakness_analysis_20260722.md`
원본 수치: `reports/hybrid_motion_score_guided_weakness_analysis_20260722.json`

### 반복 broad red-team 탐색 - 2026-07-22

점수 유도형 약점 세트의 선택 편향을 줄이기 위해 넓은 모션 범위(곡률, jitter, 시간 곡선,
속도-회전 감속, 프레임 간격, event coalescing, late correction)를 후보마다 바꾸며 독립 seed
5회로 고정 detector 탐색을 반복했다. 각 실행은 3,000건이었고, 모든 출력은
`redteam_only`다. 봉인 external holdout은 읽지 않았다.

| 항목 | 결과 |
|---|---:|
| 전체 후보 | 15,000건 |
| 현재 threshold 통과 | 73건 = 0.487% |
| 실행별 ASR 범위 | 0.133% ~ 0.933% |
| 통과 + 경계 약점 세트 | 120건 |
| dynamics view가 최종 낮은 점수를 준 약점 세트 | 100건 = 83.3% |

다음 두 방향은 5/5회 반복됐다.

- `pause_position_entropy`: 약점 세트가 차단 대조군보다 높음 (평균 차이 `+0.404156`)
- `turn_change_smoothness`: 약점 세트가 차단 대조군보다 낮음 (평균 차이 `-0.131669`)

하지만 두 신호는 모두 이미 schema v2.3의 dynamics physics feature다. 반면
`speed_turn_abs_correlation`은 방향이 2/5회만 일치해 재현되지 않았다. 따라서 새 feature를
추가하지 않았고, `redteam_only` 데이터로 detector를 재학습하거나 임계값을 조정하지 않았다.
모델 설정 변화가 없으므로 OOF를 다시 실행하지 않은 것도 의도된 결정이다.

반복 보고서: `reports/redteam_repeated_scan_20260722/반복_redteam_약점_요약.md`
원본 집계: `reports/redteam_repeated_scan_20260722/반복_redteam_약점_요약.json`

### VAE Hybrid 반복 red-team 탐색 - 2026-07-22

PCA-GMM 후보만으로 찾은 패턴이 생성기 편향인지 확인하기 위해, 같은 개발 Human 15,802건으로
conditional VAE를 한 번 학습했다. VAE가 생성한 후보마다 동일한 broad 모션 정책을 적용하고,
품질 검사와 VAE 원본·변형 후 최근접 거리 검사를 모두 통과한 것만 `redteam_only`로 저장했다.

PyTorch VAE 생성과 LightGBM detector 점수화는 별도 프로세스로 분리했다. 이는 둘을 한 프로세스에
올렸을 때 발생한 macOS 병렬 런타임 충돌을 피하기 위한 구현 방식이며, detector fitting이나
threshold tuning과는 무관하다.

| 생성기 | 후보 | 통과 | 전체 ASR | 실행별 중앙 ASR | dynamics binding |
|---|---:|---:|---:|---:|---:|
| PCA-GMM hybrid | 15,000 | 73 | 0.487% | 0.600% | 83.3% |
| VAE hybrid | 15,000 | 118 | 0.787% | 0.800% | 88.6% |

VAE와 PCA-GMM 모두 `pause_position_entropy`가 높고 `turn_change_smoothness`가 낮은 방향을
반복했다. VAE에서는 `speed_turn_abs_correlation`이 추가로 5/5회 반복됐지만, 세 신호는 모두
이미 schema v2.3에 포함되어 있다. 따라서 새 feature 추가, VAE red-team 데이터 기반 재학습,
threshold 조정, OOF 재실행은 하지 않았다.

생성기: `tools/vae_redteam_weakness_search.py`
VAE 반복 보고서: `reports/vae_redteam_repeated_scan_20260722/반복_vae_redteam_약점_요약.md`
PCA-GMM 비교 보고서: `reports/redteam_generator_comparison_20260722/PCA_GMM_VAE_redteam_비교.md`
