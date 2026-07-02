# 학습 가이드

## 0. 전제

- **현재 실제 데이터가 없으므로 프로젝트는 자동으로 모델을 학습하지 않습니다.**
- 준비 검사를 통과하지 못하면: 모델 학습 금지, 기존 모델 덮어쓰기 금지,
  `reports/data_readiness.json` 생성 후 종료 코드 2.
- 테스트 fixture로 나온 수치는 **최종 성능이 아닙니다.**

## 1. 데이터 준비 상태 검사

```bash
python -m training.check_data_readiness
# 또는
python -m training.run_training_pipeline --check-only
```

검사 항목: MySQL 연결, 테이블/뷰 존재, Human/Bot 개수, Human 참여자 수, Bot family
수, `label_source` 누락, valid/pending/rejected, Feature NULL/NaN/Infinity, Feature
스키마 버전 일치, 그룹 split 가능 여부, 클래스 불균형 비율.

### 준비 기준 (환경변수, 프로젝트 기본값)

| 변수 | 기본값 |
|------|--------|
| `MIN_HUMAN_SAMPLES` | 500 |
| `MIN_BOT_SAMPLES` | 500 |
| `MIN_HUMAN_PARTICIPANTS` | 20 |
| `MIN_BOT_FAMILIES` | 3 |

> 이 값들은 **절대적인 연구 기준이 아니라**, 데이터가 너무 적을 때 학습을 막기 위한
> 프로젝트 기본 설정입니다. `--min-*` 옵션이나 환경변수로 조정하세요.

`data_not_ready` 예시:

```json
{
  "ready": false,
  "reason": "data_not_ready",
  "human_samples": 120, "required_human_samples": 500,
  "bot_samples": 0, "required_bot_samples": 500,
  "missing": ["Human 데이터 380개 부족", "Bot 데이터 500개 부족", "Bot family 3종 부족"]
}
```

## 1.5 규칙 기반 봇 생성 (Bot 데이터 부트스트랩)

실제 사용자 없이도 **지금 만들 수 있는 유일한 데이터**가 규칙 기반 봇입니다. 봇
데이터의 부트스트랩이자 baseline 모델의 "쉬운 시험"이며, GAN 봇(어려운 시험)보다
먼저 필요합니다. (방어 연구용 — 우리 CAPTCHA 탐지 학습용이며 우회 도구가 아닙니다.)

```bash
# 파일로 생성 (기본): straight / accel / jitter 3종
python -m training.generate_rule_bots --count 600 --out data/interim/rule_bots.jsonl

# collect API로 바로 전송 → 품질검사·Feature 계산을 거쳐 DB 적재
python -m training.generate_rule_bots --count 600 --post http://<host>/api/v1/behavior/collect
```

- 라벨: `label=bot, label_source=rule_bot, bot_family∈{straight,accel,jitter}`,
  `generator_version=rule_v1`.
- family 3종이라 readiness의 `MIN_BOT_FAMILIES=3` 기준을 채웁니다.
- GAN 봇은 실제 Human 데이터가 쌓인 뒤의 후속 단계입니다(§9).

## 2. 학습에 쓰는 데이터

`ai_training_dataset` 뷰 = `quality_status='valid'` + `label ∈ {human,bot}` +
`label_source` 존재. 여기에 더해 `feature_schema_version` 이 일치해야 합니다.
Human=1, Bot=0으로 변환합니다.

## 3. 분할 (누수 없는 그룹 split)

- 비율: train 70 / validation 15 / test 15
- **그룹 단위** split (행 단위 random split 금지):
  - 같은 Human 참여자는 한 split에만
  - 같은 Bot `generator_version`/템플릿 그룹도 한 split에만
  - GAN Bot은 원본 Human과 같은 split (origin participant로 그룹화)
  - Replay 원본/복사본은 같은 generator 그룹으로 같은 split
- split manifest는 `data/metadata/` 에 저장, 누수 자동 검사(`LeakageError`).

## 4. 세 모델 학습

RandomForest / XGBoost / LightGBM — **동일한 29 Feature, 동일한 split, 고정 seed,
클래스 불균형 처리.** validation으로 임계값(threshold)을 정하고, **test는 최종 평가에
한 번만** 사용합니다. 학습 중 오류가 나면 기존 production 모델을 보존합니다.

## 5. 평가 지표

Accuracy, Human Precision/Recall/F1, Bot Recall, **Human False Rejection Rate(FRR)**,
ROC-AUC, PR-AUC, Confusion Matrix, 평균 추론 시간, Feature Importance.

## 6. 최종 모델 선택 기준

1. **Human FRR ≤ 3%** 인 모델만 후보
2. 후보 중 **Bot Recall 최고**
3. 동률 → **Human F1 높은** 모델
4. 또 동률 → **추론 시간 빠른** 모델
5. 만족하는 모델이 없으면 → **production 교체하지 않고 경고 보고서** 생성

## 7. 한 명령 파이프라인

```bash
python -m training.run_training_pipeline
```

순서: MySQL 확인 → readiness → (미준비면 보고서만) → 데이터 로드 → 그룹 split →
RF/XGB/LGBM 학습 → validation threshold → test 평가 → 최적 선택 → candidate 저장 →
조건 충족 시 production 교체.

옵션: `--check-only`, `--dataset-version`, `--seed`, `--min-human-samples`,
`--min-bot-samples`, `--min-human-participants`, `--min-bot-families`, `--no-promote`.

### 산출물

```
reports/model_comparison.csv
reports/training_summary.json
reports/confusion_matrix_<모델>.png
reports/feature_importance_<모델>.csv
models/candidate/<모델>.joblib
models/production/production_<모델>.joblib   # 승격된 최종 모델만
data/metadata/split_manifest_<버전>.json
```

### 모델 번들에 저장되는 것

model 객체, model_name, model_version, Feature 목록, feature_schema_version,
dataset_version, 학습 시각, 선택 threshold, validation 지표, 라이브러리 버전.

## 8. production 교체 조건

- 새 후보가 **선택 기준(§6)을 통과**해야만 교체합니다.
- 새 모델 검증이 끝날 때까지 기존 production 모델은 유지됩니다.
- 교체는 파이프라인의 **마지막 단계**이며, `--no-promote`로 보류할 수 있습니다.
- 교체 후 무중단 반영: `POST /api/v1/admin/model/reload` (`X-Admin-Key`).

## 9. GAN (실제 Human 데이터가 충분해진 뒤에만)

추가 게이트: `MIN_GAN_HUMAN_SAMPLES=2000`, `MIN_GAN_HUMAN_PARTICIPANTS=50`.
GAN은 `quality_status=valid` + `label=human` + `label_source=controlled_collection`
+ **train split** + 완전한 pointerdown~pointerup 원본만 사용합니다. 입력은 Feature CSV가
아니라 **원본 포인터 이벤트**(x_normalized, y_normalized, t_ms, 순서)이며, 시작(0,0)
/끝(1,0)/시간(0~1) 정규화 후 고정 길이 시계열로 변환합니다.

```bash
python -m training.train_gan --check-readiness   # 게이트 보고서만
python -m training.train_gan --train             # 게이트 통과 시에만 학습
python -m training.generate_gan_bots             # human-like bot 생성
```

- 원본과 지나치게 유사한 생성 경로는 제거합니다.
- 생성물은 `label=bot, label_source=gan_bot, bot_family=gan` 으로 다룹니다.
- 일부 GAN Bot은 **학습에 넣지 않고 test 전용**으로 보관합니다.
- GAN 추가 전/후 모델 성능을 비교합니다. GAN도 **자동 실행하지 않습니다.**
