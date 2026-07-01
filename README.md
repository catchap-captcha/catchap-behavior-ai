# catchap ai-service

웹 **드래그 CAPTCHA**의 Human/Bot **행동 분석** 서비스입니다. 마우스/터치 포인터
드래그 궤적을 받아 행동 Feature를 계산하고, 데이터가 충분히 쌓이면 모델을 학습해
"사람인지 봇인지"를 판정합니다.

> ⚠️ **현재는 실제 수집 데이터가 없습니다.** 이 저장소는 **수집 → 전처리 → 학습 →
> 검증 파이프라인**을 완성해 둔 상태이며, 데이터가 쌓이기 전에는 모델을 만들지
> 않습니다. 테스트 fixture로 나오는 수치는 **최종 성능이 아닙니다.**

## 역할 분담 (전제)

| 담당 | 범위 |
|------|------|
| **다른 팀** | CAPTCHA 화면(프론트) + CAPTCHA 백엔드 개발 |
| **DB 팀** | MySQL 8.0 **생성·운영** ( `db/schema_mysql.sql` 적용) |
| **우리(ai-service)** | 행동 데이터 수집 API·전처리·Feature·학습·검증·추론 API |

## 프로젝트 구조

```
app/            FastAPI 앱 (api / schemas / services / database)
db/             MySQL DDL (DB 팀 전달용)
data/           raw/interim/processed/metadata (실데이터는 git 제외)
training/       readiness → build → split → train → evaluate → select → (gan)
models/         candidate / production 모델 번들 (git 제외)
reports/        학습·평가 산출물 (git 제외)
tests/          pytest (fixture는 tests/fixtures)
docs/           API_CONTRACT / DB_REQUEST / DATA_SCHEMA / TRAINING_GUIDE
```

## 빠른 시작

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # 값 채우기 (비밀번호 등은 커밋 금지)

# API 실행
uvicorn app.main:app --reload
```

- `GET /health` — API/MySQL/모델 상태
- `POST /api/v1/behavior/collect` — 통제 수집 (CAPTCHA 백엔드만, `X-API-Key`)
- `POST /api/v1/behavior/predict` — 운영 추론 (모델 없으면 **503 `model_not_ready`**)
- `POST /api/v1/admin/model/reload` — 모델 재로딩 (`X-Admin-Key`)

## 데이터가 쌓이기 전 (지금)

```bash
# DB 준비 상태만 확인 (모델 학습 안 함)
python -m training.check_data_readiness
python -m training.run_training_pipeline --check-only
```

데이터가 부족하면 `reports/data_readiness.json` 에 부족 항목을 한국어로 기록하고
**종료 코드 2**로 끝납니다. 모델 파일은 만들지 않습니다.

## 데이터가 쌓인 뒤 (학습)

```bash
# 준비 검사 통과 시에만 3개 모델 학습 → 검증 → 최적 모델 선택 → production 승격
python -m training.run_training_pipeline

# 학습만 하고 production 교체는 보류
python -m training.run_training_pipeline --no-promote
```

## GAN (실제 Human 데이터가 충분해진 뒤에만)

```bash
python -m training.train_gan --check-readiness   # 게이트 확인
python -m training.train_gan --train             # 게이트 통과 시에만 학습
python -m training.generate_gan_bots             # human-like bot 생성
```

## 어린이 데이터 주의

- 참여자는 **익명 ID(`anonymous_participant_id`)**로만 저장하며 개인 식별정보를
  넣지 않습니다.
- `age_group`(adult/child/unknown)과 `consent_version`(동의서 버전)을 반드시
  기록하고, **아동 데이터는 보호자 동의 절차**를 거친 것만 수집합니다.
- 성인 데이터로만 학습한 모델은 아동 조작을 오탐할 수 있으므로 배포 대상 연령을
  확인하세요. 자세한 내용은 [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md).

## 문서

- [docs/API_CONTRACT.md](docs/API_CONTRACT.md) — CAPTCHA 팀 API 규격
- [docs/DB_REQUEST.md](docs/DB_REQUEST.md) — DB 팀 테이블 생성 요청
- [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md) — 원본/Feature 스키마
- [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) — 학습·평가·선택·GAN 가이드

## 테스트

```bash
pytest -q      # DB는 인메모리 SQLite 사용, MySQL 불필요
```
