# Catchap 보안 중심 합격 기준

- 버전: 1.0
- 적용일: 2026-07-17
- 적용 범위: 인간/봇 판별 모델, 메인 CAPTCHA, replay 방어, 배포 후보 판정

## 1. 판정 원칙

1. 일반 Accuracy만으로 배포 합격을 선언하지 않는다.
2. P0 필수 항목이 하나라도 실패하면 전체 판정은 `FAIL`이다.
3. P0 또는 P1 항목을 구현하지 못했거나 시험하지 못했으면 `PASS`가 아니라 `INCOMPLETE`로 표시한다.
4. 서버가 확정적으로 판단할 수 있는 토큰 재사용·만료·변조는 즉시 차단한다.
5. ML 점수만 의심스러운 경우는 즉시 차단보다 추가 CAPTCHA(step-up)를 우선한다.
6. 모든 0% 결과는 `0%`만 쓰지 않고 `0/N`, 시험 조건, 95% 신뢰구간을 함께 기록한다.

## 2. 합격 게이트

ASR(Attack Success Rate)은 봇이 사람으로 판정되어 통과한 비율이다. 낮을수록 보안이 강하다.

| 우선순위 | 항목 | 합격 기준 | 최소 증거 |
|---|---|---:|---:|
| P0 | 동일 challenge 재사용 | 추가 성공 0건 | 재전송 10,000회 |
| P0 | CAPTCHA/verdict/login 토큰 재사용 | 추가 성공 0건 | 재전송 10,000회 |
| P0 | 동시 재전송 경쟁 조건 | 최초 1건 외 성공 0건 | 50개 동시 요청 x 100회 |
| P0 | 만료·변조·다른 세션 사용 | 성공 0건 | 유형별 1,000회 |
| P0 | 정답·비밀키·토큰 노출 | 0건 | 응답·로그·프론트 검사 |
| P0 | 학습/검증 데이터 누수 | 0건 | ID·참여자·family·seed 검사 |
| P1 | 알려진 Bot ASR | 5% 이하 | family별 보고 |
| P1 | 미지 Bot 최악 ASR | 10% 이하 | leave-one-family-out |
| P1 | 정확히 같은 궤적 replay ASR | 1% 이하 | 1,000건 이상 |
| P1 | `replay_warp` ASR | 5% 이하 | 시간·크기·회전 변형별 보고 |
| P1 | 제어된 브라우저 자동화 ASR | 10% 이하 | 학습 미포함 시나리오 |
| P1 | 레이트리밋 우회 | 0건 | IP·세션·계정·site key 축 |
| P2 | Human FRR | 실험 3% 이하, 배포 1% 이하 | 참여자 그룹 분할 |
| P2 | API 응답 지연 | p95 100ms 이하 | 보안 로직 포함 부하 시험 |

## 3. 필수 시험 집합

### 3.1 모델 보안

- Human은 participant/session 단위로 그룹 분할한다.
- 신원을 연결할 수 없는 Human은 기본적으로 train-only로 처리한다.
- Bot은 단순 무작위 분할만 하지 않고 family·generator·seed를 통째로 holdout한다.
- 같은 generator의 랜덤 시드 분할 성능만으로 미지 공격 방어를 선언하지 않는다.
- 동일 궤적과 근접 중복 궤적이 train/test에 나뉘어 들어갔는지 검사한다.
- Accuracy, ROC-AUC, F1은 참고 지표로만 사용하고 합격은 ASR, 최악 family, Human FRR로 판정한다.

### 3.2 Replay 및 프로토콜 보안

- 성공·실패와 관계없이 challenge를 한 번만 소비하는지 시험한다.
- challenge와 성공 토큰을 원자적으로 소비하는지 동시성 시험으로 확인한다.
- challenge가 pre-auth session, site key, purpose, 문제 배치와 연결되는지 검사한다.
- 만료 전·후 경계값과 서버 시계 차이를 시험한다.
- 동일 궤적, 시간 와핑, 공간 스케일링, 회전, 노이즈 추가 replay를 각각 시험한다.
- 모델 단독 결과와 nonce·세션·궤적 지문·레이트리밋을 합친 전체 결과를 분리해 보고한다.

### 3.3 실제 자동화와 운영

- Playwright/Selenium 등으로 자사 로컬·스테이징 CAPTCHA를 조작한 보유 시나리오를 사용한다.
- 화면 크기, DPR, 마우스/터치, 지연, 재시도, 복수 워커를 바꿔 시험한다.
- 단일 프로세스와 다중 워커/다중 인스턴스에서 같은 결과가 나오는지 확인한다.
- 배포 전 Shadow Mode에서 실제 Human FRR, step-up 비율, 차단 후보 비율을 기록한다.

## 4. 표준 결과 보고서

앞으로 모든 학습/보안 검증 보고서는 다음 순서를 사용한다.

1. 전체 판정: `PASS`, `FAIL`, `INCOMPLETE`
2. P0 프로토콜/토큰 결과
3. P1 모델, replay, 미지 Bot, 자동화 결과
4. P2 Human FRR, 지연, 운영 결과
5. 공격 유형별 분모/분자, 95% 신뢰구간
6. 이전 체크포인트 대비 절대값과 변화량
7. 잔존 위험, 미구현, 미검증 항목
8. 배포/보류 근거와 다음 조치

## 5. 2026-07-21 체크포인트 판정

최신 판정은 `FAIL / DO NOT PROMOTE`다. 상세 근거는
[`CHECKPOINT_20260721.md`](CHECKPOINT_20260721.md)에 기록했다.

- Human 20,066건, 연결 참여자 46명
- 합성 Bot 10종 10,000건
- Chrome+Playwright 미학습 Bot 3종 90건
- RandomForest, ExtraTrees, XGBoost, LightGBM 비교
- Feature v1 29개와 Feature v2 44개 동일 split 비교
- 사용자·봇 그룹 5-fold OOF 점수로 임계값 보정, test는 보정에 미사용
- ML·Fingerprint·DTW·세션 빈도 결합 후 알려진 Bot 전체 ASR 0.84%
- exact replay ASR 0%, 60초 봇 버스트 ASR 0.8%
- 1차 결합의 `replay_warp` 최악 ASR은 8.75%로 5% 기준 실패
- 추가 9개 replay 신호는 보지 않은 참가자 기반 새 seed holdout에서
  `replay_warp` ASR 0/1,000, Human FRR 0/2,219로 현재 family 기준 통과
- 회전·재표본화·비선형 시간·국소 속도 변경 결합 holdout 1,000건에서
  최종 Replay ASR 93.8%로 실패
- 개발용 일부 변형 2,000건만 추가하고 분리 외부 holdout 1,000건으로 다시
  검증했으나 최신 최종 Replay ASR은 80.7%, Human FRR은 0.18%로 ASR 기준 실패
- 해당 적대적 변형을 학습에 넣지 않은 상태의 실패 결과이며, 앱·API에는 미적용
- XGBoost·LightGBM의 외부 브라우저 Bot ASR 0%, 나머지 모델은 최악 100%
- 전체 모델이 최소 하나 이상의 Human FRR/ASR 또는 다양성 게이트 실패
- production 모델 승격 없음

### 일회성 Challenge 프로토콜 구현 상태

`ai_captcha_challenges`와 CAPTCHA 백엔드 전용 발급·소비 API를 추가했다. nonce와
문제 바인딩은 해시만 저장하며, 세션·site key·purpose·문제 바인딩이 모두 일치한
첫 요청만 조건부 UPDATE로 소비한다. failed verdict도 소비되며, 로컬 SQLite에서
재사용·만료·바인딩 불일치·2개 동시 소비를 시험했다.

이는 구현 및 단위 검증 상태일 뿐 P0 합격은 아니다. 실제 MySQL/InnoDB 환경에서
재전송 10,000회와 50개 동시 요청 x 100회 시험을 마친 뒤에만 P0 판정이 가능하다.

### 이전 2026-07-15 체크포인트

2026-07-15 체크포인트는 일반 모델 성능과 leave-one-family-out을 검증했지만, P0 프로토콜 공격·동시성·세션 바인딩·실제 브라우저 자동화를 전부 시험하지 않았다. 따라서 이 기준으로는 `INCOMPLETE`이며 배포 합격이 아니다.

RandomForest 기준 주요 기록은 다음과 같다.

| 항목 | 결과 | 판정 |
|---|---:|---|
| 일반 Accuracy | 95.57% | 참고 |
| Human FRR | 1.12% | 실험 기준 통과, 배포 1% 미달 |
| 일반 Bot Recall | 91.13% | 알려진 Bot ASR 8.87%, 미달 |
| 미지 Bot 최악 Recall | 0.1% | ASR 99.9%, 실패 |
| replay 제외 미지 9종 최악 Recall | 98.9% | 통과 |
| P0 보안 시험 | 미완료 | INCOMPLETE |

## 6. 참고

- OWASP: nonce, timestamp, 중복 거부를 통한 replay 방어
  - https://scs.owasp.org/SCWE/SCSVS-COMM/SCWE-022/
- Redis `GETDEL`: 값을 읽고 원자적으로 삭제
  - https://redis.io/docs/latest/commands/getdel/
