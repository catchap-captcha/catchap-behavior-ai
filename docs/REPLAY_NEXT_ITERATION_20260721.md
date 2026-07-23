# Replay 다음 검증 단계 - 2026-07-21

## 현재 상태

이전 외부 `replay_warp` holdout은 봉인했다. 해당 파일의 SHA-256은
`training.holdout_registry`에 등록되어 있어, 다음 학습 명령의 `--bot-attempts`로
넣으면 즉시 실패한다. 이 holdout 결과로 모델을 다시 맞추거나 합격을 주장하지 않는다.

## 이번에 완료한 준비

1. 개발용 replay 생성기를 확장했다.
   - 넓은 재표본화 비율
   - 작은 회전과 미세 곡률 이동
   - 다중 구간의 국소 속도 변화
   - 비선형 시간 변화
2. 개발 split의 사람 궤적만 사용해 3,000건을 생성했다.
   - `data/interim/adversarial_replay_broad_development_3000_20260721.jsonl`
   - `training_usage=development_only`
3. 복합 replay 개발용 3,000건을 추가 생성했다.
   - `data/interim/adversarial_replay_composite_development_3000_20260721.jsonl`
   - 적응형 재표본화, 곡률 변화, 회전, 비선형 시간 변화, 국소 속도 변화,
     속도 연동 미세 흔들림을 한 궤적에 조합한다.
   - 각 3,000건 모두 세 가지 신규 변형 신호를 포함하는지 확인했다.
4. 신규 참여자 외부 holdout 생성기를 만들었다.
   - `tools/generate_fresh_replay_holdout.py`
   - 기존 모델 Human 데이터의 참여자 ID와 겹치는 궤적은 소스로 사용할 수 없다.
   - 생성 결과는 `training_usage=external_holdout_only`로 표시되며 학습 입력에서 거부된다.
5. 후보의 shadow mode 게이트를 강화했다.
   - 참여자 100명 이상
   - 일반 test Human FRR 3% 이하
   - 알려진 Bot ASR 5% 이하
   - 미지 family 최악 ASR 10% 이하
   - replay_warp ASR 5% 이하
   - 새 참여자 기반 외부 holdout ASR 5% 이하

## 현재 막힌 조건

현재 Human 데이터는 연결 참여자 46명이며, 기존 모델이 보지 않은 참여자는 0명이다.

| 항목 | 현재 | 필요 |
|---|---:|---:|
| 연결 참여자 | 46명 | 100명 이상 |
| 추가 신규 참여자 | 0명 | 54명 이상 |
| 신규 참여자 외부 holdout | 생성 불가 | 1,000건 권장 |

준비 결과는 `reports/fresh_external_holdout_readiness_20260721.json`에 저장했다.
따라서 다음 후보를 학습할 수는 있어도, 새 외부 holdout 검증 없이 shadow mode로 가면 안 된다.
다만 참여자 5명 이상과 일반 test Human FRR 3% 이하를 만족하는 후보는
`observation_only_eligible=true`으로 표시할 수 있다. 관찰 전용 모드는 점수·오탐 후보만
기록하며, CAPTCHA 통과·차단·추가 인증 결정을 바꾸지 않는다.

## 새 데이터 수집 뒤 실행 순서

1. 새 참여자 54명 이상이 포함된 Human snapshot을 만든다.
2. snapshot을 정제해 Human feature 파일을 만든다.
3. 다음 명령으로 신규 참여자 외부 holdout을 만든다.

```bash
.venv/bin/python -m tools.generate_fresh_replay_holdout \
  --human-attempts <새_snapshot>/human_attempts.jsonl \
  --known-human-features data/processed/human_confirmed_20066_v2_20260721/human_features.jsonl \
  --output data/interim/fresh_external_replay_1000.jsonl \
  --count 1000 \
  --min-fresh-participants 54
```

4. 넓은 개발용 replay 3,000건과 복합 replay 3,000건을 포함해 후보를 재학습한다.
5. 생성된 신규 외부 holdout은 `--external-bot-holdout`으로만 넣어 ASR을 측정한다.
6. 모든 게이트가 통과할 때만 `shadow_mode_eligible=true`을 확인한다.

## 주의

새 외부 holdout의 결과를 본 뒤에는 그 파일을 학습·임계값 보정에 사용하지 않는다. 성능을
개선하려면 새 개발 변형과 또 다른 신규 참여자 holdout을 다시 분리해야 한다.
