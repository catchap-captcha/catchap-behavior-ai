# 김민서님께 — behavior 데이터 두 DB 검토 회신

보낸 사람: 조성원 · 2026-07-30

---

## 0. 먼저 — 전제가 반대입니다

메일에 이렇게 적어주셨는데,

> `captcha_ms` (210 MySQL) — 캡차앱
> `ai_*` (GPU 로컬 3306, 조성원님 DB)

**실제로는 정확히 반대입니다.** 서버에서 확인했습니다.

```
ai_*        →  10.0.1.168:3306   (host-10-0-1-168, MySQL 8.0.46)
               DB: catchap_dev_db,  계정: catchap_backend

captcha_ms  →  GPU 로컬          (127.0.0.1 → 유닉스 소켓)
```

근거 두 가지입니다.

**(1) AI 의 접속 설정**

```
MYSQL_HOST=10.0.1.168
MYSQL_DATABASE=catchap_dev_db
MYSQL_USER=catchap_backend
```

접속해서 `SELECT @@hostname` 하면 `host-10-0-1-168` 이 나옵니다.
GPU 에서 나가는 established 연결도 `10.0.1.52 → 10.0.1.168:3306` 하나뿐입니다.

**(2) 캡차의 접속 경로**

`app/db.py::_connect` 가 이렇게 되어 있습니다.

```python
local_hosts = {"localhost", "127.0.0.1", "::1"}
if self.settings.db_host in local_hosts and socket_path.exists():
    kwargs["unix_socket"] = str(socket_path)     # ← 여기로 갑니다
else:
    kwargs.update(host=..., port=...)
```

GPU 에 `/var/run/mysqld/mysqld.sock` 이 있고 `mysqld` 가 돌고 있습니다.
캡차에 요청을 5건 넣어봐도 **TCP 연결이 하나도 새로 안 생깁니다** — 소켓으로
로컬에 붙기 때문입니다.

> 혹시 `DB_HOST` 를 명시적으로 원격으로 두셨다면 제가 틀린 것이니 알려주세요.
> `.env` 를 못 읽어서 코드 경로와 소켓·연결 관찰로만 판단했습니다.

---

## 1. "GPU 로컬이라 성능·격리에 좋다"는 걱정은 이미 답이 나와 있습니다

물어보신 것:

> 모델서비스의 pointer_events 대량쓰기가 내부망(10.0.1.x) 왕복이 됩니다.
> 이게 모델 채점 지연/처리량에 문제될까요?

**이미 그렇게 돌고 있고, 문제없습니다.** `ai_*` 는 처음부터 10.0.1.168 에
씁니다. 오늘 실측한 풀이 한 건이 이렇습니다.

```
ai_pointer_events         36행
ai_attempt_features        1행
ai_interaction_summaries   1행
ai_security_features       1행
ai_model_predictions       1행
ai_shadow_outcomes         1행
```

내부망 왕복 포함해서 캡차 `verify` 응답이 정상 시간 안에 떨어집니다.
풀이 1건당 40행 규모라 대량 쓰기라고 할 것도 아닙니다. 초당 수백 건이 아니라
**하루 수백 건 규모**입니다.

즉 **옮기는 방향이 반대이고, 옮길 이유였던 성능 우려도 실재하지 않습니다.**

---

## 2. 크로스 조인이 자주 필요한가 — 거의 필요 없습니다

> participant별 FRR은 ai_* 단독으로 계산되는 걸로 보이는데,
> 캡차측 데이터와 실제로 조인해야 할 분석이 있나요?

**맞습니다. `ai_*` 단독으로 됩니다.** 그리고 캡차 판정도 이미 제 쪽에 넘어와
있습니다.

```sql
-- ai_shadow_outcomes
main_captcha_verdict   passed | failed     ← 캡차의 정답 판정
final_verdict          shadow 적용 후 최종
would_have_action      모델이 권했을 조치
```

캡차가 `POST /api/v1/behavior/shadow/outcomes` 로 판정을 넘겨주기 때문에,
**"모델 점수 × 캡차 판정" 교차분석이 제 스키마 안에서 끝납니다.** 이게 참여자별
FRR 계산의 핵심 조인인데 이미 한 곳에 있습니다.

캡차 쪽에만 있는 것 중 제가 아쉬운 건 하나뿐입니다 — **원시 배치의 수신 시각**
(`captcha_behavior_batches.received_at`). 배치 전송 타이밍 분석에 쓸 수 있는데,
지금은 `detect_batch_delivery_timing` 이 캡차 안에서 계산해 요약만 넘깁니다.
필요해지면 그때 요약 필드를 하나 더 실어 보내면 되고, 조인할 일은 아닙니다.

**결론: 상관키 correlate 로 충분합니다.** 합병도 co-locate 도 지금은 이득이 없습니다.

---

## 3. 그런데 진짜 문제는 따로 있습니다

co-locate 보다 먼저 정리할 게 있습니다. **`ai_*` 가 백엔드 스키마 안에,
백엔드 계정으로 들어가 있습니다.**

```
DB      catchap_dev_db        ← 백엔드 소유 스키마
계정    catchap_backend       ← 백엔드 런타임 계정을 공유
권한    SELECT, INSERT, UPDATE, DELETE on catchap_dev_db.*
        (USAGE on *.* — DDL 없음)
```

민서님이 걱정하신 "소유 경계·격리" 문제가 **제가 만든 쪽에 이미 있습니다.**
구체적으로 세 가지가 걸립니다.

**(1) 자격증명 공유** — 백엔드와 같은 계정을 씁니다. 팀 할 일 목록에 있는
"시크릿·관리자키 로테이션" 을 하면 **제 서비스가 같이 죽습니다.** 로테이션할 때
제 쪽도 같이 바꿔야 한다는 걸 아무도 모르는 상태입니다.

**(2) 스키마 소유가 섞임** — `catchap_dev_db` 안에 백엔드 테이블과 `ai_*` 가
같이 있습니다. 백엔드가 스키마를 정리하다가 `ai_*` 를 건드릴 여지가 있고,
반대로 제 테이블 때문에 백엔드 마이그레이션이 헷갈릴 수 있습니다.

**(3) DDL 권한 없음** — 제가 컬럼 하나 추가하려면 DB 담당자를 불러야 합니다.
`bot_suspicion` 때 겪은 절차를 모델 스키마 바꿀 때마다 밟게 됩니다.

### 제안 — co-locate 대신 이걸 하는 게 맞다고 봅니다

```sql
CREATE DATABASE catchap_ai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'catchap_ai'@'10.0.1.%' IDENTIFIED BY '<새 비밀번호>';
GRANT ALL PRIVILEGES ON catchap_ai.* TO 'catchap_ai'@'10.0.1.%';
```

- **같은 서버(10.0.1.168), 다른 스키마** — 민서님이 제안하신 "같은 서버 두 스키마"
  구조가 그대로 됩니다. 다만 짝이 `captcha_ms` 가 아니라 `catchap_dev_db` 입니다
- 제 서비스는 **접속 설정 3줄만** 바뀝니다(`MYSQL_DATABASE`/`USER`/`PASSWORD`)
- 백엔드 계정 로테이션이 저를 안 깨뜨립니다
- 제 스키마는 제가 관리합니다 (DDL 권한)
- 기존 7행은 버려도 되는 테스트 데이터라 **마이그레이션 없이 새로 시작하면 됩니다**

이건 DB 담당자 작업이고, 제 쪽 변경은 재기동 한 번입니다.

---

## 4. 네 질문에 짧게

| 질문 | 답 |
|---|---|
| 크로스DB 조인이 자주 필요한가 | **아니요.** 캡차 판정은 `ai_shadow_outcomes` 로 이미 넘어와 있어 제 스키마 안에서 끝납니다 |
| 내부망 왕복이 채점 지연을 만드나 | **아니요. 이미 그렇게 돌고 있습니다.** 풀이당 40행, 하루 수백 건 규모입니다 |
| GPU 로컬이 성능·격리상 꼭 필요한가 | 제 쪽은 GPU 로컬을 안 씁니다. **`captcha_ms` 가 GPU 로컬입니다** — 그건 민서님 판단 영역입니다 |
| 별도 유지 + correlate 가 나은가 | **네.** 다만 3절(스키마·계정 분리)은 별개로 필요합니다 |

---

## 5. 하나 덧붙이면

`captcha_ms` 가 GPU 로컬이라는 점은 제 영역은 아니지만, 한 가지만 말씀드립니다.
**GPU 서버는 모델 학습·추론이 도는 곳이라 디스크·메모리 압력이 큽니다.** 어제도
디스크가 꽉 차서 18G 확보하셨다고 하셨는데, 그때 로컬 MySQL 도 같은 디스크를
씁니다. 실서비스 캡차 데이터가 거기 있으면 학습 작업 하나가 DB 를 멈출 수
있습니다.

지금 판단하실 일은 아니고, **공개 개방 전에 한 번 짚어보시면 좋겠습니다.**

---

문의: 조성원 (wwdhogo@gmail.com)
