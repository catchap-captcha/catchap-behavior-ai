"""승급 로직 단위 검증 — DB·FastAPI 없이 순수 로직만 확인한다."""
import sys, types, logging
from dataclasses import dataclass

# ---- 최소 스텁: lecture_service 가 임포트하는 것들만 ----
def _mk(name, **attrs):
    m = types.ModuleType(name); [setattr(m, k, v) for k, v in attrs.items()]; sys.modules[name] = m; return m

@dataclass
class _Settings:
    BOT_ESCALATION_MODE: str = "record"
    BOT_SUSPICION_THRESHOLD: int = 10
    MAIN_CAPTCHA_URL: str = "https://captcha.example"
    MAIN_CAPTCHA_SITE_SECRET: str = "s3cret"

SETTINGS = _Settings()
_mk("app"); _mk("app.core")
_mk("app.core.config", get_settings=lambda: SETTINGS)
_mk("app.db"); _mk("app.db.base", _now=lambda: __import__("datetime").datetime(2026,7,29,12,0,0))
_mk("app.models", **{n: type(n, (), {}) for n in
    ["Lecture","LectureCheckpointEvent","LectureMaterial","LectureQuestion",
     "LectureQuestionGenJob","LectureReview","LectureTranscript","LectureWatchProgress"]})

sys.path.insert(0, "/private/tmp/claude-501/-Users-apple-Documents----/5d3fc21e-53e5-49ae-9182-8aeaed0b6968/scratchpad/backend-work")
import lecture_service as ls

class P:
    def __init__(s): s.bot_suspicion = 0; s.student_id = "stu1"; s.lecture_id = "lec1"

logging.disable(logging.CRITICAL)
fails = 0
def check(label, got, want):
    global fails
    ok = got == want
    if not ok: fails += 1
    print(f"  {'✅' if ok else '❌'} {label}: {got} (기대 {want})")

print("=== 가산·상한 ===")
p = P()
ls.bump_suspicion(p, ls.SUSPICION_SPEED_VIOLATION, "t"); check("속도위반 1회", p.bot_suspicion, 3)
ls.bump_suspicion(p, ls.SUSPICION_SESSION_CONFLICT, "t"); check("동시접속 추가", p.bot_suspicion, 8)
for _ in range(20): ls.bump_suspicion(p, 5, "t")
check("상한 적용", p.bot_suspicion, ls.SUSPICION_MAX)

print("\n=== 임계 판정 ===")
p = P(); p.bot_suspicion = 9
check("임계 미달(9<10)", ls.captcha_required(p), False)
p.bot_suspicion = 10
check("임계 도달(10)", ls.captcha_required(p), True)

print("\n=== off 모드에서는 아무 일도 없어야 ===")
SETTINGS.BOT_ESCALATION_MODE = "off"
p = P(); ls.bump_suspicion(p, 9, "t")
check("off: 가산 안 됨", p.bot_suspicion, 0)
check("off: 판정 False", ls.captcha_required(p), False)

print("\n=== 설정 누락 시 off 로 강등 ===")
SETTINGS.BOT_ESCALATION_MODE = "enforce"; SETTINGS.MAIN_CAPTCHA_URL = ""
check("URL 없으면 off", ls._escalation_mode(), "off")
SETTINGS.MAIN_CAPTCHA_URL = "https://captcha.example"; SETTINGS.MAIN_CAPTCHA_SITE_SECRET = ""
check("시크릿 없으면 off", ls._escalation_mode(), "off")
SETTINGS.MAIN_CAPTCHA_SITE_SECRET = "s3cret"
check("둘 다 있으면 enforce", ls._escalation_mode(), "enforce")
SETTINGS.BOT_ESCALATION_MODE = "garbage"
check("알 수 없는 값은 off", ls._escalation_mode(), "off")

print("\n=== 리셋 ===")
SETTINGS.BOT_ESCALATION_MODE = "record"
p = P(); p.bot_suspicion = 25; ls.clear_suspicion(p)
check("통과 후 0", p.bot_suspicion, 0)

print(f"\n{'모두 통과' if fails==0 else f'{fails}건 실패'}")
sys.exit(1 if fails else 0)
