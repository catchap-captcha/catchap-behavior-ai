"""운영 DB 의 배치를 조준 연구 도구가 읽는 모양으로 꺼낸다.

왜 변환이 필요한가
------------------
조준 구간은 원래 로컬 수집기(`/collect-aim`)로만 받았고, 그때는 한 행이
`{aim_events: [...], drag_events: [...]}` 로 이미 갈라져 있었다. `aim_drag_path.join`
을 비롯한 연구 도구 전부가 그 모양을 기대한다.

2026-08-12 부터 조준이 캡차 서버의 배치에 함께 실려 온다(`catchap-captcha#20`, `#22`).
운영에 로컬 수집기가 없어 조준이 기록만 되고 사라지던 것을 고친 결과다. 대신 이제는
조준과 드래그가 **하나의 순서열**로 저장된다 — 짝을 맞출 키가 필요 없어진 대신,
옛 도구가 그대로는 못 읽는다. 이 도구가 그 사이를 잇는다.

    captcha_behavior_batches.events_json  →  {aim_events, drag_events} 행

가르는 규칙
-----------
한 챌린지의 이벤트를 seq 순으로 훑으며 `pointer_down` 을 경계로 삼는다. 그 앞의
`aim_move` 가 조준, 뒤의 `pointer_move`~`drop` 이 그 드래그다. 드래그가 여러 번이면
각각을 한 행으로 낸다 — 조준은 첫 집기 앞 구간만 있으므로 두 번째 드래그부터는
조준이 비고, 그건 정상이다(사람은 한 번 조준하고 여러 번 끌기도 한다).

봉인 참가자는 거른다. `collection_split_20260806.json` 의 holdout 은 최종 검증용이라
학습 경로에 한 번이라도 들어가면 그 검증이 무의미해진다.

**참가자는 `catchap_ai.ai_behavior_attempts.participant_id` 에서 문항 단위로 가져온다.**
세션 ID 로는 사람을 못 가른다 — 로그인 캡차의 세션 ID 는 `guard-<시각>-<난수>` 라
사람 정보가 없고(그래서 앞글자로 거르던 옛 방식은 봉인 필터가 한 번도 안 걸렸다),
수집 페이지에서도 한 세션을 두 사람이 이어 쓴 사례가 실제로 있었다(2026-08-12:
`2b4bdda8` = jk+nn, `a696efa7` = jy+wg). 문항 하나에 코드가 둘 붙은 경우는 없다.

    .venv/bin/python tools/export_aim_from_production.py \
        --password-file <file> --since '2026-08-12 04:30:00' \
        --out data/interim/aim_production_20260812.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "data" / "metadata" / "collection_split_20260806.json"
JUMP = "sw@210.109.82.148"
DB_HOST = "10.0.2.193"

# 조준은 집기 전 구간이다. 집은 뒤의 이동은 드래그다.
GRAB = "pointer_down"
DRAG_TYPES = {"pointer_move", "drop"}


QUERY = """
SELECT c.id, c.session_id, COALESCE(a.participant_id, ''), b.batch_seq, b.events_json
FROM captcha_challenges_v2 c
JOIN captcha_behavior_batches b ON b.challenge_id = c.id
LEFT JOIN (SELECT challenge_id, MAX(participant_id) AS participant_id
           FROM catchap_ai.ai_behavior_attempts
           GROUP BY challenge_id) a ON a.challenge_id = c.id
WHERE c.created_at >= '{since}'
ORDER BY c.id, b.batch_seq;
"""


def fetch(sql: str, key: Path, password_file: Path) -> str:
    """비밀번호를 명령줄에 두지 않고 원격에서 한 번 조회한다(tools/q.py 와 같은 방식)."""
    password = password_file.read_text().strip()
    payload = (f"[client]\nuser=sw\npassword={password}\nhost={DB_HOST}\nport=3306\n"
               "---SQL---\n" + sql)
    remote = (
        'umask 077; A=$(mktemp); cat > "$A"; chmod 600 "$A"; '
        'sed -n "1,/---SQL---/p" "$A" | grep -v -- "---SQL---" > ~/.mycnf.tmp; chmod 600 ~/.mycnf.tmp; '
        'sed -n "/---SQL---/,\\$p" "$A" | tail -n +2 > ~/q.sql; '
        'mysql --defaults-extra-file=$HOME/.mycnf.tmp -B --raw catchap_captcha < ~/q.sql; '
        'rm -f ~/.mycnf.tmp ~/q.sql "$A"'
    )
    result = subprocess.run(
        ["ssh", "-i", str(key), "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", JUMP, remote],
        input=payload, capture_output=True, text=True, check=False)
    if not result.stdout:
        raise SystemExit(result.stderr or "조회가 아무것도 돌려주지 않았다")
    return result.stdout


def is_sealed(person: str | None, sealed: set[str]) -> bool:
    """봉인 참가자인가. **정확히 일치로 보면 안 된다.**

    명단(`collection_split_20260806.json`)에는 사람 코드가 `sw`·`ms` 로 적혀 있는데,
    운영에 실제로 들어온 값은 목적별로 갈라져 있다.

        ms · ms-captcha · sw-mouse · sw-captcha · sw-mouse-v2 · sw-aim

    정확히 일치로 거르면 `ms` 46건만 빠지고 나머지 537건이 학습 자료에 남는다.
    2026-08-13 에 인프라 담당자가 잡았다 — 내 도구도 같은 구멍이었다. 다행히 내가 쓴
    구간(08-12 04:29 이후)에는 봉인 기록이 0건이라 실제 오염은 없었다.

    뒤가 글자·숫자로 이어지면 다른 사람일 수 있으므로 거르지 않는다(`swan`·`msg`).
    """
    if not person:
        return False
    for code in sealed:
        if person == code:
            return True
        if person.startswith(code) and not person[len(code):len(code) + 1].isalnum():
            return True
    return False


def split_runs(events: list[dict]) -> list[dict]:
    """한 챌린지의 순서열을 (조준, 드래그) 쌍들로 가른다."""
    events = sorted(events, key=lambda e: (e.get("seq") if e.get("seq") is not None else 0))
    rows: list[dict] = []
    aim: list[dict] = []
    drag: list[dict] | None = None
    for event in events:
        kind = event.get("type")
        if kind == "aim_move":
            # 드래그 중에는 조준이 기록되지 않지만, 드래그가 끝난 뒤 다음 집기까지의
            # 이동은 다시 조준이다. 그 구간은 다음 행의 조준이 된다.
            if drag is not None:
                rows.append({"aim_events": aim, "drag_events": drag})
                aim, drag = [], None
            aim.append(event)
        elif kind == GRAB:
            if drag is not None:
                rows.append({"aim_events": aim, "drag_events": drag})
                aim = []
            drag = []
        elif kind in DRAG_TYPES and drag is not None:
            drag.append(event)
    if drag is not None:
        rows.append({"aim_events": aim, "drag_events": drag})
    return [r for r in rows if r["drag_events"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--password-file", required=True, type=Path)
    ap.add_argument("--since", required=True, help="UTC 'YYYY-MM-DD HH:MM:SS'")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--key", type=Path,
                    default=Path.home() / "finally project/keypair/catchap_keypair")
    ap.add_argument("--require-participant", action="store_true",
                    help="참가자 코드가 없는 문항도 뺀다. 봉인 참가자가 코드를 안 붙이고 "
                         "푼 경우까지 막으려면 학습 경로에서는 이걸 쓴다.")
    args = ap.parse_args()

    sealed = set(json.loads(SPLIT.read_text())["holdout_people"])
    raw = fetch(QUERY.format(since=args.since), args.key, args.password_file)

    by_challenge: dict[str, dict] = {}
    for line in raw.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        cid, session, participant = parts[0], parts[1], parts[2].strip()
        events_json = "\t".join(parts[4:])
        try:
            events = json.loads(events_json.replace("\\n", "\n"))
        except json.JSONDecodeError:
            continue
        slot = by_challenge.setdefault(
            cid, {"session_id": session, "participant_id": participant, "events": []})
        slot["events"].extend(events)

    written = skipped_sealed = skipped_unknown = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        for cid, slot in by_challenge.items():
            person = slot["participant_id"]
            if is_sealed(person, sealed):
                skipped_sealed += 1
                continue
            if not person:
                # 코드가 없으면 봉인 참가자인지 아닌지를 확인할 방법이 없다. 연구용
                # 측정에는 써도 되지만 학습 경로에서는 --require-participant 로 뺀다.
                skipped_unknown += 1
                if args.require_participant:
                    continue
            for row in split_runs(slot["events"]):
                row["challenge_id"] = cid
                row["session_id"] = slot["session_id"]
                row["participant_id"] = person or None
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    kept_unknown = "제외" if args.require_participant else "포함"
    print(f"  챌린지 {len(by_challenge)}개 · 봉인 제외 {skipped_sealed}개 "
          f"· 코드없음 {skipped_unknown}개({kept_unknown}) · 기록 {written}행")
    print(f"  -> {args.out}")
    if written:
        sys.path.insert(0, str(ROOT))
        from tools.aim_drag_path import join_record, summarize  # noqa: E402
        rows = [json.loads(l) for l in args.out.read_text().splitlines()]
        stats = summarize([join_record(r) for r in rows])
        print(f"  이어붙인 경로: 중앙 {stats.get('median_points', 0):.0f}점 "
              f"(조준 {stats.get('median_aim_points', 0):.0f} + 드래그 "
              f"{stats.get('median_drag_points', 0):.0f})")
        clears = sum(1 for r in rows if join_record(r).total_points >= 31)
        print(f"  31점 이상: {clears}/{len(rows)}  ({clears / len(rows) * 100:.1f}%)")
        print("  → 31점이 변형 재생 검출이 96.8% 로 올라가는 경계다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
