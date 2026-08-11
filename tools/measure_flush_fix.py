"""Measure what the flush fix changes, against predictions made before it shipped.

Why the predictions are written down here
-----------------------------------------
Once the numbers are in, any of them can be explained after the fact. So the two
expectations are recorded in the code, before deployment, and the tool prints
them next to what it measures:

  ② The attempts that start being scored should look like the ones that were
     being lost: 2,986ms and 0.65 drags on average, against 7,020ms and 1.53 for
     the ones that already scored.

  ③ The false-reject rate should rise from 8.5% to about 9.1%. The 100 lost
     trajectories were pulled from `captcha_behavior_batches` and scored with the
     new bundle ahead of time: they come out at 16.0%, and weighting that against
     8.5% over 1,110 already-scored attempts gives 9.1%.

Reading ③ well matters more than ③ being right. If it lands far above 9.1%, the
100 recovered trajectories did not represent the loss — and the most likely
reason is `behavior_batches_missing` (78 attempts with zero batches), which could
not be included because no trajectory was ever stored for them. That would make
it a third population rather than a miss in the arithmetic.

Background: `flushBehavior` returned an in-flight promise that predated the
`submit` event, so `await flushBehavior()` in verify resolved without having sent
it and the server rejected the attempt as `behavior_lifecycle_missing_submit`.
The loss was not random — it selected fast solvers, who are also the ones the
model is most likely to doubt.

    .venv/bin/python tools/measure_flush_fix.py --since '2026-08-11 03:00:00'
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Measured 2026-08-11, before the fix shipped. Do not update these to match a
# later observation — that is what makes them a prediction.
BASELINE = {
    "lost_duration_ms": 2986.0,
    "lost_drags": 0.65,
    "scored_duration_ms": 7020.0,
    "scored_drags": 1.53,
    "frr_scored": 8.5,
    "frr_lost_rescored": 16.0,
    "frr_predicted": 9.1,
    "loss_rate": 9.5,
}

JUMP = "sw@210.109.82.148"
DB_HOST = "10.0.2.193"

QUERY = """
SELECT
  CASE WHEN p.status = 'scored' THEN 'scored' ELSE 'lost' END AS bucket,
  COUNT(*)                                            AS n,
  ROUND(AVG(a.duration_ms))                           AS avg_duration_ms,
  ROUND(AVG(JSON_EXTRACT(a.behavior_summary, '$.drag_count')), 2) AS avg_drags,
  ROUND(100 * SUM(p.recommended_action <> 'allow')
        / NULLIF(SUM(p.status = 'scored'), 0), 1)     AS flagged_pct
FROM behavior_shadow_predictions p
JOIN captcha_attempts a ON a.id = p.captcha_attempt_id
WHERE a.failure_reason IS NULL
  AND p.created_at >= '{since}'
  AND (p.status = 'scored'
       OR p.detail IN ('behavior_lifecycle_missing_submit',
                       'behavior_lifecycle_missing_load',
                       'behavior_action_binding_missing'))
GROUP BY 1;
"""


def run_remote(sql: str, key: Path, password_file: Path) -> str:
    """Run one query through the jump host without putting the password on a command line."""
    password = password_file.read_text().strip()
    payload = (f"[client]\nuser=sw\npassword={password}\nhost={DB_HOST}\nport=3306\n"
               "---SQL---\n" + sql)
    remote = (
        'umask 077; A=$(mktemp); cat > "$A"; chmod 600 "$A"; '
        'sed -n "1,/---SQL---/p" "$A" | grep -v -- "---SQL---" > ~/.mycnf.tmp; chmod 600 ~/.mycnf.tmp; '
        # $p 는 원격 셸이 자기 변수로 먹으므로 이스케이프한다.
        'sed -n "/---SQL---/,\\$p" "$A" | tail -n +2 > ~/q.sql; '
        'mysql --defaults-extra-file=$HOME/.mycnf.tmp -t catchap_captcha < ~/q.sql; '
        'rm -f ~/.mycnf.tmp ~/q.sql "$A"'
    )
    result = subprocess.run(
        ["ssh", "-i", str(key), "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", JUMP, remote],
        input=payload, capture_output=True, text=True, check=False)
    return result.stdout or result.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="배포 시각 이후만 본다 (YYYY-MM-DD HH:MM:SS)")
    ap.add_argument("--key", type=Path,
                    default=Path.home() / "finally project/keypair/catchap_keypair")
    ap.add_argument("--password-file", type=Path, required=True,
                    help="DB 비밀번호가 든 파일. 명령줄에 비밀번호를 두지 않기 위해서다")
    args = ap.parse_args()

    print("배포 전 예측 (2026-08-11 기록, 고치지 말 것)")
    print(f"  ② 유실되던 쪽   {BASELINE['lost_duration_ms']:.0f}ms · 드래그 {BASELINE['lost_drags']}")
    print(f"     채점되던 쪽   {BASELINE['scored_duration_ms']:.0f}ms · 드래그 {BASELINE['scored_drags']}")
    print(f"  ③ 오탐          {BASELINE['frr_scored']}% → 약 {BASELINE['frr_predicted']}% "
          f"(유실분 재채점 {BASELINE['frr_lost_rescored']}% · 유실 비중 {BASELINE['loss_rate']}%)\n")

    print(f"배포 후 실측 ({args.since} 이후)")
    print(run_remote(QUERY.format(since=args.since), args.key, args.password_file))

    print("읽는 법")
    print("  · lost 가 0 에 가까우면 flush 수정이 들은 것이다.")
    print("  · scored 의 평균 소요·드래그가 내려갔으면 ② 가 맞은 것 — 짧은 부류가 합류했다.")
    print(f"  · flagged 가 {BASELINE['frr_predicted']}% 를 크게 넘으면 ③ 의 전제가 틀린 것이다.")
    print("    가장 유력한 이유는 batches_missing(배치 0개) 78건이다. 궤적이 저장되지 않아")
    print("    미리 채점할 수 없었고, 그래서 예측에 들어가지 못했다 — 세 번째 부류일 수 있다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
