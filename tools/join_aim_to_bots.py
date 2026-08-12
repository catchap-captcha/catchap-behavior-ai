"""봇 궤적 앞에 조준 구간을 붙인다 — 그리고 그 이음매가 사람과 구분되지 않는지 잰다.

왜 필요한가
-----------
분류기 오탐 6.2%p 는 짧고 곧은 드래그에 몰려 있다(2026-08-12, 실사용 1,167건:
튕긴 시도의 드래그 10.9점 vs 통과 15.2점). 그런데 그 시도들의 조준 구간은 멀쩡하다
(19.2점 vs 20.1점) — 판단 근거가 없는 게 아니라 분류기가 그 구간을 못 보고 있다.

조준을 분류기에 넣으려면 봇에도 조준이 있어야 한다. 봇 계열 전부와 옛 사람 수집분은
집기 전 이동이 0점이다. 사람만 조준이 있는 채로 학습하면 모델은 "조준 있으면 사람"을
배운다 — 그건 검출이 아니라 데이터 누수다.

무엇이 위험한가
---------------
조준을 합성해 붙이면, 모델이 봇의 *행동* 이 아니라 내 *합성 흔적* 을 배울 수 있다.
그러면 평가 숫자는 좋아지고 실제 방어력은 그대로다. 그래서 이 도구는 붙이는 일보다
**안 들키게 붙었는지 재는 일**에 무게를 둔다. 이음매에서 새는 것이 있으면
`--verify` 가 그것을 AUC 로 보여준다. 0.5 에서 멀면 붙이는 방식이 틀린 것이다.

이음매를 사람에게서 그대로 가져온다
-----------------------------------
실사용 1,074건에서 잰 값(정규화 좌표):

    공간 간격  중앙 0.0019 · p90 0.0164     조준 끝과 집기 지점은 정확히 같지 않다
    시간 간격  중앙 217ms · p10 67 · p90 475   집기 직전에 멈칫한다
    조준 점수  중앙 16점 · p10 7 · p90 34
    조준 변위  중앙 0.559 · p10 0.252 · p90 0.840

이 넷을 봇에도 같은 분포로 준다. "간격 정확히 0" 이나 "항상 20점" 같은 값은 그 자체가
표식이 되므로, 상수가 아니라 사람 표본에서 재추출한다.

    .venv/bin/python tools/join_aim_to_bots.py \\
        --bots data/interim/extended_bots_10000_20260721.jsonl \\
        --family B_eased --out data/interim/joined/extended_B.jsonl --verify
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import make_aim_bots as mab  # noqa: E402
from tools.aim_segments import to_arrays  # noqa: E402

HUMAN_JOINED = ROOT / "data" / "interim" / "aim_production_20260812.jsonl"
FAMILIES = mab.FAMILIES


# ---------------------------------------------------------------- 사람 기준값

class HumanSeam:
    """사람에게서 잰 이음매 분포. 봇에 그대로 나눠준다."""

    def __init__(self, path: Path) -> None:
        self.gap_xy: list[float] = []
        self.gap_ms: list[float] = []
        self.points: list[int] = []
        self.displacement: list[float] = []
        self.duration: list[float] = []
        self.aims: list[list[dict]] = []
        # 한 조준의 네 값은 서로 얽혀 있다 — 멀리 뻗으면 오래 걸린다. 따로 뽑으면
        # 그 상관이 끊기고, 끊긴 것 자체가 표식이 된다(D_replay 가 `linearity`
        # 하나로 AUC 0.695 에 갈렸다). 그래서 표본 단위로 묶어 둔다.
        self.rows: list[dict] = []

        for line in path.read_text().splitlines():
            row = json.loads(line)
            # 조준을 빌려준 사람을 기록해 둔다. D_replay 봇은 진짜 사람 조준을 그대로
            # 쓰므로, 그 사람이 학습에 들어가 있으면 같은 조준이 한쪽은 사람 한쪽은
            # 봇으로 들어간다. 폴드를 나눌 때 이 사람 기준으로 묶어야 그게 막힌다.
            row_person = row.get("participant_id")
            aim, drag = row.get("aim_events") or [], row.get("drag_events") or []
            if len(aim) < 2 or len(drag) < 2:
                continue
            if aim[-1].get("x") is None or drag[0].get("x") is None:
                continue
            t0, t1 = aim[-1].get("timestamp_ms"), drag[0].get("timestamp_ms")
            span = float(aim[-1].get("timestamp_ms") or 0) - float(aim[0].get("timestamp_ms") or 0)
            if not (t0 and t1 and 0 <= float(t1) - float(t0) <= 5000):
                continue
            if not 0 < span <= 30000:
                continue

            self.aims.append(aim)
            row = {"person": row_person,
                   "gap_xy": _dist(aim[-1], drag[0]),
                   "gap_ms": float(t1) - float(t0),
                   "displacement": _dist(aim[0], aim[-1]),
                   "duration": span,
                   # 크기만이 아니라 **방향까지** 사람에게서 가져온다. 각도를 무작위로
                   # 뽑으면 화면 밖으로 나가는 배치가 늘고, `d_replay` 는 밖으로
                   # 나가는 본보기를 버리므로 곧은 조준만 살아남는다 — 그 편향이
                   # `linearity` AUC 0.69 로 나왔다. 사람은 아무 방향에서나 오지 않는다.
                   "offset": (float(aim[0]["x"]) - float(aim[-1]["x"]),
                              float(aim[0]["y"]) - float(aim[-1]["y"])),
                   # 집는 지점. 조준이 어디서 오는지는 목표가 어디냐에 달려 있다 —
                   # 왼쪽 끝을 집는 사람은 오른쪽에서 온다. 목표를 무시하고 뽑으면
                   # 화면 밖 배치가 늘고, 그걸 버리느라 표본이 편향된다.
                   "grab": (float(drag[0]["x"]), float(drag[0]["y"])),
                   # 조준 끝에서 집는 지점까지의 벡터. 크기만 뽑아 임의 방향으로
                   # 두면 방향 분포가 사람과 어긋나므로 벡터째 쓴다.
                   "seam_vec": (float(drag[0]["x"]) - float(aim[-1]["x"]),
                                float(drag[0]["y"]) - float(aim[-1]["y"])),
                   # D_replay 는 이 표본 **자신의** 조준을 되쓴다. 아무 조준이나 골라
                   # 돌리고 늘리면 화면을 벗어나는 것이 버려지는데, 버려지는 쪽이
                   # 헤매는 조준(직진도 0.337)이고 남는 쪽이 곧은 조준(0.706)이라
                   # 사람 전체(0.628)보다 곧아진다 — 그게 `linearity` AUC 0.69 였다.
                   # 자기 것을 쓰면 회전·확대가 없어 버릴 일이 없다. 캡처 라이브러리를
                   # 든 공격자도 목표가 같은 자리인 기록을 고르지, 90도 돌려 쓰지 않는다.
                   "aim": aim}
            self.rows.append(row)
            self.gap_xy.append(row["gap_xy"])
            self.gap_ms.append(row["gap_ms"])
            self.displacement.append(row["displacement"])
            self.duration.append(row["duration"])
            self.points.append(len(aim))

        if not self.rows:
            raise SystemExit("사람 이음매 표본이 없다 — 조준 내보내기를 먼저 하라")

        # 집는 지점을 4x4 로 나눠 담아둔다. 봇의 집는 지점과 같은 칸에서 뽑으면
        # 사람에게 실제로 가능했던 기하만 나오므로, 화면 밖 배치를 버리는 일 자체가
        # 거의 사라진다 — 버리는 순간 표본이 편향되기 때문에 안 버리는 편이 낫다.
        self.buckets: dict[tuple[int, int], list[dict]] = {}
        for row in self.rows:
            self.buckets.setdefault(_cell(*row["grab"]), []).append(row)
        self._grabs = np.array([r["grab"] for r in self.rows], dtype=float)

    def draw(self, rng: random.Random) -> dict:
        """사람 조준 하나를 **그대로** 뽑는다. 조건을 걸지 않는다.

        조준을 봇의 집는 지점에 맞추려던 시도는 전부 같은 자리에서 무너졌다. 옮기면
        일부가 화면을 벗어나고, 벗어난 것을 버리면 그 버림이 곧 편향이 된다 —
        크게 휘는 조준(직진도 0.337 vs 0.706), 지나쳤다 돌아오는 조준(overshoot
        0.0074 vs 0.0056)이 먼저 사라졌다. 가까운 기록만 고르게 하면 이번엔 같은
        기록이 반복돼 변위가 새어나갔다(AUC 0.721).

        그래서 방향을 뒤집는다. 조준은 손대지 않고, **봇 드래그를 조준 끝으로 옮긴다**
        (`join_record`). 평행이동은 드래그의 모양·속도를 바꾸지 않으므로 봇은 그대로
        남고, 조준은 편향 없는 사람 표본이 된다.
        """
        return dict(rng.choice(self.rows))


def _dist(a: dict, b: dict) -> float:
    return float(np.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"])))


def _cell(x: float, y: float, n: int = 4) -> tuple[int, int]:
    """화면을 n x n 으로 나눈 칸. 조준 기하를 목표 위치별로 나눠 담는 데 쓴다."""
    clamp = lambda v: min(n - 1, max(0, int(v * n)))
    return clamp(x), clamp(y)


# ---------------------------------------------------------------- 잇기

def _norm_xy(event: dict) -> tuple[float, float] | None:
    x = event.get("x_normalized")
    y = event.get("y_normalized")
    if x is None or y is None:
        return None
    return float(x), float(y)


def replay_aim(seam: HumanSeam, rng: random.Random) -> tuple[list[dict], dict]:
    """사람 조준을 손대지 않고 그대로 쓴다 (D_replay).

    좌표도 시각도 원본 그대로다. 이 궤적을 어디로도 옮기지 않기 때문에 화면을
    벗어날 일이 없고, 따라서 버릴 일도 없다 — 편향이 들어올 자리가 없다.
    """
    row = seam.draw(rng)
    base = float(row["aim"][0].get("timestamp_ms") or 0.0)
    events = [{"event_type": "aimmove",
               "t_ms": float(e.get("timestamp_ms") or 0.0) - base,
               "x_normalized": float(e["x"]), "y_normalized": float(e["y"])}
              for e in row["aim"] if e.get("x") is not None]
    return events, row


def build_aim(family: str, seam: HumanSeam,
              rng: random.Random) -> tuple[list[dict], dict] | None:
    """사람이 실제로 집었던 자리에서 끝나는 조준 구간을 만든다.

    봇의 집는 지점에 맞추지 않는다 — 맞추려면 옮겨야 하고, 옮기면 일부가 화면을
    벗어나고, 벗어난 것을 버리면 그 버림이 편향이 된다(`draw` 주석 참고). 사람의
    자리에서 만들면 화면 안이 보장되고, 위치 맞추기는 나중에 조준·드래그를 함께
    미는 것으로 끝난다. 평행이동은 모양 특징을 바꾸지 않는다.
    """
    drawn = seam.draw(rng)
    target = np.array(drawn["grab"], dtype=float)

    # 끝점·시작점 모두 이 사람이 실제로 지나간 자리다. 각도를 무작위로 뽑던 앞 판은
    # 화면 밖 배치를 만들고 그것을 버리느라 방향·변위 분포를 망가뜨렸다.
    end = np.array([float(drawn["aim"][-1]["x"]), float(drawn["aim"][-1]["y"])])
    start = end + np.array(drawn["offset"], dtype=float)

    events = mab.GENERATORS[family](start, end, drawn["duration"], rng, seam.aims)
    if len(events) < 4:
        return None

    # 마지막으로 **살아남은** 점을 목표 지점에 맞춘다.
    #
    # 생성기는 `end` 까지 그리지만 40ms throttle 이 끝부분 표본을 떨궈서, 실제 마지막
    # 점은 목표에 못 미친다. 그대로 두면 이음매 간격이 사람의 2.5배가 되고 그 값
    # 하나로 봇이 AUC 0.632 에 갈렸다(2026-08-12 확인). 경로를 통째로 옮겨 모양은
    # 두고 끝점만 맞춘다.
    dx = end[0] - float(events[-1]["x"])
    dy = end[1] - float(events[-1]["y"])
    for e in events:
        e["x"] = float(np.clip(float(e["x"]) + dx, 0.0, 1.0))
        e["y"] = float(np.clip(float(e["y"]) + dy, 0.0, 1.0))

    # 시각은 조준 시작을 0 으로 둔다. 드래그를 여기에 붙이는 일은 `join_record` 가
    # 사람 이음매 시간(`gap_ms`)만큼 띄워서 한다 — 네 계열 모두 같은 경로다.
    base = float(events[0]["timestamp_ms"])
    aim = [{"event_type": "aimmove", "t_ms": float(e["timestamp_ms"]) - base,
            "x_normalized": float(e["x"]), "y_normalized": float(e["y"])}
           for e in events]
    return aim, drawn


LIMIT = 0.02  # 화면 밖 허용 폭. 위젯이 가장자리에서 잡아주는 만큼만 둔다.


def _shift_drag(drag: list[dict], to_xy: tuple[float, float], to_t: float,
                width: float, height: float, *, bounded: bool = True) -> list[dict] | None:
    """드래그를 통째로 옮긴다. 평행이동이라 모양·속도·간격은 그대로다."""
    x0, y0 = _norm_xy(drag[0])
    dx, dy = to_xy[0] - x0, to_xy[1] - y0
    dt = to_t - float(drag[0].get("t_ms") or 0.0)
    out = []
    for e in drag:
        nx, ny = _norm_xy(e)
        nx, ny = nx + dx, ny + dy
        if bounded and not (-LIMIT <= nx <= 1 + LIMIT and -LIMIT <= ny <= 1 + LIMIT):
            return None
        out.append({**e, "x_normalized": nx, "y_normalized": ny,
                    "x": nx * width, "y": ny * height,
                    "t_ms": float(e.get("t_ms") or 0.0) + dt})
    return out


def _fit_into_stage(aim: list[dict], drag: list[dict] | None):
    """이어붙인 궤적 전체를 화면 안으로 최소한만 민다.

    조준과 드래그를 **같이** 옮기므로 둘의 상대 위치가 유지되고, 모양 특징은
    평행이동에 영향받지 않는다. 전체 폭이 화면보다 크면 어떤 위치로도 못 넣으니
    그때만 버린다 — 그 버림은 봇 드래그 크기에서 오는 것이라 사람 조준을 편향시키지
    않는다.
    """
    if drag is None:
        return None
    xs = [e["x_normalized"] for e in aim] + [e["x_normalized"] for e in drag]
    ys = [e["y_normalized"] for e in aim] + [e["y_normalized"] for e in drag]
    lo, hi = -LIMIT, 1 + LIMIT
    if max(xs) - min(xs) > hi - lo or max(ys) - min(ys) > hi - lo:
        return None
    dx = max(0.0, lo - min(xs)) - max(0.0, max(xs) - hi)
    dy = max(0.0, lo - min(ys)) - max(0.0, max(ys) - hi)
    move = lambda rows: [{**e, "x_normalized": e["x_normalized"] + dx,
                          "y_normalized": e["y_normalized"] + dy} for e in rows]
    return move(aim), move(drag)


def join_record(record: dict, family: str, seam: HumanSeam,
                rng: random.Random) -> dict | None:
    drag = [e for e in (record.get("events") or []) if _norm_xy(e) is not None]
    if len(drag) < 2:
        return None
    width = float(record.get("captcha", {}).get("width") or 500.0)
    height = float(record.get("captcha", {}).get("height") or 375.0)

    # 네 계열 모두 같은 경로다: 조준을 사람 자리에서 만들고 → 드래그를 그 끝에
    # 붙이고 → **둘을 함께** 화면 안으로 민다.
    #
    # 평행이동은 모양을 바꾸지 않는다 — 변위·직진도·overshoot 은 전부 위치와 무관하다.
    # 그래서 옮기는 것은 공짜이고, 버리는 것만이 편향이다. 앞선 판들은 전부 어딘가에서
    # 버렸고 그 자리에서 샜다: 각도를 무작위로 뽑아 버리니 직진도가(AUC 0.69),
    # 드래그만 옮기고 버리니 변위가(0.72) 갈렸다.
    for _ in range(24):
        made = (replay_aim(seam, rng) if family == "D_replay"
                else build_aim(family, seam, rng))
        if made is None:
            continue
        aim, row = made
        if len(aim) < 4:
            continue
        # 집는 지점은 조준 끝에서 사람 자신의 이음매 벡터만큼 떨어진 곳이다.
        # 정확히 같은 자리에 두면 간격 0 이 되어 그 자체가 표식이 된다(사람 0.0019).
        grab = (aim[-1]["x_normalized"] + row["seam_vec"][0],
                aim[-1]["y_normalized"] + row["seam_vec"][1])
        moved = _shift_drag(drag, grab, aim[-1]["t_ms"] + row["gap_ms"],
                            width, height, bounded=False)
        packed = _fit_into_stage(aim, moved)
        if packed is not None:
            aim, drag = packed
            break
    else:
        return None

    events = [{**e, "x": e["x_normalized"] * width, "y": e["y_normalized"] * height}
              for e in aim]
    events.extend(dict(e) for e in drag)
    for i, e in enumerate(events):
        e["seq"] = i
    out = dict(record)
    out["events"] = events
    out["aim_family"] = family
    out["aim_point_count"] = len(aim)
    # 조준을 빌려준 사람. 학습에서 폴드를 나눌 때 이 사람과 같은 쪽에 두어야
    # 같은 조준이 사람·봇 양쪽에 걸치지 않는다.
    out["aim_person"] = row.get("person")
    return out


# ---------------------------------------------------------------- 검증

def _auc(pos: list[float], neg: list[float]) -> float:
    """순위 기반 AUC. 0.5 면 그 값만으로는 두 무리를 못 가른다."""
    if len(pos) < 5 or len(neg) < 5:
        return float("nan")
    values = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    values.sort(key=lambda kv: kv[0])
    ranks, i = {}, 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1][0] == values[i][0]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = shared
        i = j + 1
    rank_sum = sum(ranks[k] for k, (_, lab) in enumerate(values) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def verify(joined: list[dict], seam: HumanSeam) -> int:
    """이음매만 보고 봇과 사람을 가를 수 있으면 붙이는 방식이 틀린 것이다."""
    bot = {"이음매 공간간격": [], "이음매 시간간격": [], "조준 점수": [], "조준 변위": []}
    for row in joined:
        aim = [e for e in row["events"] if e["event_type"] == "aimmove"]
        drag = [e for e in row["events"] if e["event_type"] != "aimmove"]
        if len(aim) < 2 or not drag:
            continue
        a0, a1, d0 = aim[0], aim[-1], drag[0]
        g = lambda e: {"x": e["x_normalized"], "y": e["y_normalized"]}
        bot["이음매 공간간격"].append(_dist(g(a1), g(d0)))
        bot["이음매 시간간격"].append(float(d0.get("t_ms") or 0) - float(a1["t_ms"]))
        bot["조준 점수"].append(float(len(aim)))
        bot["조준 변위"].append(_dist(g(a0), g(a1)))

    human = {"이음매 공간간격": seam.gap_xy, "이음매 시간간격": seam.gap_ms,
             "조준 점수": [float(p) for p in seam.points], "조준 변위": seam.displacement}

    print(f"\n  이음매 검증 — 봇 {len(bot['조준 점수'])}건 vs 사람 {len(seam.points)}건")
    print(f"  {'값':>16}{'봇 중앙':>10}{'사람 중앙':>11}{'AUC':>8}   판정")
    worst = 0.0
    for key in human:
        a = _auc(bot[key], human[key])
        worst = max(worst, abs(a - 0.5))
        flag = "OK" if abs(a - 0.5) <= 0.10 else ("주의" if abs(a - 0.5) <= 0.20 else "샌다")
        print(f"  {key:>16}{np.median(bot[key]):>10.4f}{np.median(human[key]):>11.4f}"
              f"{a:>8.3f}   {flag}")
    print(f"\n  최악 편차 {worst:.3f} — 0.10 이하라야 이음매가 표식이 되지 않는다.")
    return 0 if worst <= 0.10 else 1


# ---------------------------------------------------------------- 실행

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bots", required=True, type=Path)
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--human", type=Path, default=HUMAN_JOINED)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    seam = HumanSeam(args.human)
    # 생성기의 간격 표본은 사람 조준에서 나온다. 여기서 채워두지 않으면 봇이
    # 50ms 등간격으로 찍혀 `interval_cv` 하나에 AUC 0.966 으로 갈린다.
    mab._POOL = mab.human_intervals(seam.aims)
    rng = random.Random(args.seed)

    joined, skipped = [], 0
    with args.bots.open() as fh:
        lines = itertools.islice(fh, args.limit) if args.limit else fh
        for line in lines:
            row = join_record(json.loads(line), args.family, seam, rng)
            if row is None:
                skipped += 1
            else:
                joined.append(row)

    print(f"  {args.bots.name} · {args.family}")
    print(f"  이은 것 {len(joined)}건 · 건너뜀 {skipped}건")
    if joined:
        pts = [r["aim_point_count"] for r in joined]
        print(f"  조준 점수 중앙 {int(np.median(pts))} (사람 {int(np.median(seam.points))})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            for row in joined:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  -> {args.out}")

    return verify(joined, seam) if args.verify else 0


if __name__ == "__main__":
    raise SystemExit(main())
