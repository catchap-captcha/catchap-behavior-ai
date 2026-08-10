"""A red team that learns from its own hits, instead of sampling blindly.

`redteam_weakness_search.py` draws motion policies at random and keeps whatever
evades. That makes it cheap but wildly seed-dependent: on the same model with the
same budget, six seeds produced 0, 1, 17, 18, 21 and 28 evaders. A defence that
looks solved under one seed and broken under another has not been measured.

A real attacker does not resample from scratch. They keep what worked and search
around it. So this runs an evolutionary loop:

    sample policies -> score -> keep the best -> mutate around them -> repeat

Two things follow. The estimate stops depending on luck, because generations
converge on the same regions from any start. And the number it reports is an
upper bound on the defence, not a lower one — which is the number worth knowing.

Nothing here is fitted and no threshold is tuned. The detector is fixed, read
only, exactly as in the sweep tool.

    .venv/bin/python tools/redteam_evolution_search.py \
        --model models/candidate/.../two_view_fusion.joblib --generations 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.feature_extractor_v23 import extract_features  # noqa: E402
from training.generate_hybrid_redteam_bots import (MotionPolicy, POLICIES,  # noqa: E402
                                                   _motion_events)

HUMAN_ATTEMPTS = Path("data/raw/human_db_snapshot_20260721T012239Z/human_attempts.jsonl")

# The knobs the sweep varies, with the widest range it ever explores. An attacker
# is not bound by the calibration profile, so neither is this.
BOUNDS = {
    "curvature": (0.002, 0.120),
    "jitter": (0.0002, 0.0120),
    "time_power": (0.50, 1.80),
    "duration_scale": (0.50, 1.80),
    "turn_slowdown": (0.02, 0.60),
    "coalesce_fraction": (0.01, 0.30),
}
FRAME_CHOICES = (8, 10, 12, 16, 20)


def random_policy(rng: random.Random) -> MotionPolicy:
    def span(key):
        lo, hi = BOUNDS[key]
        a, b = sorted((rng.uniform(lo, hi), rng.uniform(lo, hi)))
        return (a, b)
    return MotionPolicy(
        curvature=span("curvature"), jitter=span("jitter"),
        time_power=span("time_power"), duration_scale=span("duration_scale"),
        turn_slowdown=span("turn_slowdown"), coalesce_fraction=span("coalesce_fraction"),
        frame_ms=tuple(rng.sample(FRAME_CHOICES, k=rng.randint(1, 3))),
        late_correction_probability=rng.uniform(0.0, 1.0),
    )


def mutate(policy: MotionPolicy, rng: random.Random, scale: float) -> MotionPolicy:
    """Small moves around a hit. Large enough to explore, small enough to keep it."""
    def jog(pair, key):
        lo, hi = BOUNDS[key]
        width = (hi - lo) * scale
        a, b = (min(max(v + rng.gauss(0, width), lo), hi) for v in pair)
        return tuple(sorted((a, b)))
    return replace(
        policy,
        curvature=jog(policy.curvature, "curvature"),
        jitter=jog(policy.jitter, "jitter"),
        time_power=jog(policy.time_power, "time_power"),
        duration_scale=jog(policy.duration_scale, "duration_scale"),
        turn_slowdown=jog(policy.turn_slowdown, "turn_slowdown"),
        coalesce_fraction=jog(policy.coalesce_fraction, "coalesce_fraction"),
        frame_ms=(tuple(rng.sample(FRAME_CHOICES, k=rng.randint(1, 3)))
                  if rng.random() < 0.3 else policy.frame_ms),
        late_correction_probability=min(1.0, max(0.0,
            policy.late_correction_probability + rng.gauss(0, 0.15))),
    )


MAIN_CAPTCHA = Path("data/interim/collection_20260806.jsonl")
SEALED_PEOPLE = {"sw", "ms"}


def load_main_captcha_bases(limit: int, only: set[str] | None = None) -> list[dict]:
    """Substrate shaped like the surface we deploy on.

    Warping legacy traces produces ~107-event drags against a surface whose real
    drags are ~12, and `event_count` alone then separates them with AUC 1.000.
    That flatters the defence for a reason no attacker would concede. Sealed
    people are excluded — warping their traces would leak them into anything
    fitted against this red team.
    """
    out = []
    with MAIN_CAPTCHA.open() as f:
        for line in f:
            record = json.loads(line)
            code = record.get("participant_id") or ""
            person = code.split("-")[0]
            if person in SEALED_PEOPLE:
                continue
            if only and person not in only:
                continue
            if record.get("quality_status") != "valid":
                continue
            rows = record.get("events") or []
            events = [{
                "seq": r.get("seq"), "event_type": r.get("event_type"),
                "t_ms": float(r.get("client_timestamp_ms") or 0),
                "x": float(r["x_pixel"]), "y": float(r["y_pixel"]),
                "x_normalized": r.get("x_normalized"), "y_normalized": r.get("y_normalized"),
            } for r in rows if r.get("x_pixel") is not None]
            if len(events) >= 8:
                out.append({"events": events,
                            "captcha": {"width": record.get("stage_width") or 500,
                                        "height": record.get("stage_height") or 375}})
            if len(out) >= limit:
                break
    return out


def load_bases(limit: int) -> list[dict]:
    """Real human traces are the substrate the sweep warps; keep that identical."""
    out = []
    with HUMAN_ATTEMPTS.open() as f:
        for line in f:
            rec = json.loads(line)
            events = rec.get("events") or []
            if len(events) >= 8:
                out.append(rec)
            if len(out) >= limit:
                break
    return out


def scorer(bundle: dict):
    views, models = bundle["feature_views"], bundle["models"]

    def score(feature_rows: list[dict]) -> np.ndarray:
        per = []
        for name, model in models.items():
            X = np.array([[float(r.get(n) or 0.0) for n in views[name]] for r in feature_rows])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            per.append(model.predict_proba(X)[:, list(model.classes_).index(1)])
        return np.min(np.vstack(per), axis=0)

    return score


def evaluate(policy: MotionPolicy, bases: list[dict], rng: random.Random,
             score, per_policy: int, normalized: bool = False) -> np.ndarray:
    rows = []
    for base in rng.sample(bases, k=min(per_policy, len(bases))):
        captcha = base.get("captcha") or {}
        width = int(captcha.get("width") or 500)
        height = int(captcha.get("height") or 320)
        # Returns (events, metadata) — counting the tuple instead of the events
        # silently produced "2" for every policy and filtered the whole run away.
        events, _meta = _motion_events(base["events"], width=width, height=height,
                                       policy=policy, randomizer=rng)
        if normalized:
            # Score a scale-invariant model on the coordinates it was trained on;
            # feeding it pixels would compare two different feature definitions.
            events = [dict(e, x=e.get("x_normalized"), y=e.get("y_normalized"))
                      for e in events if e.get("x_normalized") is not None]
        if len(events) >= 3:
            rows.append(extract_features(events, None))
    return score(rows) if rows else np.array([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--population", type=int, default=40)
    ap.add_argument("--per-policy", type=int, default=12)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--dump-elites", type=Path,
                    help="마지막 세대의 엘리트 정책을 저장한다 (합산 판정 분석용)")
    ap.add_argument("--only-person", nargs="*", default=[],
                    help="이 사람 궤적만 공격 기반으로 쓴다")
    ap.add_argument("--surface", choices=("legacy","main"), default="legacy",
                    help="공격 기반 궤적. main = 실제 배포 표면")
    ap.add_argument("--normalized", action="store_true",
                    help="model was trained on 0..1 coordinates rather than pixels")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    threshold = float(bundle["threshold"])
    score = scorer(bundle)
    rng = random.Random(args.seed)
    bases = (load_main_captcha_bases(300, set(args.only_person)) if args.surface == "main"
             else load_bases(400))

    population = [random_policy(rng) for _ in range(args.population)]
    print(f"모델 {Path(args.model).parent.name} · 임계 {threshold:.7f}")
    print(f"세대 {args.generations} · 개체 {args.population} · 정책당 {args.per_policy}건\n")
    print(f"  {'세대':>4s}{'평가':>8s}{'회피율':>9s}{'최고 점수':>12s}{'상위8 평균':>12s}")

    best_overall = 0.0
    for generation in range(1, args.generations + 1):
        scored = []
        evaded = total = 0
        for policy in population:
            values = evaluate(policy, bases, rng, score, args.per_policy, args.normalized)
            if values.size == 0:
                continue
            scored.append((float(values.max()), float(values.mean()), policy))
            evaded += int((values >= threshold).sum())
            total += values.size

        scored.sort(key=lambda t: -t[0])
        elite = [p for _, _, p in scored[: args.elite]]
        top_mean = float(np.mean([m for _, m, _ in scored[: args.elite]])) if scored else 0.0
        best_overall = max(best_overall, scored[0][0] if scored else 0.0)
        print(f"  {generation:>4d}{total:>8d}{evaded/max(total,1)*100:>8.1f}%"
              f"{scored[0][0]:>12.6f}{top_mean:>12.6f}")

        # Narrow the step as the search closes in, the usual annealing.
        scale = 0.25 * (1 - generation / (args.generations + 1))
        population = elite + [mutate(rng.choice(elite), rng, scale)
                              for _ in range(args.population - len(elite))]

    print(f"\n  최고 점수 {best_overall:.6f} · 임계 {threshold:.7f} "
          f"→ {'회피 성공' if best_overall >= threshold else '회피 실패'}")

    if args.dump_elites:
        # The converged policies ARE the attacker. Re-deriving them with a short
        # search gives a much weaker adversary — a 2-generation best policy sat
        # below the threshold while the 30-generation run evaded 73.7%.
        args.dump_elites.parent.mkdir(parents=True, exist_ok=True)
        args.dump_elites.write_text(json.dumps({
            "model": args.model,
            "threshold": threshold,
            "generations": args.generations,
            "seed": args.seed,
            "surface": args.surface,
            "elites": [{k: list(v) if isinstance(v, tuple) else v
                        for k, v in vars(p).items()} for p in elite],
        }, ensure_ascii=False, indent=1) + "\n")
        print(f"  엘리트 정책 {len(elite)}개 -> {args.dump_elites}")


if __name__ == "__main__":
    main()
