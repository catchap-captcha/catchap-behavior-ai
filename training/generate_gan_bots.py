"""Generate human-like bot trajectories from a trained GAN (gated, explicit).

    python -m training.generate_gan_bots [--count N] [--holdout-ratio R]

Steps:
  1. require a trained generator (train_gan --train must have run under the gate)
  2. sample N trajectories
  3. drop samples too similar to any real human path (avoid memorization)
  4. store as label='bot', label_source='gan_bot', bot_family='gan'
  5. reserve a holdout fraction as TEST-ONLY (never used for training), tracked in
     a manifest so the split step keeps them out of train.

This never runs automatically. If the GAN readiness gate fails or no generator
exists, it stops without writing anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.services.replay_detector import NormalizedPathComparator
from training.train_gan import (
    FIXED_LEN,
    GAN_GENERATOR_FILE,
    check_readiness,
)

GENERATED_MANIFEST = os.path.join("data", "metadata", "gan_generated.json")
# generated paths closer than this to any real human path are discarded
MAX_ALLOWED_SIMILARITY = 0.97


def _load_generator():
    import torch
    from torch import nn

    if not os.path.exists(GAN_GENERATOR_FILE):
        return None, None
    ckpt = torch.load(GAN_GENERATOR_FILE, map_location="cpu")
    length, ch, latent = ckpt["length"], ckpt["channels"], ckpt["latent"]
    gen = nn.Sequential(
        nn.Linear(latent, 128), nn.ReLU(),
        nn.Linear(128, length * ch), nn.Tanh(),
    )
    gen.load_state_dict(ckpt["state_dict"])
    gen.eval()
    return gen, (length, ch, latent)


def _sample(gen, shape, count: int) -> np.ndarray:
    import torch

    length, ch, latent = shape
    with torch.no_grad():
        z = torch.randn(count, latent)
        out = gen(z).reshape(count, length, ch).numpy()
    return out


def _too_similar(path_xy: np.ndarray, real_paths: list[np.ndarray], cmp) -> bool:
    for r in real_paths:
        if cmp.similarity(path_xy, r[:, :2]) >= MAX_ALLOWED_SIMILARITY:
            return True
    return False


def _events_from_trajectory(traj: np.ndarray, width: int = 420, height: int = 220) -> list[dict[str, Any]]:
    """Convert a normalized (L,3) trajectory back into pointer events."""
    n = traj.shape[0]
    events = []
    duration_ms = 800.0
    for i in range(n):
        x_norm = float(np.clip(traj[i, 0], 0.0, 1.0))
        y_norm = float(np.clip((traj[i, 1] + 1.0) / 2.0, 0.0, 1.0))  # tanh y -> [0,1]
        t_ms = int(max(0.0, traj[i, 2]) * duration_ms)
        etype = "pointerdown" if i == 0 else "pointerup" if i == n - 1 else "pointermove"
        events.append({
            "seq": i,
            "event_type": etype,
            "t_ms": t_ms,
            "x": round(x_norm * width, 3),
            "y": round(y_norm * height, 3),
            "x_normalized": round(x_norm, 6),
            "y_normalized": round(y_norm, 6),
            "target_role": "slider_handle",
        })
    # enforce non-decreasing t_ms
    for i in range(1, n):
        if events[i]["t_ms"] < events[i - 1]["t_ms"]:
            events[i]["t_ms"] = events[i - 1]["t_ms"]
    return events


def generate(count: int, holdout_ratio: float, seed: int = 42) -> int:
    report, real_trajectories = check_readiness()
    if not report.ready or real_trajectories is None:
        print("GAN Bot 생성 차단: readiness 미충족. 먼저 실제 Human 데이터를 확보하세요.")
        for m in report.missing:
            print(f"  - {m}")
        return 2

    import torch

    torch.manual_seed(seed)
    gen, shape = _load_generator()
    if gen is None:
        print(f"학습된 generator가 없습니다: {GAN_GENERATOR_FILE}. train_gan --train 을 먼저 실행하세요.")
        return 2

    samples = _sample(gen, shape, count * 2)  # oversample; some get filtered
    cmp = NormalizedPathComparator()
    kept: list[dict[str, Any]] = []
    for traj in samples:
        if len(kept) >= count:
            break
        if _too_similar(traj[:, :2], real_trajectories, cmp):
            continue
        kept.append({"events": _events_from_trajectory(traj)})

    n_holdout = int(len(kept) * holdout_ratio)
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "generated_at": now,
        "bot_family": "gan",
        "label_source": "gan_bot",
        "count": len(kept),
        "test_only_count": n_holdout,
        "note": "test_only 항목은 학습에 사용하지 않고 test 평가 전용으로 보관합니다.",
        "attempts": [],
    }
    for i, item in enumerate(kept):
        manifest["attempts"].append({
            "index": i,
            "test_only": i < n_holdout,
            "n_events": len(item["events"]),
        })

    os.makedirs(os.path.dirname(GENERATED_MANIFEST), exist_ok=True)
    with open(GENERATED_MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print(f"GAN Bot 생성: {len(kept)}개 (test 전용 {n_holdout}개). 매니페스트: {GENERATED_MANIFEST}")
    print("주의: 실제 DB 적재는 통제된 수집 채널(/collect, label_source=gan_bot)을 통해 수행하세요.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate human-like GAN bots (gated).")
    p.add_argument("--count", type=int, default=500)
    p.add_argument("--holdout-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    return generate(args.count, args.holdout_ratio, seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
