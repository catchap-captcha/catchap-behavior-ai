"""GAN training for human-like drag trajectories (LATE-STAGE, gated).

The GAN learns from REAL human drags only and generates human-like bot paths for
adversarial hardening. It never runs automatically and never runs unless there
is enough real human data:

    python -m training.train_gan --check-readiness   # gate report only
    python -m training.train_gan --train             # train (gated)

Eligible data (strict):
  * quality_status = 'valid'
  * label = 'human'
  * label_source = 'controlled_collection'
  * train split only
  * complete pointerdown .. pointerup raw event sequence

PyTorch is imported lazily inside --train so the rest of the service does not
depend on it. The readiness check needs only row counts (no torch).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from app.config import get_settings

GAN_MODEL_DIR = "models/candidate"
GAN_GENERATOR_FILE = os.path.join(GAN_MODEL_DIR, "gan_generator.pt")
GAN_READINESS_REPORT = os.path.join("reports", "gan_readiness.json")
FIXED_LEN = 64  # resample every trajectory to this many points


@dataclass
class GanThresholds:
    min_gan_human_samples: int
    min_gan_human_participants: int

    @classmethod
    def from_settings(cls) -> "GanThresholds":
        s = get_settings()
        return cls(s.min_gan_human_samples, s.min_gan_human_participants)


@dataclass
class GanReadiness:
    ready: bool
    reason: str
    human_samples: int
    required_human_samples: int
    human_participants: int
    required_human_participants: int
    missing: list[str] = field(default_factory=list)


def compute_gan_readiness(rows: list[dict[str, Any]], thr: GanThresholds) -> GanReadiness:
    """Assess GAN readiness from eligible human rows (already filtered)."""
    samples = len(rows)
    participants = len({r.get("anonymous_participant_id") for r in rows if r.get("anonymous_participant_id")})
    missing: list[str] = []
    if samples < thr.min_gan_human_samples:
        missing.append(f"GAN용 Human 원본 {thr.min_gan_human_samples - samples}개 부족")
    if participants < thr.min_gan_human_participants:
        missing.append(f"GAN용 Human 참여자 {thr.min_gan_human_participants - participants}명 부족")
    ready = not missing
    return GanReadiness(
        ready=ready,
        reason="ready" if ready else "gan_data_not_ready",
        human_samples=samples,
        required_human_samples=thr.min_gan_human_samples,
        human_participants=participants,
        required_human_participants=thr.min_gan_human_participants,
        missing=missing,
    )


# --------------------------------------------------------------------------- #
# trajectory preprocessing (pure, unit-testable, no torch)
# --------------------------------------------------------------------------- #
def normalize_trajectory(events: Sequence[dict[str, Any]], fixed_len: int = FIXED_LEN) -> np.ndarray:
    """Normalize one raw drag to a fixed-length (fixed_len, 3) array.

    Channels: (x, y, t) with start anchored to (0, 0), end to (1, 0) via an
    affine transform along the drag axis, and time linearly rescaled to [0, 1].
    Returns zeros for degenerate input.
    """
    pts = sorted(events, key=lambda e: e.get("seq", 0))
    xy, t = [], []
    for e in pts:
        x = e.get("x_normalized", e.get("x"))
        y = e.get("y_normalized", e.get("y"))
        tm = e.get("t_ms")
        if x is None or y is None or tm is None:
            continue
        xy.append((float(x), float(y)))
        t.append(float(tm))
    if len(xy) < 2:
        return np.zeros((fixed_len, 3), float)

    xy = np.asarray(xy, float)
    t = np.asarray(t, float)

    # rotate/scale so start->end lies on the x-axis from (0,0) to (1,0)
    start, end = xy[0], xy[-1]
    vec = end - start
    length = np.hypot(*vec)
    if length < 1e-9:
        norm_xy = xy - start
    else:
        angle = np.arctan2(vec[1], vec[0])
        c, s = np.cos(-angle), np.sin(-angle)
        rot = np.array([[c, -s], [s, c]])
        norm_xy = ((xy - start) @ rot.T) / length

    t_span = t[-1] - t[0]
    norm_t = (t - t[0]) / t_span if t_span > 1e-9 else np.linspace(0, 1, len(t))

    # resample to fixed length along cumulative index
    idx = np.linspace(0, len(norm_xy) - 1, fixed_len)
    rx = np.interp(idx, np.arange(len(norm_xy)), norm_xy[:, 0])
    ry = np.interp(idx, np.arange(len(norm_xy)), norm_xy[:, 1])
    rt = np.interp(idx, np.arange(len(norm_t)), norm_t)
    return np.stack([rx, ry, rt], axis=1)


# --------------------------------------------------------------------------- #
# DB access for eligible human trajectories
# --------------------------------------------------------------------------- #
def _fetch_eligible_human_sequences() -> list[dict[str, Any]] | None:
    """Return eligible human attempts with their raw events, restricted to the
    train split when a split manifest is available."""
    from sqlalchemy import text

    from app.database.connection import check_connection, get_sessionmaker

    if not check_connection():
        print("MySQL 연결 실패.", file=sys.stderr)
        return None

    train_ids = _train_split_ids()
    session = get_sessionmaker()()
    try:
        rows = session.execute(text(
            """
            SELECT a.attempt_id, a.anonymous_participant_id
            FROM ai_behavior_attempts a
            WHERE a.quality_status = 'valid'
              AND a.label = 'human'
              AND a.label_source = 'controlled_collection'
            """
        )).mappings().all()
        attempts = []
        for r in rows:
            aid = r["attempt_id"]
            if train_ids is not None and aid not in train_ids:
                continue
            ev = session.execute(text(
                "SELECT seq, event_type, t_ms, x, y, x_normalized, y_normalized "
                "FROM ai_pointer_events WHERE attempt_id = :aid ORDER BY seq"
            ), {"aid": aid}).mappings().all()
            types = {e["event_type"] for e in ev}
            if "pointerdown" in types and "pointerup" in types and len(ev) >= 2:
                attempts.append({
                    "attempt_id": aid,
                    "anonymous_participant_id": r["anonymous_participant_id"],
                    "events": [dict(e) for e in ev],
                })
        return attempts
    finally:
        session.close()


def _train_split_ids() -> set[str] | None:
    """Read the newest split manifest to restrict GAN input to the train split."""
    meta_dir = "data/metadata"
    if not os.path.isdir(meta_dir):
        return None
    manifests = sorted(f for f in os.listdir(meta_dir) if f.startswith("split_manifest_"))
    if not manifests:
        return None
    with open(os.path.join(meta_dir, manifests[-1]), encoding="utf-8") as fh:
        data = json.load(fh)
    return {aid for aid, split in data.get("attempt_to_split", {}).items() if split == "train"}


def _write_readiness(report: GanReadiness) -> None:
    os.makedirs(os.path.dirname(GAN_READINESS_REPORT), exist_ok=True)
    with open(GAN_READINESS_REPORT, "w", encoding="utf-8") as fh:
        json.dump(asdict(report), fh, ensure_ascii=False, indent=2)


def check_readiness() -> tuple[GanReadiness, list[np.ndarray] | None]:
    """Gate: fetch eligible human data and assess GAN readiness."""
    attempts = _fetch_eligible_human_sequences()
    thr = GanThresholds.from_settings()
    if attempts is None:
        report = GanReadiness(
            False, "db_unavailable", 0, thr.min_gan_human_samples,
            0, thr.min_gan_human_participants, ["MySQL 연결/뷰 확인 필요"],
        )
        _write_readiness(report)
        return report, None
    report = compute_gan_readiness(
        [{"anonymous_participant_id": a["anonymous_participant_id"]} for a in attempts], thr
    )
    _write_readiness(report)
    trajectories = [normalize_trajectory(a["events"]) for a in attempts] if report.ready else None
    return report, trajectories


def train(epochs: int = 300, seed: int = 42) -> int:
    """Train the GAN — only if the readiness gate passes. Torch imported here."""
    report, trajectories = check_readiness()
    if not report.ready or not trajectories:
        print("GAN 학습 차단: 실제 Human 데이터가 부족합니다 (gan_data_not_ready).")
        for m in report.missing:
            print(f"  - {m}")
        return 2

    import torch
    from torch import nn

    torch.manual_seed(seed)
    data = torch.tensor(np.stack(trajectories), dtype=torch.float32)  # (N, L, 3)
    n, length, ch = data.shape
    flat = data.reshape(n, length * ch)
    latent = 32

    gen = nn.Sequential(
        nn.Linear(latent, 128), nn.ReLU(),
        nn.Linear(128, length * ch), nn.Tanh(),
    )
    disc = nn.Sequential(
        nn.Linear(length * ch, 128), nn.LeakyReLU(0.2),
        nn.Linear(128, 1), nn.Sigmoid(),
    )
    opt_g = torch.optim.Adam(gen.parameters(), lr=2e-4)
    opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4)
    bce = nn.BCELoss()

    for epoch in range(epochs):
        # discriminator
        opt_d.zero_grad()
        real = flat
        z = torch.randn(n, latent)
        fake = gen(z).detach()
        loss_d = bce(disc(real), torch.ones(n, 1)) + bce(disc(fake), torch.zeros(n, 1))
        loss_d.backward(); opt_d.step()
        # generator
        opt_g.zero_grad()
        z = torch.randn(n, latent)
        loss_g = bce(disc(gen(z)), torch.ones(n, 1))
        loss_g.backward(); opt_g.step()

    os.makedirs(GAN_MODEL_DIR, exist_ok=True)
    torch.save({"state_dict": gen.state_dict(), "length": length, "channels": ch,
                "latent": latent}, GAN_GENERATOR_FILE)
    print(f"GAN generator 저장: {GAN_GENERATOR_FILE}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="GAN training (gated).")
    p.add_argument("--check-readiness", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if args.check_readiness:
        report, _ = check_readiness()
        print(f"GAN ready={report.ready} samples={report.human_samples}/{report.required_human_samples} "
              f"participants={report.human_participants}/{report.required_human_participants}")
        for m in report.missing:
            print(f"  - {m}")
        return 0 if report.ready else 2
    if args.train:
        return train(epochs=args.epochs, seed=args.seed)
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
