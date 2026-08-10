"""Turn a participant list into the exact links each person opens.

The codes are not decoration. `person_of` splits on the first hyphen, so the
first segment IS the unit every promotion criterion is measured on — worst
participant FRR, the sealed/training split, "5 verification participants". A
code shaped wrong silently breaks the split: on 08-03 `sw-mouse` and
`sw-mouse-v2` were treated as two people and the numbers improved for no reason.

    {person}-{device}-{surface}      --with-device
    {person}-{surface}               default

    person    the unit. must be identical across every code that person opens
    device    mouse | trackpad, only with --with-device
    surface   captcha | player | bank. which screen the trajectory came from

Device is off by default (2026-08-06 decision). It cannot be recovered later:
the Pointer Events spec reports a trackpad as pointerType "mouse", so the
browser cannot tell them apart — `ai_behavior_attempts.device_type` is NULL on
all 619 rows and `ai_pointer_events.pointer_type` is "mouse" or NULL. Self-
reporting through the code was the only channel, and it doubled the link count
for people who mostly own one pointing device. The cost is real: trackpad
trajectories differ from mouse ones, so `MIN_SESSIONS_PER_PERSON_PER_DEVICE`
stops being checkable and the target collapses to 80 sessions per person. Anyone
who does use both can still open two codes with --with-device.

Surface lives in the code rather than a new column because `ai_behavior_attempts`
already stores `participant_id` and nothing else needed changing — and because a
schema migration on production to hold a label we can encode in a string is not
a trade worth making six days before M5.

    .venv/bin/python tools/collection_links.py sw jy ms p4 p5 p6 p7 p8
    .venv/bin/python tools/collection_links.py --format markdown sw jy ms
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.collection_split import (  # noqa: E402
    MIN_PEOPLE,
    MIN_SESSIONS_PER_PERSON_PER_DEVICE,
    person_of,
    rank,
)

CAPTCHA_HOST = "https://captcha.catchap5.com"
PLATFORM = "https://www.catchap5.com"

DEVICES = ("mouse", "trackpad")
# (surface tag, human label, url template). `captcha` needs no platform change —
# it is the direct link that has been collecting since 07-31.
SURFACES = (
    ("captcha", "민서 캡차 (직링크)", CAPTCHA_HOST + "/?participant={code}"),
    ("player", "강의 시청 화면", PLATFORM + "/student/lecture?collect={code}"),
    ("bank", "문제은행", PLATFORM + "/student/game?collect={code}"),
)

# The platform strips anything outside this set and truncates at 64 (하지영,
# 2026-08-06). Generating a code it would rewrite means the code stored next to
# the trajectory is not the code we handed out.
ALLOWED = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def codes_for(person: str, devices: tuple[str, ...]) -> list[tuple[str, str, str]]:
    out = []
    for device in devices or ("",):
        for tag, label, template in SURFACES:
            code = "-".join(p for p in (person, device, tag) if p)
            out.append((code, f"{device} · {label}" if device else label,
                        template.format(code=code)))
    return out


def validate(person: str, devices: tuple[str, ...]) -> str | None:
    if "-" in person:
        return "사람 식별자에 하이픈이 들어가면 person_of 가 잘라먹는다"
    for code, _, _ in codes_for(person, devices):
        if not ALLOWED.match(code):
            return f"플랫폼이 다시 쓸 코드: {code}"
        if person_of(code) != person:
            return f"person_of('{code}') = '{person_of(code)}' — 사람이 안 맞는다"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("people", nargs="+")
    ap.add_argument("--format", choices=("text", "markdown"), default="text")
    ap.add_argument("--with-device", action="store_true",
                    help="마우스/트랙패드를 코드로 나눈다. 둘 다 쓰는 사람에게만.")
    args = ap.parse_args()

    devices = DEVICES if args.with_device else ()
    people = list(dict.fromkeys(args.people))
    problems = {p: validate(p, devices) for p in people}
    problems = {p: why for p, why in problems.items() if why}
    if problems:
        for person, why in problems.items():
            print(f"거부 {person}: {why}")
        raise SystemExit(1)

    per_person = max(len(devices), 1) * len(SURFACES)
    target = MIN_SESSIONS_PER_PERSON_PER_DEVICE * len(DEVICES)

    if args.format == "markdown":
        print("| 사람 | 코드 | 어디서 | 링크 |")
        print("|---|---|---|---|")
        for person in people:
            for code, where, url in codes_for(person, devices):
                print(f"| {person} | `{code}` | {where} | {url} |")
    else:
        for person in people:
            print(f"\n■ {person}")
            for code, where, url in codes_for(person, devices):
                print(f"  {where}")
                print(f"    {url}")

    print(f"\n사람 {len(people)}명 · 1인당 링크 {per_person}개 · 총 {len(people)*per_person}개")
    if len(people) < MIN_PEOPLE:
        print(f"⚠️  {MIN_PEOPLE}명이 최소치다. 지금 {len(people)}명.")
    if devices:
        print(f"목표  1인당 {target}세션 이상 (장치별 {MIN_SESSIONS_PER_PERSON_PER_DEVICE})"
              f" · 전체 {len(people)*target}세션")
    else:
        print(f"목표  1인당 {target}세션 이상 · 전체 {len(people)*target}세션")
        print("  장치 축 없음 — 트랙패드가 pointerType 'mouse' 로 보고돼 나중에 복구 불가.")

    ordered = sorted({person_of(p) for p in people}, key=rank)
    print("\n봉인 순서 (collection_split 의 소금값으로 미리 고정된 것):")
    print("  " + " < ".join(ordered))
    print("  실제 봉인 인원은 수집이 끝난 뒤 collection_split.py 로 확정한다.")
    print("  ⚠️  이 순서를 보고 사람을 바꾸면 사전 고정의 의미가 사라진다.")


if __name__ == "__main__":
    main()
