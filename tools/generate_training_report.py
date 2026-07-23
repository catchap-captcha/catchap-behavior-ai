"""Generate the Korean Catchap AI training and validation PDF report."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "reports/local_h4786_b3000_20260713/training_summary.json"
HUMAN_MANIFEST_PATH = ROOT / "data/processed/human_confirmed_4786_20260713/manifest.json"
DATASET_MANIFEST_PATH = ROOT / "data/processed/local_h4786_b3000_20260713/manifest.json"
MODEL_COMPARISON_PATH = ROOT / "reports/local_h4786_b3000_20260713/model_comparison.csv"
REPORT_DIR = ROOT / "reports/local_h4786_b3000_20260713"
OUTPUT_PATH = ROOT / "output/pdf/catchap_ai_model_training_report.pdf"

FONT_PATH = Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf")
FONT = "AppleGothic"

NAVY = colors.HexColor("#142536")
INK = colors.HexColor("#1E2932")
TEAL = colors.HexColor("#178C85")
CORAL = colors.HexColor("#D95D4F")
GOLD = colors.HexColor("#D6A438")
BLUE = colors.HexColor("#3D6D9A")
MINT = colors.HexColor("#E5F4F1")
PALE_BLUE = colors.HexColor("#EAF1F7")
PALE_CORAL = colors.HexColor("#FBECE9")
PALE_GOLD = colors.HexColor("#FAF3DF")
LIGHT = colors.HexColor("#F4F6F7")
MID = colors.HexColor("#D7DEE2")
GRAY = colors.HexColor("#66727B")
WHITE = colors.white

PAGE_W, PAGE_H = A4
CONTENT_W = PAGE_W - 34 * mm


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont(FONT, str(FONT_PATH)))
    pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=FONT, italic=FONT, boldItalic=FONT)


def build_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=sample["Title"],
            fontName=FONT,
            fontSize=29,
            leading=40,
            textColor=WHITE,
            alignment=TA_LEFT,
            spaceAfter=12,
            wordWrap="CJK",
        ),
        "cover_sub": ParagraphStyle(
            "CoverSub",
            fontName=FONT,
            fontSize=12,
            leading=19,
            textColor=colors.HexColor("#DDE8ED"),
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "H1",
            fontName=FONT,
            fontSize=20,
            leading=28,
            textColor=NAVY,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "H2",
            fontName=FONT,
            fontSize=13,
            leading=19,
            textColor=TEAL,
            spaceBefore=8,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "BodyKR",
            fontName=FONT,
            fontSize=9.4,
            leading=15,
            textColor=INK,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "body_small": ParagraphStyle(
            "BodySmallKR",
            fontName=FONT,
            fontSize=8.1,
            leading=12.2,
            textColor=INK,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "CaptionKR",
            fontName=FONT,
            fontSize=7.3,
            leading=10,
            textColor=GRAY,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "TableKR",
            fontName=FONT,
            fontSize=7.5,
            leading=10.5,
            textColor=INK,
            wordWrap="CJK",
        ),
        "table_head": ParagraphStyle(
            "TableHeadKR",
            fontName=FONT,
            fontSize=7.7,
            leading=10.5,
            textColor=WHITE,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "CodeKR",
            fontName=FONT,
            fontSize=7.4,
            leading=11,
            textColor=colors.HexColor("#24323D"),
            backColor=LIGHT,
            borderPadding=7,
            leftIndent=3,
            rightIndent=3,
            wordWrap="CJK",
        ),
        "callout": ParagraphStyle(
            "CalloutKR",
            fontName=FONT,
            fontSize=9.3,
            leading=15,
            textColor=NAVY,
            wordWrap="CJK",
        ),
        "quote": ParagraphStyle(
            "QuoteKR",
            fontName=FONT,
            fontSize=10,
            leading=17,
            leftIndent=12,
            rightIndent=8,
            textColor=NAVY,
            wordWrap="CJK",
        ),
    }


class HorizontalBars(Flowable):
    def __init__(self, items: list[tuple[str, float, str]], maximum: float, height: float = 54 * mm):
        super().__init__()
        self.items = items
        self.maximum = maximum
        self.width = CONTENT_W
        self.height = height

    def draw(self) -> None:
        canvas = self.canv
        left = 37 * mm
        right = 16 * mm
        bar_w = self.width - left - right
        row_h = self.height / max(len(self.items), 1)
        palette = {"teal": TEAL, "blue": BLUE, "coral": CORAL, "gold": GOLD}
        for index, (label, value, color_name) in enumerate(self.items):
            y = self.height - (index + 0.72) * row_h
            canvas.setFont(FONT, 8)
            canvas.setFillColor(INK)
            canvas.drawRightString(left - 3 * mm, y + 1.4 * mm, label)
            canvas.setFillColor(LIGHT)
            canvas.roundRect(left, y, bar_w, 6 * mm, 2 * mm, fill=1, stroke=0)
            fill_w = bar_w * min(value / self.maximum, 1.0)
            canvas.setFillColor(palette[color_name])
            canvas.roundRect(left, y, fill_w, 6 * mm, 2 * mm, fill=1, stroke=0)
            canvas.setFillColor(NAVY)
            canvas.setFont(FONT, 8)
            canvas.drawString(left + bar_w + 3 * mm, y + 1.4 * mm, f"{value:,.0f}")


class PipelineFlow(Flowable):
    def __init__(self):
        super().__init__()
        self.width = CONTENT_W
        self.height = 44 * mm

    def draw(self) -> None:
        canvas = self.canv
        labels = ["원시 궤적", "29 Feature", "그룹 분할", "3개 모델", "적대적 검증"]
        box_w = 28 * mm
        gap = (self.width - len(labels) * box_w) / (len(labels) - 1)
        y = 13 * mm
        for i, label in enumerate(labels):
            x = i * (box_w + gap)
            canvas.setFillColor(MINT if i < 2 else PALE_BLUE if i < 4 else PALE_CORAL)
            canvas.setStrokeColor(TEAL if i < 2 else BLUE if i < 4 else CORAL)
            canvas.roundRect(x, y, box_w, 16 * mm, 2 * mm, fill=1, stroke=1)
            canvas.setFillColor(NAVY)
            canvas.setFont(FONT, 8)
            canvas.drawCentredString(x + box_w / 2, y + 6.4 * mm, label)
            if i < len(labels) - 1:
                x1 = x + box_w + 1.5 * mm
                x2 = x + box_w + gap - 1.5 * mm
                mid_y = y + 8 * mm
                canvas.setStrokeColor(GRAY)
                canvas.setLineWidth(1)
                canvas.line(x1, mid_y, x2, mid_y)
                canvas.line(x2 - 2 * mm, mid_y + 1.5 * mm, x2, mid_y)
                canvas.line(x2 - 2 * mm, mid_y - 1.5 * mm, x2, mid_y)


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def bullets(items: list[str], styles: dict[str, ParagraphStyle]) -> list[Paragraph]:
    return [p(f"- {item}", styles["body"]) for item in items]


def section(number: str, title: str, subtitle: str, styles: dict[str, ParagraphStyle]) -> list[Any]:
    return [
        p(f"{number}. {title}", styles["h1"]),
        p(subtitle, styles["body_small"]),
        Spacer(1, 4 * mm),
    ]


def make_table(
    rows: list[list[Any]],
    widths: list[float],
    styles: dict[str, ParagraphStyle],
    header: bool = True,
    aligns: dict[int, str] | None = None,
    cell_backgrounds: dict[tuple[int, int], colors.Color] | None = None,
) -> Table:
    rendered = []
    for row_index, row in enumerate(rows):
        rendered.append(
            [
                value
                if isinstance(value, (Paragraph, Flowable))
                else p(str(value), styles["table_head"] if header and row_index == 0 else styles["table"])
                for value in row
            ]
        )
    table = Table(rendered, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY if header else WHITE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE if header else INK),
        ("GRID", (0, 0), (-1, -1), 0.35, MID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header and len(rows) > 1:
        for row_index in range(1, len(rows)):
            if row_index % 2 == 0:
                commands.append(("BACKGROUND", (0, row_index), (-1, row_index), LIGHT))
    for column, alignment in (aligns or {}).items():
        commands.append(("ALIGN", (column, 1 if header else 0), (column, -1), alignment))
    for (row_index, column), background in (cell_backgrounds or {}).items():
        commands.append(("BACKGROUND", (column, row_index), (column, row_index), background))
    table.setStyle(TableStyle(commands))
    return table


def callout(text: str, styles: dict[str, ParagraphStyle], tone: str = "teal") -> Table:
    palette = {
        "teal": (MINT, TEAL),
        "coral": (PALE_CORAL, CORAL),
        "gold": (PALE_GOLD, GOLD),
        "blue": (PALE_BLUE, BLUE),
    }
    background, border = palette[tone]
    table = Table([[p(text, styles["callout"])]], colWidths=[CONTENT_W])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.8, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return table


def metric_cards(cards: list[tuple[str, str, str]], styles: dict[str, ParagraphStyle]) -> Table:
    data = []
    for label, value, note in cards:
        data.append(
            p(
                f'<font color="#66727B" size="7">{label}</font><br/>'
                f'<font color="#142536" size="16">{value}</font><br/>'
                f'<font color="#66727B" size="7">{note}</font>',
                styles["table"],
            )
        )
    table = Table([data], colWidths=[CONTENT_W / len(cards)] * len(cards))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), WHITE),
                ("BOX", (0, 0), (-1, -1), 0.6, MID),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, MID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return table


def heat_color(value: float) -> colors.Color:
    if value >= 0.8:
        return MINT
    if value >= 0.5:
        return PALE_GOLD
    return PALE_CORAL


def first_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, PAGE_H - 10 * mm, PAGE_W, 10 * mm, fill=1, stroke=0)
    canvas.setFillColor(CORAL)
    canvas.rect(0, 0, PAGE_W, 5 * mm, fill=1, stroke=0)
    canvas.restoreState()


def later_pages(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(MID)
    canvas.line(17 * mm, PAGE_H - 14 * mm, PAGE_W - 17 * mm, PAGE_H - 14 * mm)
    canvas.setFont(FONT, 7.2)
    canvas.setFillColor(GRAY)
    canvas.drawString(17 * mm, PAGE_H - 10.5 * mm, "Catchap AI - 모델 학습 및 검증 보고서")
    canvas.drawRightString(PAGE_W - 17 * mm, 10 * mm, f"{doc.page}")
    canvas.drawString(17 * mm, 10 * mm, "2026-07-13 | candidate only | production 미승격")
    canvas.restoreState()


def build_report() -> Path:
    register_fonts()
    styles = build_styles()
    summary = load_json(SUMMARY_PATH)
    human_manifest = load_json(HUMAN_MANIFEST_PATH)
    dataset_manifest = load_json(DATASET_MANIFEST_PATH)
    comparison = {row["model_name"]: row for row in load_csv(MODEL_COMPARISON_PATH)}
    holdout = summary["family_holdout_stress_test"]
    robust = summary["selection"]["robust_candidate"]
    human_counts = human_manifest["counts"]
    split = summary["split"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=19 * mm,
        bottomMargin=17 * mm,
        title="Catchap AI 모델 학습 및 검증 보고서",
        author="Catchap AI Team",
        subject="Human/Bot 행동 모델 데이터, 학습, 검증 및 배포 준비 상태",
    )

    story: list[Any] = []

    # Cover
    story.extend(
        [
            Spacer(1, 38 * mm),
            p("Catchap AI", styles["cover_sub"]),
            Spacer(1, 3 * mm),
            p("모델 학습 및<br/>검증 보고서", styles["cover_title"]),
            Spacer(1, 6 * mm),
            p(
                "Human 행동 데이터 약 5천 건 체크포인트<br/>"
                "데이터 구조, 전처리, 세 모델 비교, 적대적 검증, 배포 기준",
                styles["cover_sub"],
            ),
            Spacer(1, 24 * mm),
            Table(
                [
                    [p("Human", styles["cover_sub"]), p("Bot", styles["cover_sub"]), p("현재 판정", styles["cover_sub"])],
                    [
                        p("4,786", styles["cover_title"]),
                        p("3,000", styles["cover_title"]),
                        p("Candidate", styles["cover_title"]),
                    ],
                    [
                        p("학습 사용", styles["cover_sub"]),
                        p("3 family", styles["cover_sub"]),
                        p("Production 미승격", styles["cover_sub"]),
                    ],
                ],
                colWidths=[CONTENT_W / 3] * 3,
                style=TableStyle(
                    [
                        ("TEXTCOLOR", (0, 0), (-1, -1), WHITE),
                        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#496273")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#496273")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 9),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                ),
            ),
            Spacer(1, 24 * mm),
            p("작성일 2026-07-13 | 데이터 및 결과 자동 반영", styles["cover_sub"]),
            PageBreak(),
        ]
    )

    # 1 Executive summary
    story.extend(section("1", "핵심 요약", "의사결정자가 먼저 알아야 할 결론입니다.", styles))
    story.append(
        metric_cards(
            [
                ("Human 학습 데이터", f"{human_counts['included_human_rows']:,}", "원본 4,816 중 30 제외"),
                ("Bot 학습 데이터", f"{dataset_manifest['bot_rows']:,}", "straight/accel/jitter"),
                ("Robust 후보", "LightGBM", "상대적 1순위"),
                ("배포 가능", "아니요", "unseen gate 실패"),
            ],
            styles,
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(
        callout(
            "<b>현재 결론:</b> 알려진 규칙형 Bot에는 높은 성능을 보였지만, 학습에서 완전히 제외한 "
            "Bot family의 최악 탐지율이 31.4%였습니다. LightGBM은 최종 모델이 아니라 다음 데이터 "
            "수집과 재학습을 위한 candidate입니다.",
            styles,
            "coral",
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.extend(
        bullets(
            [
                "Human FRR 목표는 3% 이하이며 LightGBM 일반 test FRR은 2.16%로 통과했습니다.",
                "일반 Bot Recall은 세 모델 모두 100%였으나 같은 Bot 분포 안에서의 성능입니다.",
                "unseen-family 최악 Bot Recall 목표 80% 대비 현재 최고가 31.4%라 production 승격을 막았습니다.",
                "현재 가장 큰 부족분은 시도 건수가 아니라 연결 Human 참여자 25명과 Bot family 3종의 다양성입니다.",
            ],
            styles,
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(
        make_table(
            [
                ["항목", "현재 상태", "판정"],
                ["데이터 무결성", "Feature 29개, NULL/NaN/Infinity 0", "PASS"],
                ["사람 오탐", "LightGBM FRR 2.16%", "PASS"],
                ["처음 보는 Bot", "최악 Recall 31.4%", "FAIL"],
                ["Production", "모델 파일 없음", "보류"],
            ],
            [48 * mm, 90 * mm, 32 * mm],
            styles,
            aligns={2: "CENTER"},
            cell_backgrounds={(1, 2): MINT, (2, 2): MINT, (3, 2): PALE_CORAL, (4, 2): PALE_GOLD},
        )
    )
    story.append(PageBreak())

    # 2 Work completed
    story.extend(section("2", "지금까지 진행한 작업", "서비스 연결부터 로컬 학습 후보 생성까지의 이력입니다.", styles))
    timeline = [
        ["단계", "완료 내용", "산출물/상태"],
        ["1. 코드 기준선", "frontend/backend/captcha/AI 저장소의 sw 작업 기준 정리", "sw branch"],
        ["2. 메인 CAPTCHA", "Forest CAPTCHA UI와 backend challenge API 연결, 반응형 팝업 적용", "로컬 동작 검증"],
        ["3. DB 감사", "행동 요약과 원시 궤적의 라벨·건수·중복·자동화 흔적 확인", "기계적 패턴 0, 동일 궤적 0"],
        ["4. Bot baseline", "straight/accel/jitter 규칙형 Bot 각 1,000건 생성", "Bot 3,000 JSONL"],
        ["5. Human 스냅샷", "원격 DB를 read-only로 고정 시점 복사하고 ID 익명화", "원본 4,816"],
        ["6. 라벨·전처리", "통제 수집 Human 라벨, 궤적 누락 30건 제외, 29 Feature 계산", "Human 4,786"],
        ["7. 로컬 학습", "RF/XGBoost/LightGBM 학습과 일반 test 수행", "candidate 3개"],
        ["8. 적대적 검증", "Bot family 하나를 통째로 숨기는 3-fold holdout", "배포 gate 실패"],
        ["9. 실험 기록", "약 5천 체크포인트와 향후 1만/1.5만 비교 양식 작성", "history MD/CSV"],
    ]
    story.append(make_table(timeline, [31 * mm, 103 * mm, 36 * mm], styles))
    story.append(Spacer(1, 6 * mm))
    story.append(
        callout(
            "원격 DB는 수정하지 않았습니다. 로컬 복사본에서만 Human 라벨과 학습 상태를 관리하며, "
            "실데이터·모델·보고서는 .gitignore 경로에 두어 GitHub에 자동 업로드되지 않습니다.",
            styles,
            "blue",
        )
    )
    story.append(PageBreak())

    # 3 Data inventory
    story.extend(section("3", "데이터 현황과 큐레이션", "원본을 보존하고 학습용 데이터를 분리한 방식입니다.", styles))
    story.append(
        HorizontalBars(
            [
                ("Human 원본", 4816, "blue"),
                ("Human 학습 사용", 4786, "teal"),
                ("Bot", 3000, "gold"),
                ("Human 제외", 30, "coral"),
            ],
            maximum=5000,
        )
    )
    story.append(p("그림 1. 현재 데이터 구성", styles["caption"]))
    story.append(Spacer(1, 5 * mm))
    story.append(
        make_table(
            [
                ["구분", "건수", "처리", "이유"],
                ["연결 Human", "3,899", "학습 포함", "참여자 그룹 분할 가능"],
                ["익명 Human", "887", "학습 포함, train-only", "수집 주체 확인, 참여자 그룹 불명"],
                ["궤적 누락", "30", "학습 제외", "29개 행동 Feature 계산 불가"],
                ["Rule Bot", "3,000", "학습 포함", "3 family baseline"],
            ],
            [38 * mm, 23 * mm, 47 * mm, 62 * mm],
            styles,
            aligns={1: "RIGHT"},
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.extend(
        bullets(
            [
                "Human 원본 4,816건에는 DB의 source_sample_label=organic 값을 감사 목적으로 보존했습니다.",
                "로컬 작업용 라벨은 human / controlled_collection이며 사용자가 통제 수집 사실을 확인했습니다.",
                "학생·기관 DB ID는 snapshot-local HMAC 가명값으로 변환했고 원래 ID는 내보내지 않았습니다.",
                "30건을 0 Feature로 채우면 모델이 '움직임 없음=Human'을 학습할 수 있어 제외했습니다.",
                "아동 데이터라면 보호자 동의와 연령대 기록은 production 학습 전에 별도 확인해야 합니다.",
            ],
            styles,
        )
    )
    story.append(PageBreak())

    # 4 Data structure
    story.extend(section("4", "파일 및 데이터 구조", "원본, 전처리, 통합 학습셋을 분리해 재계산 가능성을 유지합니다.", styles))
    file_rows = [
        ["계층", "파일", "역할"],
        ["Raw snapshot", "behavior_snapshot.jsonl", "DB 요약 + 정규화 원시 궤적 4,816건"],
        ["Human processed", "human_attempts.jsonl", "AI collect API 모양의 포인터 이벤트 4,786건"],
        ["Human processed", "human_labels.jsonl", "human 정답표 4,786건"],
        ["Human processed", "human_features.jsonl", "29개 Feature가 계산된 Human 입력"],
        ["Audit", "excluded_missing_trace.jsonl", "제외 30건과 missing_pointer_trace 사유"],
        ["Bot interim", "rule_bots_3000.jsonl", "3종 Bot 원시 이벤트"],
        ["Training processed", "bot_features.jsonl", "29개 Feature가 계산된 Bot 입력"],
        ["Training processed", "combined_features.jsonl", "Human 4,786 + Bot 3,000 = 7,786"],
        ["Reports", "training_summary.json", "분할, 일반 test, holdout, 선택 결과"],
        ["Models", "*.joblib", "모델 + threshold + Feature 목록 + 버전"],
    ]
    story.append(make_table(file_rows, [35 * mm, 57 * mm, 78 * mm], styles))
    story.append(Spacer(1, 6 * mm))
    story.append(
        p(
            "data/raw/human_db_snapshot_20260713T061448Z/<br/>"
            "data/processed/human_confirmed_4786_20260713/<br/>"
            "data/processed/local_h4786_b3000_20260713/<br/>"
            "reports/local_h4786_b3000_20260713/<br/>"
            "models/candidate/local_h4786_b3000_20260713/",
            styles["code"],
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(
        callout(
            "<b>핵심 원칙:</b> Feature 정의가 바뀌면 원시 궤적에서 전체를 다시 계산합니다. "
            "feature_schema_version이 다른 행은 한 학습에 섞지 않습니다.",
            styles,
            "teal",
        )
    )
    story.append(PageBreak())

    # 5 Features
    story.extend(section("5", "모델 입력 Feature 29개", "정답 여부가 아니라 어떻게 움직였는지를 수치화합니다.", styles))
    feature_rows = [
        ["그룹", "개수", "Feature", "의미"],
        [
            "기본 움직임",
            "15",
            "event_count, duration_ms, total_distance, displacement, avg/max speed, speed_std, acceleration, jerk, direction_changes, pause, linearity, y_deviation",
            "경로 길이, 속도 변화, 직선성, 흔들림",
        ],
        [
            "이벤트 간격",
            "4",
            "interval_mean_ms, interval_std_ms, interval_cv, duplicate_interval_ratio",
            "브라우저 이벤트 시간의 규칙성",
        ],
        [
            "목표 근처 보정",
            "5",
            "overshoot_count, overshoot_distance, correction_count, endpoint_adjustment_time, final_segment_speed",
            "넘침, 되돌림, 마지막 미세 조정",
        ],
        [
            "조작 요약",
            "5",
            "regrab_count, retry_count, pointercancel_count, empty_click_count, failed_drop_count",
            "재잡기, 취소, 빈 클릭, 실패",
        ],
    ]
    story.append(make_table(feature_rows, [29 * mm, 15 * mm, 88 * mm, 38 * mm], styles))
    story.append(Spacer(1, 6 * mm))
    story.append(p("모델 입력에서 반드시 제외하는 값", styles["h2"]))
    story.append(
        p(
            "attempt_id, challenge_id, session_id, participant_id, label, label_source, bot_family, "
            "generator_version, position_correct, interaction_success, final_drop_error, 기존 AI 판정값",
            styles["code"],
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.extend(
        bullets(
            [
                "정답/오답(position_correct)은 CAPTCHA 결과이지 행동 Feature가 아니므로 입력에서 제외합니다.",
                "좌표는 CAPTCHA 영역 픽셀로 복원하며 시간은 ms, 속도는 px/ms 단위를 사용합니다.",
                "동일 timestamp, 0으로 나누기, NaN/Infinity는 extractor에서 방어합니다.",
                "Replay·요청 빈도·동일 궤적 검사는 행동 모델과 분리된 보안 계층으로 결합해야 합니다.",
            ],
            styles,
        )
    )
    story.append(PageBreak())

    # 6 Pipeline
    story.extend(section("6", "학습 파이프라인", "같은 데이터와 같은 split에서 세 모델을 비교합니다.", styles))
    story.append(PipelineFlow())
    story.append(p("그림 2. 로컬 파일 기반 학습 흐름", styles["caption"]))
    story.append(Spacer(1, 5 * mm))
    pipeline_rows = [
        ["순서", "작업", "검사/산출물"],
        ["1", "Human/Bot JSONL 로드", "라벨·행 수·schema 확인"],
        ["2", "Bot 원시 이벤트 Feature 계산", "Human과 동일한 29개 계약"],
        ["3", "Readiness gate", "최소 건수, 참여자, family, nonfinite"],
        ["4", "그룹 분할", "Human 참여자와 Bot batch 누수 방지"],
        ["5", "RF/XGBoost/LightGBM 학습", "seed=42, class imbalance 처리"],
        ["6", "Validation threshold 선택", "Human FRR 3% 이내에서 Bot Recall 최대"],
        ["7", "Test 1회 평가", "Accuracy, FRR, Recall, F1, AUC, latency"],
        ["8", "Family holdout", "처음 보는 Bot family 일반화"],
        ["9", "Candidate 저장", "production 자동 승격 금지"],
    ]
    story.append(make_table(pipeline_rows, [15 * mm, 70 * mm, 85 * mm], styles))
    story.append(Spacer(1, 5 * mm))
    story.append(
        p(
            "재현 명령: <b>.venv/bin/python -m training.run_local_training</b>",
            styles["code"],
        )
    )
    story.append(PageBreak())

    # 7 Split
    story.extend(section("7", "분할과 데이터 누수 방지", "행을 무작위로 섞지 않고 사람·생성기 그룹을 기준으로 나눕니다.", styles))
    class_counts = split["class_counts"]
    split_rows = [["Split", "Human", "Bot", "합계"]]
    for name in ("train", "val", "test"):
        human = class_counts[name]["human"]
        bot = class_counts[name]["bot"]
        split_rows.append([name.title(), f"{human:,}", f"{bot:,}", f"{human + bot:,}"])
    story.append(make_table(split_rows, [42 * mm] * 4, styles, aligns={1: "RIGHT", 2: "RIGHT", 3: "RIGHT"}))
    story.append(Spacer(1, 7 * mm))
    split_policy = [
        ["대상", "정책", "이유"],
        ["연결 Human", "동일 참여자의 모든 시도를 하나의 split에 배치", "개인별 습관 누수 방지"],
        ["익명 Human 887", "train-only", "동일인 여부를 알 수 없어 검증 오염 방지"],
        ["Rule Bot", "family별 deterministic batch group", "동일 생성 batch 누수 방지"],
        ["Test", "threshold 조정에 사용하지 않음", "최종 성능 낙관 편향 방지"],
    ]
    story.append(make_table(split_policy, [38 * mm, 80 * mm, 52 * mm], styles))
    story.append(Spacer(1, 7 * mm))
    story.append(
        callout(
            "Human 4,786회는 4,786명의 의미가 아닙니다. 연결된 평가 참여자는 25명뿐이므로, "
            "실서비스 일반화에서는 새로운 참여자 수가 시도 횟수보다 더 중요합니다.",
            styles,
            "gold",
        )
    )
    story.append(PageBreak())

    # 8 Models and validation
    story.extend(section("8", "세 모델과 검증 기준", "복잡도보다 같은 조건에서의 공정한 비교를 우선합니다.", styles))
    model_rows = [
        ["모델", "주요 설정", "장점", "주의점"],
        ["RandomForest", "tree 300, class_weight=balanced", "해석 쉬움, baseline 안정적", "새 Jitter 일반화 실패"],
        ["XGBoost", "tree 300, depth 6, lr 0.1, subsample 0.9", "빠르고 일반 test 우수", "새 Accel/Jitter 취약"],
        ["LightGBM", "tree 300, lr 0.1, class_weight=balanced", "holdout 평균 최고", "threshold 포화, Accel 취약"],
    ]
    story.append(make_table(model_rows, [34 * mm, 65 * mm, 36 * mm, 35 * mm], styles))
    story.append(Spacer(1, 6 * mm))
    validation_rows = [
        ["우선순위", "검사", "현재 기준"],
        ["1", "Human FRR", "3% 이하"],
        ["2", "Unseen family 최악 Bot Recall", "candidate 80% 이상"],
        ["3", "Bot Recall", "높을수록 좋음"],
        ["4", "Human F1", "동률 비교"],
        ["5", "추론 시간", "동률 비교/실시간성"],
        ["보조", "ROC-AUC, PR-AUC, Confusion Matrix", "판정 구조 점검"],
        ["운영", "입력별 FRR, calibration, 부하, drift", "production 전 추가"],
    ]
    story.append(make_table(validation_rows, [25 * mm, 76 * mm, 69 * mm], styles))
    story.append(Spacer(1, 6 * mm))
    story.append(
        callout(
            "정확도 100% 하나로 모델을 확정하지 않습니다. CAPTCHA에서는 사람을 막는 비용과 "
            "새로운 자동화를 놓치는 비용을 분리해서 봐야 합니다.",
            styles,
            "coral",
        )
    )
    story.append(PageBreak())

    # 9 Primary test
    story.extend(section("9", "일반 Test 결과", "같은 세 가지 Rule Bot 분포를 그룹으로 나눈 시험입니다.", styles))
    primary_rows = [["모델", "Accuracy", "Human FRR", "Bot Recall", "Human F1", "추론 ms"]]
    for name in ("random_forest", "xgboost", "lightgbm"):
        row = comparison[name]
        primary_rows.append(
            [
                name.replace("random_forest", "RandomForest").replace("xgboost", "XGBoost").replace("lightgbm", "LightGBM"),
                f"{float(row['accuracy']) * 100:.2f}%",
                f"{float(row['human_frr']) * 100:.2f}%",
                f"{float(row['bot_recall']) * 100:.2f}%",
                f"{float(row['human_f1']) * 100:.2f}%",
                f"{float(row['avg_inference_ms']):.4f}",
            ]
        )
    story.append(make_table(primary_rows, [40 * mm, 27 * mm, 27 * mm, 27 * mm, 27 * mm, 22 * mm], styles, aligns={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT"}))
    story.append(Spacer(1, 7 * mm))
    image_cells = []
    for model in ("random_forest", "xgboost", "lightgbm"):
        path = REPORT_DIR / f"confusion_matrix_{model}.png"
        image = Image(str(path), width=50 * mm, height=50 * mm)
        image_cells.append(image)
    image_table = Table([image_cells], colWidths=[CONTENT_W / 3] * 3)
    image_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(image_table)
    story.append(p("그림 3. 일반 test confusion matrix", styles["caption"]))
    story.append(Spacer(1, 5 * mm))
    story.append(
        callout(
            "일반 test에서는 RandomForest와 XGBoost가 100%였습니다. 그러나 세 Bot family가 모두 학습에 "
            "포함된 상태라서 이 수치는 알려진 생성 분포에 대한 baseline 성능입니다.",
            styles,
            "gold",
        )
    )
    story.append(PageBreak())

    # 10 Holdout
    story.extend(section("10", "처음 보는 Bot 적대적 검증", "한 family를 학습에서 완전히 제외한 뒤 탐지율을 측정했습니다.", styles))
    holdout_rows = [["모델", "Accel", "Jitter", "Straight", "최악", "평균"]]
    backgrounds: dict[tuple[int, int], colors.Color] = {}
    name_map = {"random_forest": "RandomForest", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
    for row_index, model in enumerate(("random_forest", "xgboost", "lightgbm"), start=1):
        by_family = {item["held_out_bot_family"]: item["bot_recall"] for item in holdout[model]}
        values = [by_family[family] for family in ("accel", "jitter", "straight")]
        holdout_rows.append(
            [name_map[model], *(f"{value * 100:.1f}%" for value in values), f"{min(values) * 100:.1f}%", f"{sum(values) / len(values) * 100:.1f}%"]
        )
        for column, value in enumerate(values, start=1):
            backgrounds[(row_index, column)] = heat_color(value)
        backgrounds[(row_index, 4)] = heat_color(min(values))
    story.append(make_table(holdout_rows, [40 * mm, 26 * mm, 26 * mm, 26 * mm, 26 * mm, 26 * mm], styles, aligns={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT"}, cell_backgrounds=backgrounds))
    story.append(Spacer(1, 7 * mm))
    story.append(
        HorizontalBars(
            [
                ("RandomForest 최악", 2.1, "coral"),
                ("XGBoost 최악", 31.4, "gold"),
                ("LightGBM 최악", 31.4, "teal"),
                ("배포 gate", 80.0, "blue"),
            ],
            maximum=100,
            height=48 * mm,
        )
    )
    story.append(p("그림 4. unseen-family 최악 Bot Recall과 배포 gate", styles["caption"]))
    story.append(Spacer(1, 4 * mm))
    story.append(
        callout(
            "LightGBM은 최악 31.4%, 평균 71.5%로 상대적으로 가장 나았지만 gate 80%를 통과하지 "
            "못했습니다. 따라서 robust candidate로만 저장했고 deployment_eligible=false입니다.",
            styles,
            "coral",
        )
    )
    story.append(PageBreak())

    # 11 Interpretation and feature importance
    story.extend(section("11", "결과 해석과 중요한 위험", "모델이 맞힌 숫자보다 왜 틀렸는지를 봅니다.", styles))
    importance_rows = [["모델", "상위 Feature", "해석"]]
    for model in ("random_forest", "xgboost", "lightgbm"):
        top = load_csv(REPORT_DIR / f"feature_importance_{model}.csv")[:4]
        features = ", ".join(row["feature"] for row in top)
        importance_rows.append([name_map[model], features, "이벤트 시간 간격과 규칙성에 강하게 의존"])
    story.append(make_table(importance_rows, [37 * mm, 83 * mm, 50 * mm], styles))
    story.append(Spacer(1, 6 * mm))
    story.extend(
        bullets(
            [
                "세 모델 모두 interval_std_ms, interval_mean_ms, interval_cv 같은 시간 간격 Feature를 중요하게 사용했습니다.",
                "규칙형 Bot은 이벤트 간격이 단순해 알려진 분포에서는 쉽게 구분됩니다.",
                "Accel/Jitter를 숨기면 성능이 크게 하락하므로 Bot family 다양성이 현재의 핵심 병목입니다.",
                "LightGBM threshold가 약 0.999999로 포화되어 확률 calibration과 threshold 안정성 검사가 필요합니다.",
                "887 익명 Human은 train-only라 일반 test의 Human 평가는 연결 참여자 25명에 의존합니다.",
            ],
            styles,
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(
        make_table(
            [
                ["위험", "현재 영향", "대응"],
                ["Bot 분포 편향", "새 family Recall 급락", "10~12 family와 실제 자동화 추가"],
                ["Human 다양성 부족", "일부 사용자 오탐 가능", "새 참여자·기기·브라우저 확대"],
                ["확률 미보정", "threshold가 극단적", "calibration curve와 재보정"],
                ["운영 차이", "train-serving skew 가능", "Shadow 로그와 Feature 분포 모니터링"],
                ["Replay/속도 공격", "행동 모델만으로 부족", "rate limit·single use·replay 계층 결합"],
            ],
            [42 * mm, 58 * mm, 70 * mm],
            styles,
        )
    )
    story.append(PageBreak())

    # 12 Deployment gates and data targets
    story.extend(section("12", "실서비스 배포 기준", "데이터 개수와 성능 gate를 동시에 충족해야 합니다.", styles))
    target_rows = [
        ["단계", "Human", "참여자", "Bot", "Family", "목적"],
        ["현재", "4,786", "25명", "3,000", "3종", "Baseline"],
        ["다음 점검", "10,000", "75명+", "10,000", "8종+", "개선 확인"],
        ["Shadow", "15~20k", "150명+", "15~20k", "10~12종", "차단 없이 판정"],
        ["제한 배포", "20~30k", "200명+", "20~30k", "12종+", "일부 트래픽"],
        ["전체 배포", "성능 gate", "대표성", "지속 추가", "새 공격", "Canary 후"],
    ]
    story.append(make_table(target_rows, [29 * mm, 27 * mm, 28 * mm, 27 * mm, 27 * mm, 32 * mm], styles, aligns={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT"}))
    story.append(Spacer(1, 6 * mm))
    gate_rows = [
        ["배포 Gate", "목표", "현재"],
        ["Human FRR", "3% 이하", "2.16% PASS"],
        ["Unseen Bot 최악 Recall", "80% 이상", "31.4% FAIL"],
        ["Mouse/Touch별 FRR", "각 3% 이하", "미완료"],
        ["독립 Human test", "1,500~3,000, 새 참여자", "미확보"],
        ["독립 Bot test", "3 family+, family당 500~1,000", "현재 holdout만"],
        ["Shadow", "실트래픽 안정성", "미진행"],
        ["Rollback/Monitoring", "운영 준비", "미진행"],
    ]
    story.append(make_table(gate_rows, [67 * mm, 57 * mm, 46 * mm], styles))
    story.append(Spacer(1, 6 * mm))
    story.append(
        callout(
            "같은 3종 Bot을 5,000개씩 늘리는 것보다 새로운 generator·재생·곡선·구간 이동·실제 브라우저 "
            "자동화 family를 확보하는 것이 우선입니다. 수량은 다양성을 대체하지 못합니다.",
            styles,
            "gold",
        )
    )
    story.append(PageBreak())

    # 13 Next plan
    story.extend(section("13", "다음 학습 계획", "다음 체크포인트에서 성능 원인을 설명할 수 있도록 수집과 시험을 설계합니다.", styles))
    next_rows = [
        ["우선", "작업", "완료 기준"],
        ["P0", "Bot family 확대", "최소 8종, generator/version 분리"],
        ["P0", "완전 비공개 holdout 보관", "학습 미사용 Bot 3종 이상"],
        ["P0", "Human 참여자 확대", "10k 시점 연결 75명 이상"],
        ["P1", "입력별 평가", "Mouse/Touch/브라우저별 FRR"],
        ["P1", "확률 calibration", "threshold 포화 해소와 안정성 확인"],
        ["P1", "Learning curve", "5k/10k/15k 성능 변화"],
        ["P2", "Shadow 운영", "판정만 기록, 사용자 차단 없음"],
        ["P2", "보안 계층 결합", "replay/rate limit/single-use와 모델 결합"],
    ]
    story.append(make_table(next_rows, [21 * mm, 70 * mm, 79 * mm], styles))
    story.append(Spacer(1, 7 * mm))
    story.append(p("1만 건 체크포인트 비교 문장", styles["h2"]))
    story.append(
        p(
            "Human 데이터가 4,786건에서 [새 사용 건수]건으로 증가하면서 Human FRR은 "
            "2.16%에서 [새 FRR]로, unseen Bot 최악 Recall은 31.4%에서 [새 Recall]로 변했다. "
            "변화의 주된 원인은 [Human 다양성/Bot family/모델/threshold]로 분석된다.",
            styles["quote"],
        )
    )
    story.append(Spacer(1, 7 * mm))
    story.append(
        make_table(
            [
                ["항목", "약 5천", "약 1만", "약 1.5만"],
                ["Human 학습", "4,786", "추후", "추후"],
                ["Human FRR", "2.16%", "추후", "추후"],
                ["Unseen 최악 Recall", "31.4%", "추후", "추후"],
                ["Robust 후보", "LightGBM", "추후", "추후"],
                ["배포 가능", "아니요", "추후", "추후"],
            ],
            [56 * mm, 38 * mm, 38 * mm, 38 * mm],
            styles,
            aligns={1: "CENTER", 2: "CENTER", 3: "CENTER"},
        )
    )
    story.append(PageBreak())

    # 14 Reproduction and files
    story.extend(section("14", "재현 방법과 주요 파일", "다른 개발자가 같은 결과를 다시 만들기 위한 안내입니다.", styles))
    story.append(p("환경", styles["h2"]))
    story.append(
        p(
            "scikit-learn 1.9.0 | XGBoost 2.1.4 | LightGBM 4.6.0 | "
            "pandas 2.3.3 | numpy 2.4.6 | macOS libomp",
            styles["code"],
        )
    )
    story.append(p("실행", styles["h2"]))
    story.append(p("cd /Users/apple/Documents/최최최종/ai-service<br/>MPLCONFIGDIR=/private/tmp/catchap-matplotlib .venv/bin/python -m training.run_local_training", styles["code"]))
    story.append(p("검증", styles["h2"]))
    story.append(p("PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_local_training.py tests/test_split.py tests/test_readiness.py tests/test_training_select.py", styles["code"]))
    story.append(Spacer(1, 5 * mm))
    story.append(
        make_table(
            [
                ["파일", "설명"],
                ["training/run_local_training.py", "로컬 JSONL 변환·분할·학습·holdout·저장"],
                ["app/services/feature_extractor.py", "29 Feature 단일 정의"],
                ["docs/LOCAL_TRAINING.md", "구조·검증·실제 결과"],
                ["docs/EXPERIMENT_HISTORY.md/.csv", "발표용 체크포인트 이력"],
                ["reports/.../training_summary.json", "전체 수치의 기준 파일"],
                ["models/candidate/.../*.joblib", "후보 모델 3개"],
            ],
            [76 * mm, 94 * mm],
            styles,
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(
        callout(
            "검증 테스트 18개가 통과했고, 저장된 세 모델을 다시 로드해 29 Feature 입력의 predict_proba 결과가 "
            "모두 유한값인지 확인했습니다. output hash도 manifest와 일치합니다.",
            styles,
            "teal",
        )
    )
    story.append(PageBreak())

    # 15 Appendix
    story.extend(section("15", "지표 설명과 참고 자료", "발표와 의사결정에서 사용하는 용어를 통일합니다.", styles))
    metric_rows = [
        ["지표", "뜻", "해석"],
        ["Human FRR", "실제 사람을 Bot으로 잘못 막은 비율", "낮을수록 좋음, 목표 3% 이하"],
        ["Bot Recall", "실제 Bot 중 Bot으로 탐지한 비율", "높을수록 좋음"],
        ["Human Precision", "Human 판정 중 실제 Human 비율", "Bot 통과와 함께 해석"],
        ["Human F1", "Human Precision과 Recall의 조화 평균", "균형 비교"],
        ["ROC-AUC", "전체 threshold 범위의 순위 분류력", "클래스 불균형에 주의"],
        ["PR-AUC", "Precision-Recall 곡선 면적", "희소 클래스에서 유용"],
        ["Family holdout", "한 Bot 종류를 학습에서 제외한 시험", "새 공격 일반화 확인"],
        ["Calibration", "예측 확률과 실제 빈도의 일치", "threshold 안정성"],
        ["Drift", "운영 중 입력 분포 변화", "재학습·경보 기준"],
    ]
    story.append(make_table(metric_rows, [39 * mm, 77 * mm, 54 * mm], styles))
    story.append(Spacer(1, 7 * mm))
    story.append(p("참고 자료", styles["h2"]))
    refs = [
        "NIST AI RMF Core - 대표성, 배포 환경 검증, 일반화 한계, 운영 모니터링: https://airc.nist.gov/airmf-resources/airmf/5-sec-core/",
        "scikit-learn Learning Curve - 데이터 증가에 따른 train/validation 변화: https://scikit-learn.org/stable/modules/learning_curve.html",
        "Google Production ML Monitoring - raw/feature schema와 분포 모니터링: https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring",
        "프로젝트 내부 문서: docs/DATA_SCHEMA.md, docs/TRAINING_GUIDE.md, docs/LOCAL_TRAINING.md, docs/EXPERIMENT_HISTORY.md",
    ]
    story.extend(bullets(refs, styles))
    story.append(Spacer(1, 8 * mm))
    story.append(
        callout(
            "최종 요약: 현재는 모델을 확정한 단계가 아니라 baseline과 검증 체계를 확정한 단계입니다. "
            "다음 성공 조건은 Human 수량 증가만이 아니라 새로운 참여자와 새로운 Bot family에서 성능이 유지되는 것입니다.",
            styles,
            "blue",
        )
    )

    doc.build(story, onFirstPage=first_page, onLaterPages=later_pages)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(build_report())
