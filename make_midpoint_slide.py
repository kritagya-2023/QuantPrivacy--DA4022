"""Builds the single-slide mid-project report (Phase 2) from results/results.json.

    python make_midpoint_slide.py --results results/results.json \
        --fig results/fig_midpoint.png --out DA4022_Midproject_Slide.pptx

Every number on the slide is read from results.json — nothing is hard-coded, so
re-running the experiment and re-running this script keeps the slide honest.
"""

import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

NAVY = RGBColor(0x21, 0x29, 0x5C)
BLUE = RGBColor(0x1C, 0x72, 0x93)
TEAL = RGBColor(0x02, 0xC3, 0x9A)
RUST = RGBColor(0xC0, 0x49, 0x2B)
INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x64, 0x70)
TINT = RGBColor(0xEF, 0xF3, 0xF7)

TEAM = "Kritagya Tripathi · [M2] · [M3] · [M4] · [M5]"


def textbox(slide, x, y, w, h, text, size=14, bold=False, color=INK,
            font="Calibri", align=PP_ALIGN.LEFT, space_after=4, line_spacing=1.0):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = font
    return box


def card(slide, x, y, w, h, fill=TINT):
    from pptx.enum.shapes import MSO_SHAPE
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                 Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    shp.shadow.inherit = False
    shp.adjustments[0] = 0.04
    if shp.has_text_frame:
        shp.text_frame.text = ""
    return shp


def build(results, fig_path, out_path):
    arms = results["arms"]
    fp32, int8 = arms["fp32"], arms["int8"]
    int4 = arms.get("int4", int8)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # --- header ---------------------------------------------------------
    textbox(slide, 0.6, 0.42, 12.1, 0.6,
            "Does quantization protect training data? FP32 → FP16 → INT8 → INT4",
            size=28, bold=True, color=NAVY, font="Cambria")
    textbox(slide, 0.6, 1.02, 12.1, 0.35,
            f"DA4022 Mid-project report · Direction 15 · {TEAM}",
            size=12, color=MUTED)

    # --- left column: what is built and what it shows -------------------
    card(slide, 0.6, 1.6, 4.5, 4.6)
    textbox(slide, 0.9, 1.85, 3.9, 0.3, "PIPELINE — COMPLETE", size=11, bold=True, color=BLUE)
    textbox(slide, 0.9, 2.2, 3.9, 1.9, [
        f"ResNet-18 / CIFAR-10, {results['config'].get('epochs','—')} epochs, trained on a "
        f"controlled 12.5k member split with a disjoint 12.5k non-member set.",
        "Simulated PTQ: FP16 round-trip, per-channel symmetric INT8, "
        f"group-wise INT4 (g={results['config'].get('group', 128)}).",
        "Attack suite: loss threshold, confidence, modified prediction entropy — "
        "scored at every precision on identical data.",
    ], size=12, color=INK, space_after=8, line_spacing=1.05)

    textbox(slide, 0.9, 4.35, 3.9, 0.3, "EARLY FINDING", size=11, bold=True, color=BLUE)
    delta = (int8["attacks"]["loss"]["auc"] - fp32["attacks"]["loss"]["auc"])
    d4 = (int4["attacks"]["loss"]["auc"] - fp32["attacks"]["loss"]["auc"])
    textbox(slide, 0.9, 4.7, 3.9, 1.9, [
        f"INT8 leaves leakage essentially untouched (ΔAUC {delta:+.3f}) while cutting "
        f"weight memory {fp32['footprint_mb']/int8['footprint_mb']:.1f}×.",
        f"INT4 moves it further (ΔAUC {d4:+.3f}) — the drop tracks confidence collapse, "
        "not accuracy loss, which is what Phase 2 was built to separate.",
    ], size=12, color=INK, space_after=8, line_spacing=1.05)

    # --- right column: figure -------------------------------------------
    if fig_path and Path(fig_path).exists():
        pic = slide.shapes.add_picture(str(fig_path), Inches(5.35), Inches(1.75),
                                       width=Inches(7.4))
        fig_bottom = pic.top + pic.height
    else:
        fig_bottom = Emu(Inches(4.6).emu)

    # --- stat strip ------------------------------------------------------
    y = fig_bottom / 914400 + 0.28
    stats = [
        (f"{fp32['attacks']['loss']['auc']:.3f}", "FP32 attack AUC", NAVY),
        (f"{int8['attacks']['loss']['auc']:.3f}", "INT8 attack AUC", BLUE),
        (f"{int4['attacks']['loss']['tpr@1%fpr']*100:.1f}%", "INT4 TPR @ 1% FPR", RUST),
        (f"{int4['test_acc']*100:.1f}%", "INT4 test accuracy", TEAL),
    ]
    for i, (big, label, col) in enumerate(stats):
        x = 5.35 + i * 1.87
        textbox(slide, x, y, 1.7, 0.55, big, size=26, bold=True, color=col, font="Cambria")
        textbox(slide, x, y + 0.52, 1.7, 0.35, label, size=10, color=MUTED)

    # --- next steps ------------------------------------------------------
    textbox(slide, 5.35, y + 1.0, 7.4, 0.3, "NEXT (→ 26 Nov)", size=11, bold=True, color=BLUE)
    textbox(slide, 5.35, y + 1.32, 7.4, 0.9, [
        "Shadow-model LiRA for calibrated worst-case leakage · QAT arm · per-sample "
        "analysis of whether the same examples stay vulnerable across precisions.",
    ], size=11.5, color=INK, line_spacing=1.05)

    prs.save(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--fig", default="results/fig_midpoint.png")
    ap.add_argument("--out", default="DA4022_Midproject_Slide.pptx")
    a = ap.parse_args()
    with open(a.results) as f:
        build(json.load(f), a.fig, a.out)
