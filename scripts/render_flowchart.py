"""Illustrate the recorded rules-mode cache review as a lightweight animated flow.

Documentation authoring only. Uses Pillow and the existing demo font helper.
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw
from render_readme_demo import font

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets"
BG, PANEL, TEXT, MUTED = "#0b1425", "#14243a", "#edf6ff", "#a7bad1"
TEAL, AMBER = "#65ead4", "#ffd08a"
BOXES = {"A": (50, 218, 370, 350), "B": (480, 218, 800, 350), "C": (910, 218, 1230, 350),
         "D": (910, 470, 1230, 602), "E": (480, 470, 800, 602), "F": (50, 470, 370, 602)}
EDGES = {"AB": [(370, 284), (480, 284)], "BC": [(800, 284), (910, 284)],
         "CD": [(1070, 350), (1070, 470)], "DE": [(910, 536), (800, 536)],
         "EF": [(480, 536), (370, 536)],
         "FB": [(210, 470), (210, 402), (640, 402), (640, 350)]}
ACTIVE = [("AB",), ("BC",), ("CD", "DE"), ("EF", "FB"), ("BC", "CD", "DE")]
TITLES = ["1. Start with your design and requirements", "2. Check the facts you actually supplied",
          "3. Turn a missing detail into a clear question", "4. Add one fact and review again",
          "5. See exactly why the finding changed"]
CAPTIONS = ["Your requirement: prices must be less than five seconds old.",
            "ArchRivet checks the declared facts and keeps assumptions visible.",
            "Missing fact: the oldest price allowed during a database outage.",
            "Your clarification: outage prices can be up to 60 seconds old.",
            "60 seconds exceeds the five-second limit. Capacity remains unknown."]


def point_on_path(points, t):
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    distance = t * sum(lengths)
    for a, b, length in zip(points, points[1:], lengths):
        if distance <= length:
            return tuple(x + (y - x) * distance / length for x, y in zip(a, b))
        distance -= length
    return points[-1]


def render(step=None, t=0):
    image = Image.new("RGB", (1280, 780), BG)
    d = ImageDraw.Draw(image)
    mark = Image.open(OUT / "archrivet-icon-128.png").convert("RGBA").resize((52, 52))
    image.paste(mark, (44, 26), mark)
    d.text((110, 29), "ArchRivet", font=font(28, bold=True), fill=TEXT)
    d.text((941, 39), "INSIDE YOUR AI ASSISTANT", font=font(16, bold=True), fill=MUTED)
    title = "How a design becomes a review" if step is None else TITLES[step]
    d.text((50, 108), title, font=font(35, bold=True), fill=TEXT)
    d.text((50, 162), "Example: an online store's product-price cache", font=font(22), fill=MUTED)
    active = () if step is None else ACTIVE[step]
    for key, path in EDGES.items():
        color = TEAL if key in active else "#41536d"
        d.line(path, fill=color, width=4 if key in active else 2, joint="curve")
        end, previous = path[-1], path[-2]
        angle = math.atan2(end[1] - previous[1], end[0] - previous[0])
        arrow = [end] + [(end[0] - 12 * math.cos(angle + turn), end[1] - 12 * math.sin(angle + turn))
                         for turn in (-.45, .45)]
        d.polygon(arrow, fill=color)
        if key in active:
            x, y = point_on_path(path, t)
            d.ellipse((x-6, y-6, x+6, y+6), fill="#eafff7")
    d.text((304, 374), "Rerun when you supply new facts", font=font(18), fill=MUTED)
    revised = step == 4
    labels = {
        "A": ("YOUR IDEA", ["Prices must be less than", "five seconds old."]),
        "B": ("AI ASSISTANT", ["Drafts the architecture.", "Supplies stated facts."]),
        "C": ("ARCHRIVET CHECKS", ["Rules, evidence, arithmetic.", "Optional: Laya decision."]),
        "D": ("REVIEW RESULT", ["60 seconds exceeds the limit.", "Requirement contradicted."] if revised
              else ["Missing outage-age bound.", "Insufficient information."]),
        "E": ("AI EXPLANATION", ["Shows the changed finding.", "Capacity stays unknown."] if revised
              else ["Names what is missing.", "Asks a focused question."]),
        "F": ("YOUR CLARIFICATION", ["Outage prices can be", "up to 60 seconds old."] if step in (3, 4)
              else ["Supply the missing fact.", "Or leave it unknown."]),
    }
    for key, box in BOXES.items():
        selected = any(key in edge for edge in active)
        d.rounded_rectangle(box, radius=18, fill="#173e46" if selected else PANEL,
                            outline=TEAL if selected else "#314159", width=3 if selected else 1)
        x, y = box[0] + 22, box[1] + 19
        label, lines = labels[key]
        d.text((x, y), label, font=font(17, bold=True), fill=TEAL if selected else MUTED)
        for i, line in enumerate(lines):
            d.text((x, y + 38 + i * 28), line, font=font(20), fill=AMBER if key == "C" and i == 1 else TEXT)
    caption = "Ask, clarify, and compare. Unknown facts remain unknown." if step is None else CAPTIONS[step]
    d.text((50, 657), caption, font=font(24, bold=True), fill=TEXT)
    if step is not None:
        for i in range(5):
            d.rounded_rectangle((50+i*238, 712, 274+i*238, 717), radius=2, fill=TEAL if i == step else "#314159")
    d.text((50, 741), "Illustrated rules-mode example. Optional Laya is benchmarked separately. Timing is illustrative.",
           font=font(16), fill=MUTED)
    return image


def main():
    static = render()
    static.save(OUT / "archrivet-flow.png", optimize=True)
    frames, durations = [], []
    for step in range(5):
        for i in range(6):
            frames.append(render(step, i / 5).resize((960, 585), Image.Resampling.LANCZOS))
            durations.append(2400 if i == 5 else 120)
    palette_sheet = Image.new("RGB", (960, 585 * 5))
    for step in range(5):
        palette_sheet.paste(frames[step * 6], (0, step * 585))
    palette = palette_sheet.quantize(colors=128)
    indexed = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    indexed[0].save(OUT / "archrivet-flow.gif", save_all=True, append_images=indexed[1:],
                    duration=durations, loop=0, optimize=False, disposal=1)
    preview = Image.new("RGB", (1280, 780))
    for i, step in enumerate((0, 2, 3, 4)):
        preview.paste(render(step, .5).resize((640, 390)), ((i % 2)*640, (i // 2)*390))
    preview.save(OUT / "archrivet-flow-preview.png")
    print("Rendered flowchart GIF, static fallback and preview.")


if __name__ == "__main__":
    main()
