"""Render a README GIF from fresh rules-only CLI results, not invented model output.

Optional authoring dependency: Pillow==12.3.0. See docs/assets/README.md.
The animation is a designed summary, not a screen recording or latency benchmark.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont
from PIL import __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1200, 720
COLORS = {
    "background": "#0B1220", "panel": "#111D30", "border": "#25354D",
    "text": "#EDF4FF", "muted": "#A9BBD2", "teal": "#62E4C6",
    "red": "#FF9DAD", "amber": "#F4CC7A", "blue": "#9ABDFF",
}
TITLES = {"retry_safety": "Retry safety", "ack_durability": "Acknowledgement durability",
          "queue_overload": "Queue overload"}


def font(size, mono=False, bold=False):
    if mono:
        choices = ["C:/Windows/Fonts/consola.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/System/Library/Fonts/Menlo.ttc"]
    elif bold:
        choices = ["C:/Windows/Fonts/seguisb.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
    else:
        choices = ["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for candidate in choices:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def wrap(draw, text, face, maximum):
    lines, current = [], ""
    for word in text.split():
        trial = (current + " " + word).strip()
        if draw.textlength(trial, font=face) > maximum:
            if not current:
                raise ValueError("An animation word does not fit its text box")
            lines.append(current)
            current = word
        else:
            current = trial
    return lines + ([current] if current else [])


def text(draw, xy, value, size=22, color="text", bold=False, mono=False):
    draw.text(xy, value, font=font(size, mono, bold), fill=COLORS.get(color, color))


def pill(draw, xy, value, color, width=200):
    x, y = xy
    draw.rounded_rectangle((x, y, x + width, y + 34), radius=9, fill=COLORS["panel"], outline=COLORS[color])
    face = font(17, bold=True)
    x += (width - draw.textlength(value, font=face)) / 2
    draw.text((x, y + 6), value, font=face, fill=COLORS[color])


def base(step, title, subtitle):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), COLORS["background"])
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((48, 30, 88, 70), radius=11, fill=COLORS["teal"])
    text(draw, (60, 33), "R", size=26, color="background", bold=True)
    text(draw, (102, 28), "ArchRivet", size=32, bold=True)
    text(draw, (304, 42), "A SECOND OPINION FOR YOUR SYSTEM DESIGN", size=16, color="muted")
    pill(draw, (944, 34), "EXPERIMENTAL ALPHA", "amber", 208)
    text(draw, (48, 104), f"{step:02d}  {title}", size=37, bold=True)
    text(draw, (48, 158), subtitle, size=21, color="muted")
    for number, label in enumerate(["REVIEW", "REVISE", "RE-CHECK", "CALCULATE"], start=1):
        x = 48 + (number - 1) * 280
        color = "teal" if number == step else "border"
        draw.rounded_rectangle((x, 684, x + 258, 688), radius=2, fill=COLORS[color])
        text(draw, (x, 657), label, size=15, color="teal" if number == step else "muted", bold=True)
    return canvas, draw


def card(draw, y, title, body, label, color):
    draw.rounded_rectangle((48, y, 1152, y + 100), radius=13, fill=COLORS["panel"], outline=COLORS["border"])
    draw.rounded_rectangle((48, y + 12, 52, y + 88), radius=2, fill=COLORS[color])
    text(draw, (72, y + 9), title, size=24, bold=True)
    pill(draw, (932, y + 10), label, color)
    lines = wrap(draw, body, font(20), 1052)
    if len(lines) > 2:
        raise ValueError(f"Too much animation text for {title}")
    for index, line in enumerate(lines):
        text(draw, (72, y + 46 + index * 24), line, size=20, color="muted")


def review_frame(review, revised=False, visible=3):
    name = "complete" if revised else "flawed"
    title = "Review the revision" if revised else "Review the draft"
    subtitle = ("The supplied mechanisms now satisfy the three rule checks." if revised else
                "The requirements promise safe retries, durable acceptance and a bounded queue.")
    canvas, draw = base(3 if revised else 1, title, subtitle)
    draw.rounded_rectangle((48, 203, 1152, 248), radius=9, fill="#070D17", outline=COLORS["border"])
    text(draw, (66, 213), f"$ arch-review review examples/{name}.json --rules-only", size=21, mono=True, color="teal")
    for index, check in enumerate(review["checks"][:visible]):
        card(draw, 270 + index * 112, TITLES[check["check_id"]], check["reason"],
             check["verdict"].upper(), "teal" if check["verdict"] == "supported" else "red")
    text(draw, (48, 609), "Capacity: UNKNOWN", size=21, color="amber", bold=True)
    text(draw, (307, 611), "rules_only / partial  |  Real CLI results; presentation timing only", size=19, color="muted")
    return canvas


def repair_frame(request, visible=3):
    canvas, draw = base(2, "Make the guarantees explicit", "These edits are in examples/complete.json. Verify them in your implementation.")
    design = request["design"]
    retry, ack, queue = design["retry"], design["acknowledgement"], design["queue"]
    if not (retry["dedup_enforced"] and retry["dedup_atomic_with_effect"] and ack["after_durable_commit"]
            and queue["policy"] == "reject_before_accept" and queue["preserves_accepted_messages"]
            and queue["retry_backoff"]):
        raise ValueError("Example changes no longer match this walkthrough; update its explanations")
    changes = [
        ("Make retries safe", "Enforce atomic duplicate protection. Keep IDs for "
         f"{retry['dedup_retention_seconds'] / 3600:g} hours; the maximum retry age is "
         f"{retry['max_redelivery_age_seconds'] / 3600:g} hours."),
        ("Acknowledge after the commit", "Commit before reporting success, with the stated failure-survival guarantees: "
         + ", ".join(ack["survives_failures"]) + "."),
        ("Define overload handling", f"Bound the queue at {queue['max_messages']:,} messages. "
         "Reject before acceptance, preserve accepted work and back off retries."),
    ]
    text(draw, (48, 215), "CHANGE THE DESIGN FACTS, THEN RUN THE REVIEW AGAIN", size=18, color="blue", bold=True)
    for index, (title, body) in enumerate(changes[:visible]):
        card(draw, 270 + index * 112, title, body, "DESIGN CHANGE", "blue")
    text(draw, (48, 609), "A stated guarantee still needs a failure test.", size=21, color="amber")
    return canvas


def workload_frame(review, request):
    canvas, draw = base(4, "Calculate demand. Keep unknowns visible.", "Workload estimates come from code, with explicit input assumptions.")
    estimates = review["workload_estimates"]["estimates"]
    work = request["workload"]
    metrics = [
        (f"{estimates['peak_messages']['value']:,.0f}", "peak messages / second",
         f"{work['average_messages_per_second']:g} average x {work['peak_factor']:g} peak factor"),
        (f"{estimates['peak_deliveries']['value']:,.0f}", "peak deliveries / second",
         f"{estimates['peak_messages']['value']:g} messages x {work['average_recipients']:g} recipients"),
        (f"{estimates['retained_storage']['value'] / 1e12:.3f} TB", "retained storage estimate",
         f"{work['retention_days']:g} days, {work['replication_factor']} replicas, "
         f"{(work['storage_overhead_factor'] - 1) * 100:.0f}% overhead"),
    ]
    for index, (value, label, assumption) in enumerate(metrics):
        x = 48 + index * 374
        draw.rounded_rectangle((x, 230, x + 356, 413), radius=14, fill=COLORS["panel"], outline=COLORS["border"])
        text(draw, (x + 20, 249), value, size=49, color="teal", bold=True)
        text(draw, (x + 20, 315), label, size=21, bold=True)
        for row, line in enumerate(wrap(draw, assumption, font(18), 318)):
            text(draw, (x + 20, 354 + row * 22), line, size=18, color="muted")
    text(draw, (48, 439), f"Storage also assumes {work['message_bytes']:,}-byte messages and "
         f"{work['storage_copies_per_message']:g} stored copy per message.", size=21, color="muted")
    draw.rounded_rectangle((48, 493, 1152, 601), radius=14, fill="#29251D", outline=COLORS["amber"])
    text(draw, (72, 507), "PRODUCTION CAPACITY: UNKNOWN", size=25, color="amber", bold=True)
    text(draw, (72, 546), "A matching workload and hardware benchmark is still required.", size=24)
    text(draw, (48, 615), "Try the CLI today. Connect your AI through MCP when ready.", size=21, color="muted")
    return canvas


def capture(name):
    relative = f"examples/{name}.json"
    command = [sys.executable, "-m", "arch_review", "review", relative, "--rules-only"]
    env = dict(os.environ, ARCH_REVIEW_HOME=str(ROOT / ".runtime"), PYTHONUTF8="1")
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                               timeout=30, check=True)
    review = json.loads(completed.stdout)
    expected = "contradicted" if name == "flawed" else "supported"
    if (review["review_mode"] != "rules_only" or review["review_status"] != "partial"
            or len(review["checks"]) != 3 or any(c["verdict"] != expected for c in review["checks"])
            or review["capacity"]["evidence_status"] != "unknown"):
        raise ValueError("Demo results changed; review the story before regenerating the GIF")
    content = (ROOT / relative).read_bytes()
    return review, json.loads(content), {
        "input": relative, "input_sha256": hashlib.sha256(content).hexdigest(),
        "command": f"python -m arch_review review {relative} --rules-only",
        "review_mode": review["review_mode"], "review_status": review["review_status"],
        "checks": [{key: check[key] for key in ("check_id", "verdict", "reason")} for check in review["checks"]],
        "workload_estimates": review["workload_estimates"], "capacity": review["capacity"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets")
    args = parser.parse_args()
    flawed, _, flawed_record = capture("flawed")
    complete, request, complete_record = capture("complete")
    frames = [review_frame(flawed)]
    durations = [5000]
    for visible, duration in [(1, 800), (2, 800), (3, 4000)]:
        frames.append(repair_frame(request, visible))
        durations.append(duration)
    for visible, duration in [(1, 800), (2, 800), (3, 4000)]:
        frames.append(review_frame(complete, revised=True, visible=visible))
        durations.append(duration)
    frames.append(workload_frame(complete, request))
    durations.append(6000)
    args.output.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output / "archrivet-demo.png", optimize=True)
    palette_frames = [frame.quantize(colors=128, dither=Image.Dither.NONE) for frame in frames]
    gif = args.output / "archrivet-demo.gif"
    palette_frames[0].save(gif, save_all=True, append_images=palette_frames[1:], duration=durations,
                           loop=0, optimize=True, disposal=2)
    with Image.open(gif) as saved:
        if saved.n_frames != len(frames) or saved.size != (WIDTH, HEIGHT):
            raise ValueError("Saved GIF has unexpected frames or dimensions")
        for index, expected in enumerate(palette_frames):
            saved.seek(index)
            if ImageChops.difference(saved.convert("RGB"), expected.convert("RGB")).getbbox():
                raise ValueError(f"Saved GIF frame {index} differs from its rendered source")
    metadata = {
        "kind": "animated_summary_of_actual_cli_results", "screen_recording": False,
        "model_inference_used": False, "timing_is_benchmark": False, "pillow_version": pillow_version,
        "width": WIDTH, "height": HEIGHT, "frames": len(frames), "duration_ms": sum(durations),
        "source_results": [flawed_record, complete_record],
        "files": {name: {"sha256": hashlib.sha256((args.output / name).read_bytes()).hexdigest(),
                         "bytes": (args.output / name).stat().st_size}
                  for name in ("archrivet-demo.gif", "archrivet-demo.png")},
    }
    (args.output / "archrivet-demo.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames": len(frames), "duration_ms": sum(durations), "gif_bytes": gif.stat().st_size,
                      "mode": "rules_only", "real_cli_captures": 2}))


if __name__ == "__main__":
    main()
