"""Render ArchRivet's original vector mark and its lightweight looping animation.

Authoring only: pip install Pillow==11.3.0. No plugin runtime dependency.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SIZE = 512
LEFT, TOP, RIGHT = (138, 366), (256, 130), (374, 366)
BRIDGE_LEFT, HUB, BRIDGE_RIGHT = (180, 282), (256, 282), (332, 282)
MINT, BLUE, AMBER = (101, 234, 212), (124, 165, 255), (255, 208, 138)


def mix(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def interpolate(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def svg(animated=False):
    motion = """
    <style>
      .signal { opacity: 0; }
      .arch-signal { animation: arch 4.8s linear infinite; }
      .bridge-signal { animation: bridge 4.8s linear infinite; }
      @keyframes arch {
        0%, 10% { opacity: 0; transform: translate(138px,366px); }
        15% { opacity: 1; }
        42% { opacity: 1; transform: translate(256px,130px); }
        74% { opacity: 1; transform: translate(374px,366px); }
        80%, 100% { opacity: 0; transform: translate(374px,366px); }
      }
      @keyframes bridge {
        0%, 40% { opacity: 0; transform: translate(180px,282px); }
        45% { opacity: 1; }
        70% { opacity: 1; transform: translate(332px,282px); }
        75%, 100% { opacity: 0; transform: translate(332px,282px); }
      }
      @media (prefers-reduced-motion: reduce) {
        .signal { animation: none; opacity: 0; }
      }
    </style>
    <g class="signal arch-signal"><circle r="18" fill="#baffed" opacity=".15"/>
      <circle r="7" fill="#effffb"/></g>
    <g class="signal bridge-signal"><circle r="16" fill="#ffd08a" opacity=".18"/>
      <circle r="6" fill="#fff2d7"/></g>""" if animated else ""
    document = f'''<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512"
      viewBox="0 0 512 512" role="img" aria-labelledby="title desc">
  <title id="title">ArchRivet</title>
  <desc id="desc">A cyan and blue architectural A joined by an amber rivet.</desc>
  <defs>
    <linearGradient id="tile" x2="1" y2="1">
      <stop stop-color="#0b1425"/><stop offset="1" stop-color="#172a40"/>
    </linearGradient>
    <linearGradient id="arch" x1="138" y1="0" x2="374" y2="0" gradientUnits="userSpaceOnUse">
      <stop stop-color="#65ead4"/><stop offset="1" stop-color="#7ca5ff"/>
    </linearGradient>
    <clipPath id="tile-clip"><rect x="8" y="8" width="496" height="496" rx="112"/></clipPath>
  </defs>
  <rect x="8" y="8" width="496" height="496" rx="112" fill="url(#tile)"/>
  <g clip-path="url(#tile-clip)" stroke="#c4e7ff" opacity=".035">
    <path d="M104 8V504 M180 8V504 M256 8V504 M332 8V504 M408 8V504
      M8 104H504 M8 180H504 M8 256H504 M8 332H504 M8 408H504"/>
  </g>
  <rect x="9" y="9" width="494" height="494" rx="111" fill="none" stroke="#30475d" stroke-width="2"/>
  <path d="M138 366L256 130L374 366 M180 282H332" fill="none"
    stroke="#65ead4" stroke-width="54" opacity=".035" stroke-linejoin="round" stroke-linecap="round"/>
  <path d="M138 366L256 130L374 366 M180 282H332" fill="none"
    stroke="url(#arch)" stroke-width="28" stroke-linejoin="round" stroke-linecap="round"/>
  <g fill="#102134" stroke-width="7">
    <circle cx="138" cy="366" r="20" stroke="#65ead4"/>
    <circle cx="256" cy="130" r="20" stroke="#8edddf"/>
    <circle cx="374" cy="366" r="20" stroke="#7ca5ff"/>
  </g>
  <circle cx="256" cy="282" r="25" fill="#102134"/>
  <circle cx="256" cy="282" r="17" fill="#ffd08a"/>
  <path d="M250 282H262" stroke="#655231" stroke-width="4" stroke-linecap="round"/>
  {motion}
</svg>
'''
    return "\n".join(line.rstrip() for line in document.splitlines()) + "\n"


def background():
    # Oversampling keeps the raster edges clean at tiny toolbar sizes.
    scale = 2
    n = SIZE * scale
    image = Image.new("RGBA", (n, n))
    d = ImageDraw.Draw(image)
    for y in range(n):
        d.line((0, y, n, y), fill=mix((11, 20, 37), (23, 42, 64), y / n))
    mask = Image.new("L", (n, n))
    ImageDraw.Draw(mask).rounded_rectangle((16, 16, n - 16, n - 16), 224, fill=255)
    image.putalpha(mask)
    d = ImageDraw.Draw(image)
    for p in (104, 180, 256, 332, 408):
        d.line((p * scale, 40 * scale, p * scale, 472 * scale), fill=(28, 46, 63), width=1)
        d.line((40 * scale, p * scale, 472 * scale, p * scale), fill=(28, 46, 63), width=1)
    d.rounded_rectangle((18, 18, n - 18, n - 18), 222, outline=(48, 71, 93), width=3)
    return image


def render(phase=None, size=512):
    image = background()
    s = image.width / SIZE
    d = ImageDraw.Draw(image)

    def line(a, b, color, width):
        d.line(tuple(v * s for v in (*a, *b)), fill=color, width=round(width * s))

    def circle(center, radius, fill):
        x, y = center
        d.ellipse(((x - radius) * s, (y - radius) * s,
                   (x + radius) * s, (y + radius) * s), fill=fill)

    # Small overlapping round strokes form the horizontal color gradient.
    for a, b in ((LEFT, TOP), (TOP, RIGHT), (BRIDGE_LEFT, BRIDGE_RIGHT)):
        for i in range(180):
            p = interpolate(a, b, i / 180)
            q = interpolate(a, b, (i + 1) / 180)
            color = mix(MINT, BLUE, max(0, min(1, (p[0] - LEFT[0]) / (RIGHT[0] - LEFT[0]))))
            line(p, q, color, 28)
            circle(p, 14, color)
    for p in (LEFT, TOP, RIGHT):
        color = mix(MINT, BLUE, (p[0] - LEFT[0]) / (RIGHT[0] - LEFT[0]))
        circle(p, 24, color)
        circle(p, 16.5, (16, 33, 52))
    circle(HUB, 25, (16, 33, 52))
    circle(HUB, 17, AMBER)
    line((250, 282), (262, 282), (101, 82, 49), 4)

    if phase is not None:
        signals = []
        if .10 < phase < .80:
            travel = min(1, (phase - .10) / .64)
            p = interpolate(LEFT, TOP, travel * 2) if travel < .5 else interpolate(TOP, RIGHT, travel * 2 - 1)
            alpha = min(1, (phase - .10) / .05, (.80 - phase) / .06)
            signals.append((p, alpha, (215, 255, 244)))
        if .40 < phase < .75:
            p = interpolate(BRIDGE_LEFT, BRIDGE_RIGHT, min(1, (phase - .40) / .30))
            alpha = min(1, (phase - .40) / .05, (.75 - phase) / .05)
            signals.append((p, alpha, (255, 239, 203)))
        for p, alpha, color in signals:
            glow = Image.new("RGBA", image.size)
            g = ImageDraw.Draw(glow)
            x, y = (v * s for v in p)
            r = 14 * s
            g.ellipse((x-r, y-r, x+r, y+r), fill=(*color, round(160 * alpha)))
            image = Image.alpha_composite(image, glow.filter(ImageFilter.GaussianBlur(8 * s)))
            dot = Image.new("RGBA", image.size)
            ImageDraw.Draw(dot).ellipse((x-6*s, y-6*s, x+6*s, y+6*s), fill=(*color, round(255 * alpha)))
            image = Image.alpha_composite(image, dot)
    return image.resize((size, size), Image.Resampling.LANCZOS)


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "archrivet-icon.svg").write_text(svg(), encoding="utf-8")
    (output / "archrivet-icon-animated.svg").write_text(svg(animated=True), encoding="utf-8")
    for size in (128, 512):
        render(size=size).save(output / f"archrivet-icon-{size}.png", optimize=True)
    # A fixed palette prevents frame-to-frame color flicker. Index 255 is transparent.
    rgba = [render(i / 60, 256) for i in range(60)]
    sheet = Image.new("RGB", (256 * 6, 256))
    for j, i in enumerate((0, 12, 24, 32, 40, 46)):
        sheet.paste(rgba[i].convert("RGB"), (j * 256, 0))
    palette = sheet.quantize(colors=255)
    frames = []
    for frame in rgba:
        indexed = frame.convert("RGB").quantize(palette=palette, dither=Image.Dither.NONE)
        indexed.paste(255, mask=frame.getchannel("A").point(lambda a: 255 if a < 128 else 0))
        frames.append(indexed)
    frames[0].save(output / "archrivet-icon.gif", save_all=True, append_images=frames[1:],
                   duration=80, loop=0, transparency=255, disposal=1, optimize=False)
    # Visual review of multiple animation moments and small-size readability.
    contact = Image.new("RGB", (840, 390), "#f1f4f8")
    for i, phase in enumerate((None, .25, .45, .65)):
        mark = render(phase, 192)
        contact.paste(mark, (12 + i * 208, 12), mark)
    for i, size in enumerate((24, 32, 48, 64, 96, 128)):
        mark = render(size=size)
        contact.paste(mark, (24 + i * 132, 240), mark)
    contact.save(output / "archrivet-icon-preview.png")
    print(f"Created SVG, animated SVG/GIF and PNG icons in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets")
    build(parser.parse_args().output)
