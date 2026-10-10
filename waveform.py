"""Source waveform rendering and original-image coordinate/time conversion."""
import math

import numpy as np
from PIL import Image, ImageDraw


def draw_waveform(audio, rate, start, end, scope, marker=None):
    duration = len(audio) / rate
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
        raise ValueError("開始・終了秒が元音声の範囲外です。")
    low, high = (0.0, duration) if scope == "元音声全体" else (
        max(0.0, start - 2.0), min(duration, end + 2.0)
    )
    width, height, left, right = 1200, 240, 24, 1176
    top, bottom, center = 24, 192, 108
    image = Image.new("RGB", (width, height), "#15191f")
    draw = ImageDraw.Draw(image)
    def x_at(t):
        return left + (t - low) / (high - low) * (right - left)
    draw.rectangle((x_at(start), top, x_at(end), bottom), fill="#203d36")
    draw.line((left, center, right, center), fill="#58616c")
    data = np.asarray(audio)
    if data.ndim == 2:
        data = data.mean(axis=1)
    first, last = int(low * rate), min(len(data), math.ceil(high * rate))
    data = data[first:last]
    scale = max(float(np.max(np.abs(data))), 1e-6)
    edges = np.linspace(0, len(data), right-left+1, dtype=int)
    for i, (a, b) in enumerate(zip(edges, edges[1:])):
        if b > a:
            chunk = data[a:b]
            y1 = center - float(np.max(chunk)) / scale * 76
            y2 = center - float(np.min(chunk)) / scale * 76
            draw.line((left+i, y1, left+i, y2), fill="#85bafa")
    for t, color, label in [(start, "#68df9d", "START"), (end, "#f9c166", "END")]:
        x = x_at(t)
        draw.line((x, top, x, bottom), fill=color, width=2)
        draw.text((min(max(x-30, left), right-110), 6), f"{label} {t:.3f}s", fill=color)
    if marker is not None and low <= marker <= high:
        x = x_at(marker)
        draw.line((x, top, x, bottom), fill="#ff697d", width=2)
    for i in range(9):
        x = left + i * (right-left) / 8
        t = low + i * (high-low) / 8
        draw.line((x, bottom, x, bottom+5), fill="#9ba5af")
        draw.text((min(max(x-22, 0), width-65), bottom+12), f"{t:.2f}s", fill="#dae0e6")
    return image, dict(low=low, high=high, left=left, right=right, duration=duration)


def time_at_pixel(info, x):
    if not info["left"] <= x <= info["right"]:
        return None
    return info["low"] + (x-info["left"]) / (info["right"]-info["left"]) * (
        info["high"]-info["low"]
    )
