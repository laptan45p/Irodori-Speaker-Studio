"""Sentence/gap based segmentation; duration is a preference, never a cut rule."""

import math
import re
from dataclasses import dataclass


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float = 1.0


@dataclass
class Span:
    start: float
    end: float
    text: str
    confidence: float


def segment_words(words, minimum=5.0, maximum=10.0, pause=0.65):
    if not 0 < minimum <= maximum:
        raise ValueError("秒数は 0 < 最小 <= 最大 にしてください。")
    clean = [w for w in words if w.text.strip() and w.end > w.start]
    if not clean:
        return []
    if any(w.start < 0 or not math.isfinite(w.end) or not math.isfinite(w.start) for w in clean):
        raise ValueError("Invalid timestamp")
    if any(a.end > b.start + 0.05 for a, b in zip(clean, clean[1:])):
        raise ValueError("Overlapping ASR timestamps")
    blocks, block = [], []
    for w in clean:
        if block and w.start - block[-1].end > 2.0:
            blocks.append(block)
            block = []
        block.append(w)
    if block:
        blocks.append(block)
    result = []
    terminal = re.compile(r'[。！？!?．.]\s*[」』”"）)]*\s*$')
    for block in blocks:
        has_sentences = any(terminal.search(w.text) for w in block)
        units, pending = [], []
        for k, w in enumerate(block):
            pending.append(w)
            gap = block[k + 1].start - w.end if k + 1 < len(block) else 0
            if terminal.search(w.text) or (not has_sentences and gap >= pause):
                units.append(pending)
                pending = []
        if pending:
            units.append(pending)
        n = len(units)
        cost, back = [float("inf")] * (n + 1), [None] * (n + 1)
        cost[0] = 0.0
        for j in range(1, n + 1):
            for i in range(j - 1, -1, -1):
                duration = units[j - 1][-1].end - units[i][0].start
                if duration > maximum and j - i > 1:
                    continue
                score = (
                    1 + max(0.0, minimum - duration) ** 2 + max(0.0, duration - maximum) ** 2 * 0.1
                )
                if cost[i] + score < cost[j]:
                    cost[j], back[j] = cost[i] + score, i
        ranges, j = [], n
        while j:
            i = back[j]
            ranges.append((i, j))
            j = i
        for i, j in reversed(ranges):
            chunk = [w for unit in units[i:j] for w in unit]
            result.append(
                Span(
                    chunk[0].start,
                    chunk[-1].end,
                    "".join(w.text for w in chunk).strip(),
                    sum(w.probability for w in chunk) / len(chunk),
                )
            )
    return result


def padded_bounds(spans, duration, padding=0.12):
    result = []
    for i, s in enumerate(spans):
        left = (spans[i - 1].end + s.start) / 2 if i else 0
        right = (s.end + spans[i + 1].start) / 2 if i + 1 < len(spans) else duration
        result.append((max(left, s.start - padding, 0), min(right, s.end + padding, duration)))
    return result
