"""Sentence/gap based segmentation; duration is a preference, never a cut rule."""

import math
import re
from dataclasses import dataclass, replace


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
    note: str = ""


def segment_words(words, minimum=5.0, maximum=10.0, pause=0.65):
    if not 0 < minimum <= maximum:
        raise ValueError("秒数は 0 < 最小 <= 最大 にしてください。")
    # Whisper can emit standalone punctuation with zero duration. Preserve it
    # on the preceding word so sentence boundaries are not lost.
    clean = []
    for word in words:
        if not word.text.strip():
            continue
        if not math.isfinite(word.start) or not math.isfinite(word.end) or word.start < 0:
            raise ValueError("Invalid timestamp")
        if word.end == word.start and clean and re.fullmatch(r'[。！？!?．.,、」』”"）)]+', word.text.strip()):
            clean[-1] = replace(clean[-1], text=clean[-1].text + word.text)
        elif word.end > word.start:
            clean.append(word)
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
    japanese_end = re.compile(
        r'(?:です|ます|でした|ました|ません|ましょう|ください|でしょう|だよ|だね|だな|[たるい]よ|[たるい]ね)[よね]*[」』”"）)]*\s*$'
    )
    continuation = {"が", "けど", "けれど", "けれども", "ので", "のに", "から", "し", "と", "なら"}
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
        # Missing punctuation often turns an entire quick introduction into a
        # single sentence. Only revisit unusually long units; normal short clips
        # keep their original segmentation. A duration is never itself a cut.
        repaired, inferred_ends = [], set()
        for unit in units:
            if unit[-1].end - unit[0].start <= max(20.0, maximum * 2):
                repaired.append(unit)
                continue
            part = []
            for k, word in enumerate(unit):
                part.append(word)
                if k + 1 < len(unit):
                    next_word = unit[k + 1]
                    gap = next_word.start - word.end
                    text = "".join(item.text for item in part)
                    if (gap >= 0.2 and japanese_end.search(text)
                            and next_word.text.strip() not in continuation):
                        repaired.append(part)
                        inferred_ends.add(word.end)
                        part = []
            if part:
                repaired.append(part)
        units = repaired
        n = len(units)
        cost, back = [float("inf")] * (n + 1), [None] * (n + 1)
        cost[0] = 0.0
        for j in range(1, n + 1):
            for i in range(j - 1, -1, -1):
                duration = units[j - 1][-1].end - units[i][0].start
                if duration > maximum and j - i > 1:
                    continue
                # A clear pause after a sentence keeps independent short
                # utterances separate, even if they are below the preferred duration.
                if any(units[k + 1][0].start - units[k][-1].end >= pause
                       for k in range(i, j - 1)):
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
                    "句読点不足・文末表現と間から分割（要試聴）"
                    if chunk[-1].end in inferred_ends else "",
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


def acoustic_bounds(spans, audio, sample_rate, padding=0.12, search=0.4, end_search=0.8):
    """Refine ASR boundaries using nearby quiet frames.

    RMS is a conservative boundary aid, not a semantic or speech detector.
    Keep ASR ranges except a clearly silent leading region. Never move beyond
    the midpoint to adjacent material when extending a clip.
    Fall back to padded bounds when no quiet frame can be identified.
    """
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate <= 0 or audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("Invalid mono audio")
    duration = len(audio) / sample_rate
    bounds = padded_bounds(spans, duration, padding)
    if not len(audio) or not spans:
        return bounds
    frame = max(1, round(sample_rate * 0.01))
    count = (len(audio) + frame - 1) // frame
    squared = np.pad(audio.astype(np.float64) ** 2, (0, count * frame - len(audio)))
    rms = np.sqrt(squared.reshape(count, frame).mean(axis=1))
    # A relative floor works for recordings with different overall volume.
    # Require at least 30 ms of quiet to avoid zero crossings and tiny dips.
    level = float(np.percentile(rms, 90))
    threshold = max(1e-6, level * 0.08)
    def quiet_runs(floor, minimum):
        quiet = rms <= floor
        runs, start = [], None
        for i, is_quiet in enumerate(quiet):
            if is_quiet and start is None:
                start = i
            if start is not None and (not is_quiet or i == len(quiet) - 1):
                end = i if not is_quiet else i + 1
                if (end - start) * frame / sample_rate >= minimum:
                    runs.append((start * frame / sample_rate,
                                 min(duration, end * frame / sample_rate)))
                start = None
        return runs

    runs = quiet_runs(threshold, 0.03)
    # Fading vowels and short inter-syllable dips are not reliable word ends.
    # End boundaries need a quieter, sustained gap and retain trailing padding.
    end_runs = quiet_runs(max(1e-6, level * 0.03), 0.10)

    def quiet_point(lo, hi, preferred, candidates_from, trailing=False):
        candidates = []
        for left, right in candidates_from:
            left, right = max(lo, left), min(hi, right)
            if right - left >= 0.02:
                target = left + padding if trailing else preferred
                candidates.append(min(max(target, left + 0.01), right - 0.01))
        return min(candidates, key=lambda t: abs(t - preferred)) if candidates else None

    result = []
    for i, (span, (start, end)) in enumerate(zip(spans, bounds)):
        left_limit = (spans[i - 1].end + span.start) / 2 if i else 0
        right_limit = (span.end + spans[i + 1].start) / 2 if i + 1 < len(spans) else duration
        left = quiet_point(max(left_limit, span.start - search), span.start, start, runs)
        right = quiet_point(span.end, min(right_limit, span.end + end_search), end, end_runs, True)
        # ASR may timestamp a word well before its audible onset. Move forward
        # only through a continuous, very quiet run; preserve 120 ms pre-roll.
        # A faint onset above the lower end threshold prevents this adjustment.
        for quiet_start, quiet_end in end_runs:
            if (quiet_start <= span.start and quiet_end - span.start >= 0.15
                    and quiet_end <= min(span.end, span.start + 0.8)):
                left = max(start, quiet_end - padding)
                break
        result.append((start if left is None else left, end if right is None else right))
    return result
