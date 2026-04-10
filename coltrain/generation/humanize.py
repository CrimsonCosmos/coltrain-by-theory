"""Jazz MIDI humanization — timing, velocity, and duration micro-variations."""

import math
import random
from dataclasses import replace
from typing import List, Tuple

from coltrain.generation import NoteEvent, TICKS_PER_QUARTER, TICKS_PER_BAR

# ---------------------------------------------------------------------------
# Per-instrument groove profiles
# ---------------------------------------------------------------------------

GROOVE_PROFILES = {
    "drums":  {"timing_ms": 8.0,  "velocity_var": 12, "anticipate_ms": 0},
    "bass":   {"timing_ms": 15.0, "velocity_var": 14, "anticipate_ms": -8},
    "piano":  {"timing_ms": 18.0, "velocity_var": 16, "anticipate_ms": 5},
    "melody": {"timing_ms": 22.0, "velocity_var": 18, "anticipate_ms": 8},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ms_to_ticks(ms: float, tempo: float) -> int:
    """Convert milliseconds to ticks at the given BPM."""
    return round(ms / 1000.0 * (tempo / 60.0) * TICKS_PER_QUARTER)


def _find_phrases(
    notes: List[NoteEvent], gap_ticks: int = TICKS_PER_QUARTER
) -> List[Tuple[int, int]]:
    """Group notes into phrases by detecting gaps in start_tick.

    Returns list of (start_idx, end_idx) tuples — end_idx is exclusive.
    """
    if not notes:
        return []
    phrases = []
    phrase_start = 0
    for i in range(1, len(notes)):
        if notes[i].start_tick - notes[i - 1].start_tick > gap_ticks:
            phrases.append((phrase_start, i))
            phrase_start = i
    phrases.append((phrase_start, len(notes)))
    return phrases

# ---------------------------------------------------------------------------
# Main humanization
# ---------------------------------------------------------------------------


def humanize_track(
    notes: List[NoteEvent],
    instrument: str,
    tempo: float = 140,
    intensity: float = 0.5,
) -> List[NoteEvent]:
    """Apply humanization to a list of NoteEvents, returning new copies.

    Args:
        notes: source note events (not mutated).
        instrument: key into GROOVE_PROFILES.
        tempo: BPM, used to convert ms offsets to ticks.
        intensity: 0.0 (subtle) to 1.0 (full), scales timing offsets.

    Returns:
        New list of humanized NoteEvent objects, sorted by start_tick.
    """
    if instrument not in GROOVE_PROFILES:
        return [replace(n) for n in notes]
    if not notes:
        return []

    profile = GROOVE_PROFILES[instrument]
    timing_ms = profile["timing_ms"]
    velocity_var = profile["velocity_var"]
    anticipate_ms = profile["anticipate_ms"]

    # Work on copies, sorted by start_tick
    out = [replace(n) for n in sorted(notes, key=lambda n: n.start_tick)]

    # Intensity scaling factor for timing offsets
    scale = 0.3 + intensity * 0.7

    # --- (a) Timing humanization: Gaussian offset per note ---
    for i, note in enumerate(out):
        offset_ms = random.gauss(anticipate_ms, timing_ms)
        offset_ticks = _ms_to_ticks(offset_ms * scale, tempo)
        out[i] = replace(note, start_tick=max(0, note.start_tick + offset_ticks))

    # --- (b) & (e) Phrase-position velocity contour and timing bias ---
    phrases = _find_phrases(out)
    for pstart, pend in phrases:
        phrase_len = pend - pstart
        if phrase_len < 2:
            continue
        for j in range(pstart, pend):
            # Normalized position within phrase: 0.0 -> 1.0
            pos = (j - pstart) / (phrase_len - 1)

            # (b) Sine-based velocity arc — peak at phrase center
            vel_boost = math.sin(pos * math.pi) * velocity_var
            new_vel = out[j].velocity + round(vel_boost)

            # (e) Phrase-position timing bias: early at start, late at end
            # Linearly interpolate from -0.5 to +0.5 of anticipate_ms
            bias_ms = anticipate_ms * 0.5 * (2.0 * pos - 1.0)
            bias_ticks = _ms_to_ticks(bias_ms * scale, tempo)
            new_start = max(0, out[j].start_tick + bias_ticks)

            out[j] = replace(out[j], velocity=new_vel, start_tick=new_start)

    # --- (c) Velocity randomization ---
    for i, note in enumerate(out):
        jitter = round(random.gauss(0, velocity_var * 0.3))
        clamped = max(1, min(127, note.velocity + jitter))
        out[i] = replace(note, velocity=clamped)

    # --- (d) Duration humanization: slight variation around original ---
    for i, note in enumerate(out):
        factor = random.uniform(0.85, 1.05)
        new_dur = max(30, round(note.duration_ticks * factor))
        out[i] = replace(note, duration_ticks=new_dur)

    # --- (f) Preserve ordering ---
    out.sort(key=lambda n: n.start_tick)

    return out
