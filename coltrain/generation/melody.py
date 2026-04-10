"""Melody and solo generator for Coltrain rule-based jazz generation.

Provides three main generation functions:
  - generate_head_melody: Composed head melody (simple, thematic)
  - generate_solo: Improvised solo with tension curve (melodic -> motivic -> sheets of sound)
  - generate_trading_fours: Trading fours with drums (4 bars melody, 4 bars silence)
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from coltrain.generation import NoteEvent, TICKS_PER_QUARTER, TICKS_PER_8TH, TICKS_PER_16TH, TICKS_PER_BAR
from coltrain.theory.chord import ChordEvent, CHORD_TONES, TENSIONS
from coltrain.theory.scale import SCALES, CHORD_SCALE_MAP, get_scale_notes_midi

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base register (at tension = 0.0)
MELODY_LOW_BASE = 60    # C4
MELODY_HIGH_BASE = 78   # F#5

# Extreme register (at tension = 1.0)
MELODY_LOW_EXTREME = 48  # C3
MELODY_HIGH_EXTREME = 91 # G6

# Legacy aliases (kept for backward compatibility with other modules)
MELODY_LOW = MELODY_LOW_EXTREME
MELODY_HIGH = MELODY_HIGH_EXTREME

# ---------------------------------------------------------------------------
# Digital patterns (Coltrane-style interval patterns from root)
# ---------------------------------------------------------------------------

DIGITAL_PATTERNS = [
    [0, 2, 4, 7],     # 1-2-3-5 ascending
    [0, 2, 4, 9],     # 1-2-3-6
    [7, 4, 2, 0],     # 5-3-2-1 descending
    [0, 4, 7, 12],    # triad arpeggio up
    [12, 7, 4, 0],    # triad arpeggio down
    [0, 4, 7, 11],    # maj7 arpeggio
    [0, 3, 7, 10],    # min7 arpeggio
    [0, 2, 4, 7, 9],  # major pentatonic
]

# ---------------------------------------------------------------------------
# Rhythmic cells -- pre-composed duration sequences with syncopation
# ---------------------------------------------------------------------------

RHYTHMIC_CELLS_SPARSE = [
    # Total: ~2 beats each. Suited for tier 1 / head.
    (2.0,),
    (1.5, 0.5),
    (0.5, 1.5),
    (1.0, 1.0),
    (1.0, 0.5, 0.5),
    (0.5, 0.5, 1.0),
    (0.5, 1.0, 0.5),
]

RHYTHMIC_CELLS_MEDIUM = [
    # Total: ~2 beats each. Suited for tier 2.
    (0.75, 0.25, 1.0),
    (0.5, 0.5, 0.5, 0.5),
    (0.25, 0.75, 0.5, 0.5),
    (0.5, 0.75, 0.75),
    (1.0, 0.25, 0.25, 0.5),
    (0.5, 0.25, 0.25, 0.5, 0.5),
    (0.75, 0.25, 0.5, 0.5),
]

RHYTHMIC_CELLS_DENSE = [
    # Total: ~1 beat each. Suited for tier 3.
    (0.25, 0.25, 0.25, 0.25),
    (0.25, 0.25, 0.5),
    (0.5, 0.25, 0.25),
    (0.75, 0.25),
    (0.25, 0.75),
]

RHYTHMIC_CELLS_TRIPLET = [
    # Total: 1 beat each. Triplet groupings.
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    (2.0 / 3.0, 1.0 / 3.0),
    (1.0 / 3.0, 2.0 / 3.0),
]

# ---------------------------------------------------------------------------
# Motif system for motivic development
# ---------------------------------------------------------------------------


@dataclass
class Motif:
    """A short melodic idea defined by intervals and durations."""
    intervals: list   # relative semitone intervals from first note
    durations: list   # beat durations for each note

    def invert(self) -> "Motif":
        return Motif([-iv for iv in self.intervals], list(self.durations))

    def retrograde(self) -> "Motif":
        return Motif(list(reversed(self.intervals)), list(reversed(self.durations)))

    def augment(self) -> "Motif":
        return Motif(list(self.intervals), [d * 2 for d in self.durations])

    def diminish(self) -> "Motif":
        return Motif(list(self.intervals), [d * 0.5 for d in self.durations])

    def transpose(self, semitones: int) -> "Motif":
        return Motif([iv + semitones for iv in self.intervals], list(self.durations))


SEED_MOTIFS = [
    # Original motifs
    Motif([0, 2, 4, 7], [0.5, 0.5, 0.5, 0.5]),
    Motif([0, -1, -3, -5], [0.5, 0.5, 0.5, 0.5]),
    Motif([0, 4, 7, 4], [0.5, 0.5, 0.5, 0.5]),
    Motif([0, 7, 5, 3, 0], [0.25, 0.25, 0.25, 0.25, 1.0]),
    # Syncopated motifs with dotted rhythms
    Motif([0, 2, 5, 7], [0.75, 0.25, 0.5, 0.5]),
    Motif([0, -2, -3, 0, 4], [0.5, 0.25, 0.25, 0.5, 0.5]),
    Motif([0, 7, 12, 7], [0.25, 0.25, 1.0, 0.5]),
    Motif([0, 0, -1, 2, 4], [0.25, 0.25, 0.25, 0.25, 1.0]),
    Motif([0, 5, 4, 2], [1.0, 0.5, 0.75, 0.25]),
    Motif([0, -5, -3, -1, 0], [0.25, 0.25, 0.25, 0.25, 0.5]),
]

# ---------------------------------------------------------------------------
# Solo state -- carries context across phrase boundaries
# ---------------------------------------------------------------------------


@dataclass
class _SoloState:
    """Mutable state carried across phrase boundaries during solo generation."""
    pitch: int
    recent_pitches: List[int] = field(default_factory=list)
    active_motif: Optional[Motif] = None
    motif_play_count: int = 0
    motif_max_plays: int = 3
    phrase_count: int = 0
    last_phrase_length: float = 0.0

    def advance_motif(self, tension: float) -> Motif:
        """Get next motif incarnation with successive transformations."""
        if self.active_motif is None or self.motif_play_count >= self.motif_max_plays:
            self.active_motif = random.choice(SEED_MOTIFS)
            self.motif_play_count = 0
            self.motif_max_plays = random.randint(2, 4)

        count = self.motif_play_count
        self.motif_play_count += 1

        m = self.active_motif
        if count == 0:
            return m
        elif count == 1:
            return m.invert()
        elif count == 2:
            shift = random.randint(2, 5)
            return m.transpose(shift)
        else:
            return m.diminish()

    def record_pitch(self, p: int):
        self.recent_pitches.append(p)
        if len(self.recent_pitches) > 20:
            self.recent_pitches = self.recent_pitches[-20:]


# ---------------------------------------------------------------------------
# Tension curve
# ---------------------------------------------------------------------------


class TensionCurve:
    """Maps position (0.0-1.0) to tension (0.0-1.0)."""

    CURVES = {
        "arc": lambda x: (
            math.sin(x / 0.75 * math.pi / 2)
            if x < 0.75
            else math.cos((x - 0.75) / 0.25 * math.pi / 2)
        ),
        "build": lambda x: x ** 0.6,
        "wave": lambda x: 0.5 * x + 0.5 * math.sin(x * 3 * math.pi) * 0.3 + 0.3,
        "plateau": lambda x: (
            min(0.75, x / 0.3 * 0.75) if x < 0.3
            else 0.75 if x < 0.8
            else 0.75 * (1.0 - (x - 0.8) / 0.2)
        ),
        "catharsis": lambda x: (
            0.2 + x / 0.6 * 0.3 if x < 0.6
            else 0.5 + (x - 0.6) / 0.15 * 0.5 if x < 0.75
            else 1.0 - (x - 0.75) / 0.25 * 0.9
        ),
    }

    def __init__(self, curve_name: str = "arc"):
        self.fn = self.CURVES.get(curve_name, self.CURVES["arc"])

    def __call__(self, progress: float) -> float:
        return max(0.0, min(1.0, self.fn(max(0.0, min(1.0, progress)))))


# ---------------------------------------------------------------------------
# MusicParams: interpolated from tension
# ---------------------------------------------------------------------------


@dataclass
class MusicParams:
    """Performance parameters derived from tension level."""
    note_density: float
    chromatic_prob: float
    rest_prob: float
    velocity_base: int
    velocity_range: int
    register_low: int
    register_high: int
    motif_complexity: float


def interpolate_params(tension: float) -> MusicParams:
    """Interpolate musical parameters from a tension value in [0, 1]."""
    t = max(0.0, min(1.0, tension))
    return MusicParams(
        note_density=1.0 + t * 3.0,
        chromatic_prob=t * 0.4,
        rest_prob=0.20 - t * 0.17,
        velocity_base=int(65 + t * 35),
        velocity_range=int(5 + t * 15),
        register_low=int(MELODY_LOW_BASE - t * (MELODY_LOW_BASE - MELODY_LOW_EXTREME)),
        register_high=int(MELODY_HIGH_BASE + t * (MELODY_HIGH_EXTREME - MELODY_HIGH_BASE)),
        motif_complexity=t,
    )


# ---------------------------------------------------------------------------
# Helper functions -- pitch
# ---------------------------------------------------------------------------


def _chord_tones_in_range(root_pc: int, quality: str, low: int, high: int) -> List[int]:
    intervals = CHORD_TONES.get(quality, (0, 4, 7))
    result = []
    for midi_note in range(low, high + 1):
        interval = (midi_note % 12 - root_pc) % 12
        if interval in intervals:
            result.append(midi_note)
    return result


def _scale_tones_in_range(root_pc: int, quality: str, low: int, high: int) -> List[int]:
    scale_names = CHORD_SCALE_MAP.get(quality, ["ionian"])
    scale_name = scale_names[0]
    intervals = SCALES.get(scale_name, SCALES["ionian"])
    interval_set = set(intervals)
    result = []
    for midi_note in range(low, high + 1):
        interval = (midi_note % 12 - root_pc) % 12
        if interval in interval_set:
            result.append(midi_note)
    return result


def _nearest_chord_tone(current_midi: int, root_pc: int, quality: str,
                        low: int, high: int) -> int:
    tones = _chord_tones_in_range(root_pc, quality, low, high)
    if not tones:
        return max(low, min(high, current_midi))
    return min(tones, key=lambda t: abs(t - current_midi))


def _nearest_scale_tone(current_midi: int, root_pc: int, quality: str,
                        low: int, high: int) -> int:
    tones = _scale_tones_in_range(root_pc, quality, low, high)
    if not tones:
        return max(low, min(high, current_midi))
    return min(tones, key=lambda t: abs(t - current_midi))


def _guide_tones_in_range(root_pc: int, quality: str, low: int, high: int) -> List[int]:
    """Return 3rds and 7ths of the chord within [low, high]."""
    intervals = CHORD_TONES.get(quality, (0, 4, 7))
    guide_intervals = set()
    if len(intervals) > 1:
        guide_intervals.add(intervals[1])
    if len(intervals) > 3:
        guide_intervals.add(intervals[3])
    result = []
    for midi_note in range(low, high + 1):
        interval = (midi_note % 12 - root_pc) % 12
        if interval in guide_intervals:
            result.append(midi_note)
    return result


def _extensions_in_range(root_pc: int, quality: str, low: int, high: int) -> List[int]:
    tension_intervals = TENSIONS.get(quality, [])
    result = []
    for midi_note in range(low, high + 1):
        interval = (midi_note % 12 - root_pc) % 12
        for t in tension_intervals:
            if interval == t % 12:
                result.append(midi_note)
                break
    return result


def _substitute_key_tones(key_center_pc: int, low: int, high: int) -> List[int]:
    sub_keys = [(key_center_pc + 4) % 12, (key_center_pc + 8) % 12]
    result = []
    for sub_key in sub_keys:
        for interval in (0, 4, 7):
            pc = (sub_key + interval) % 12
            for midi_note in range(low, high + 1):
                if midi_note % 12 == pc and midi_note not in result:
                    result.append(midi_note)
    result.sort()
    return result


def _choose_target_pitch(current_pitch: int, root_pc: int, quality: str,
                         low: int, high: int, tension: float,
                         beat_in_bar: float) -> int:
    """Choose a target pitch with intervallic variety based on tension."""
    is_strong_beat = beat_in_bar < 0.1 or abs(beat_in_bar - 2.0) < 0.1
    chord_tones = _chord_tones_in_range(root_pc, quality, low, high)
    guide_tones = _guide_tones_in_range(root_pc, quality, low, high)

    if not chord_tones:
        return max(low, min(high, current_pitch))

    roll = random.random()

    # Octave displacement (5% at low tension, 20% at high tension)
    octave_prob = 0.05 + tension * 0.15
    if roll < octave_prob:
        direction = random.choice([-12, 12])
        candidate = current_pitch + direction
        if low <= candidate <= high:
            return min(chord_tones, key=lambda t: abs(t - candidate))

    # Strong beat: guide tone targeting
    if is_strong_beat and guide_tones and roll < 0.7:
        sorted_guides = sorted(guide_tones, key=lambda t: abs(t - current_pitch))
        if len(sorted_guides) >= 2 and random.random() < 0.3:
            return sorted_guides[1]
        return sorted_guides[0]

    # Leap-based selection (probability scales with tension)
    leap_prob = 0.05 + tension * 0.35
    if roll < leap_prob + octave_prob:
        leap_size = random.choice([5, 7, 8, 9, 10, 12])
        direction = random.choice([-1, 1])
        target_area = current_pitch + direction * leap_size
        target_area = max(low, min(high, target_area))
        return min(chord_tones, key=lambda t: abs(t - target_area))

    # Stepwise / nearby motion (default)
    scale_tones = _scale_tones_in_range(root_pc, quality, low, high)
    nearby = [t for t in set(chord_tones + scale_tones)
              if abs(t - current_pitch) <= 5 and t != current_pitch]
    if nearby:
        nearby_chord = [t for t in nearby if t in chord_tones]
        if nearby_chord and random.random() < 0.6:
            return random.choice(nearby_chord)
        return random.choice(nearby)

    return min(chord_tones, key=lambda t: abs(t - current_pitch))


# ---------------------------------------------------------------------------
# Helper functions -- harmony and anticipation
# ---------------------------------------------------------------------------


def _get_chord_at_beat(chords: List[ChordEvent], beat: float) -> Optional[ChordEvent]:
    for chord in chords:
        if chord.start_beat <= beat < chord.end_beat:
            return chord
    if chords:
        return chords[-1]
    return None


def _get_next_chord(chords: List[ChordEvent], beat: float) -> Optional[ChordEvent]:
    """Return the next chord AFTER the current beat position."""
    for chord in chords:
        if chord.start_beat > beat + 1e-9:
            return chord
    return None


def _beats_until_chord_change(chords: List[ChordEvent], beat: float) -> float:
    """Return how many beats until the next chord change."""
    current = _get_chord_at_beat(chords, beat)
    if current is None:
        return float('inf')
    return current.end_beat - beat


def _approach_next_chord(current_pitch: int, chords: List[ChordEvent],
                         beat: float, low: int, high: int) -> Optional[int]:
    """If close to a chord change, return a chromatic approach note
    targeting the next chord's 3rd or 7th."""
    beats_left = _beats_until_chord_change(chords, beat)
    if beats_left > 1.0:
        return None
    next_chord = _get_next_chord(chords, beat)
    if next_chord is None:
        return None
    next_guides = _guide_tones_in_range(next_chord.root_pc, next_chord.quality, low, high)
    if not next_guides:
        return None
    target_guide = min(next_guides, key=lambda t: abs(t - current_pitch))
    if current_pitch >= target_guide:
        approach = target_guide + 1
    else:
        approach = target_guide - 1
    if low <= approach <= high:
        return approach
    return None


# ---------------------------------------------------------------------------
# Helper functions -- timing and rhythm
# ---------------------------------------------------------------------------


def _beat_to_tick(beat: float) -> int:
    return int(beat * TICKS_PER_QUARTER)


def _apply_swing(tick: int, swing_ratio: float = 0.667) -> int:
    beat_pos = tick % TICKS_PER_QUARTER
    beat_start = tick - beat_pos
    if beat_pos >= TICKS_PER_8TH:
        swing_point = int(TICKS_PER_QUARTER * swing_ratio)
        offset_within_offbeat = beat_pos - TICKS_PER_8TH
        remaining = TICKS_PER_QUARTER - swing_point
        if TICKS_PER_8TH > 0:
            scaled_offset = int(offset_within_offbeat * remaining / TICKS_PER_8TH)
        else:
            scaled_offset = 0
        return beat_start + swing_point + scaled_offset
    else:
        swing_point = int(TICKS_PER_QUARTER * swing_ratio)
        if TICKS_PER_8TH > 0:
            scaled = int(beat_pos * swing_point / TICKS_PER_8TH)
        else:
            scaled = 0
        return beat_start + scaled


def _humanize(tick: int, amount: int = 10) -> int:
    return max(0, tick + random.randint(-amount, amount))


def _choose_rhythmic_cell(tension: float) -> Tuple[float, ...]:
    """Select a rhythmic cell appropriate for the current tension level."""
    roll = random.random()
    if tension < 0.3:
        pool = RHYTHMIC_CELLS_SPARSE if roll > 0.15 else RHYTHMIC_CELLS_MEDIUM
    elif tension < 0.6:
        if roll < 0.10:
            pool = RHYTHMIC_CELLS_SPARSE
        elif roll < 0.20:
            pool = RHYTHMIC_CELLS_DENSE
        else:
            pool = RHYTHMIC_CELLS_MEDIUM
    else:
        if roll < 0.10:
            pool = RHYTHMIC_CELLS_MEDIUM
        elif roll < 0.25:
            pool = RHYTHMIC_CELLS_TRIPLET
        else:
            pool = RHYTHMIC_CELLS_DENSE
    return random.choice(pool)


def _choose_velocity(params: MusicParams) -> int:
    v = params.velocity_base + random.randint(-params.velocity_range, params.velocity_range)
    return max(1, min(127, v))


# ---------------------------------------------------------------------------
# Contour and melodic helpers
# ---------------------------------------------------------------------------


def _contour_check(recent_pitches: List[int], max_monotonic: int = 6) -> bool:
    if len(recent_pitches) < max_monotonic:
        return False
    tail = recent_pitches[-max_monotonic:]
    ascending = all(tail[i] <= tail[i + 1] for i in range(len(tail) - 1))
    descending = all(tail[i] >= tail[i + 1] for i in range(len(tail) - 1))
    return ascending or descending


def _chromatic_enclosure(target_midi: int) -> List[Tuple[int, float]]:
    if random.random() < 0.5:
        return [(target_midi + 1, 0.25), (target_midi - 1, 0.25), (target_midi, 0.5)]
    else:
        return [(target_midi - 1, 0.25), (target_midi + 1, 0.25), (target_midi, 0.5)]


def _scalar_run(current_midi: int, direction: int, root_pc: int, quality: str,
                length: int, low: int, high: int) -> List[Tuple[int, float]]:
    scale_tones = _scale_tones_in_range(root_pc, quality, low, high)
    if not scale_tones:
        return [(current_midi, 0.5)]
    start = min(scale_tones, key=lambda t: abs(t - current_midi))
    idx = scale_tones.index(start)
    result = []
    for i in range(length):
        if 0 <= idx < len(scale_tones):
            result.append((scale_tones[idx], 0.5))
            idx += direction
        else:
            break
    if not result:
        result.append((start, 0.5))
    return result


def _arpeggio_run(current_midi: int, direction: int, root_pc: int, quality: str,
                  low: int, high: int) -> List[Tuple[int, float]]:
    chord_notes = _chord_tones_in_range(root_pc, quality, low, high)
    if not chord_notes:
        return [(current_midi, 0.5)]
    start = min(chord_notes, key=lambda t: abs(t - current_midi))
    idx = chord_notes.index(start)
    result = []
    max_notes = random.randint(3, 6)
    for _ in range(max_notes):
        if 0 <= idx < len(chord_notes):
            result.append((chord_notes[idx], 0.5))
            idx += direction
        else:
            break
    if not result:
        result.append((start, 0.5))
    return result


def _digital_pattern_fragment(root_pc: int, current_midi: int,
                              low: int, high: int) -> List[Tuple[int, float]]:
    pattern = random.choice(DIGITAL_PATTERNS)
    root_candidates = [n for n in range(low, high + 1) if n % 12 == root_pc]
    if not root_candidates:
        return [(current_midi, 0.5)]
    root_midi = min(root_candidates, key=lambda r: abs(r - current_midi))
    result = []
    for interval in pattern:
        note = root_midi + interval
        clamped = max(low, min(high, note))
        result.append((clamped, 0.5))
    return result


def _select_tier(tension: float) -> int:
    """Select generation tier with probability blending near boundaries."""
    if tension < 0.3:
        return 1
    elif tension < 0.5:
        tier2_prob = (tension - 0.3) / 0.2
        return 2 if random.random() < tier2_prob else 1
    elif tension < 0.6:
        return 2
    elif tension < 0.8:
        tier3_prob = (tension - 0.6) / 0.2
        return 3 if random.random() < tier3_prob else 2
    else:
        return 3


# ---------------------------------------------------------------------------
# Tier 1: Melodic -- guide tones, voice leading, syncopated rhythm
# ---------------------------------------------------------------------------


def _generate_tier1_phrase(current_beat: float, phrase_beats: float,
                           chords: List[ChordEvent], params: MusicParams,
                           swing: bool, state: _SoloState,
                           tension: float) -> Tuple[List[NoteEvent], float]:
    notes = []
    beat = current_beat
    phrase_end = current_beat + phrase_beats

    while beat < phrase_end - 1e-9:
        beat_before = beat  # Stall guard

        chord = _get_chord_at_beat(chords, beat)
        if chord is None:
            beat += 1.0
            continue

        low = params.register_low
        high = params.register_high

        # Choose a rhythmic cell
        cell = _choose_rhythmic_cell(tension)

        for dur in cell:
            if beat + dur > phrase_end + 1e-9:
                # Try to fit a truncated note
                remaining = phrase_end - beat
                if remaining < 0.125:
                    break
                dur = remaining

            # Check for harmonic anticipation
            approach = _approach_next_chord(state.pitch, chords, beat, low, high)
            if approach is not None and random.random() < 0.4:
                target = approach
            else:
                target = _choose_target_pitch(
                    state.pitch, chord.root_pc, chord.quality,
                    low, high, tension, beat % 4.0)

            # Contour check
            if _contour_check(state.recent_pitches):
                direction = -1 if state.recent_pitches[-1] > state.recent_pitches[-2] else 1
                candidates = _chord_tones_in_range(chord.root_pc, chord.quality, low, high)
                shifted = [t for t in candidates
                           if (t > state.pitch) == (direction > 0)]
                if shifted:
                    target = shifted[0] if direction > 0 else shifted[-1]

            tick = _beat_to_tick(beat)
            if swing:
                tick = _apply_swing(tick)
            tick = _humanize(tick)
            dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
            vel = _choose_velocity(params)

            notes.append(NoteEvent(
                pitch=target, start_tick=tick,
                duration_ticks=dur_ticks, velocity=vel,
            ))

            state.pitch = target
            state.record_pitch(target)
            beat += dur

            if beat >= phrase_end - 1e-9:
                break

        # Stall guard: if no progress was made, advance to end
        if beat <= beat_before + 1e-9:
            beat = phrase_end

    return notes, beat


# ---------------------------------------------------------------------------
# Tier 2: Motivic Development -- motifs, enclosures, runs, syncopation
# ---------------------------------------------------------------------------


def _generate_tier2_phrase(current_beat: float, phrase_beats: float,
                           chords: List[ChordEvent], params: MusicParams,
                           swing: bool, coltrane: bool,
                           state: _SoloState,
                           tension: float) -> Tuple[List[NoteEvent], float]:
    notes = []
    beat = current_beat
    phrase_end = current_beat + phrase_beats

    # Motif development: get the next transformation of the active motif
    motif = state.advance_motif(tension)

    # Apply motif if complexity allows and it fits
    if params.motif_complexity > 0.2 and beat + sum(motif.durations) <= phrase_end + 1e-9:
        chord = _get_chord_at_beat(chords, beat)
        if chord is not None:
            low = params.register_low
            high = params.register_high
            motif_start = _nearest_chord_tone(state.pitch, chord.root_pc, chord.quality, low, high)

            for iv, dur in zip(motif.intervals, motif.durations):
                if beat + dur > phrase_end + 1e-9:
                    break
                note_midi = max(low, min(high, motif_start + iv))
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick)
                dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                vel = _choose_velocity(params)
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                state.pitch = note_midi
                state.record_pitch(note_midi)
                beat += dur

    # Continue with connective strategies
    strategies = ["enclosure", "scalar_run", "arpeggio", "digital_pattern", "direct"]
    weights = [0.20, 0.30, 0.15, 0.20, 0.15]
    if coltrane:
        weights = [0.15, 0.20, 0.15, 0.35, 0.15]

    while beat < phrase_end - 1e-9:
        beat_before = beat  # Stall guard

        chord = _get_chord_at_beat(chords, beat)
        if chord is None:
            beat += 0.5
            continue

        low = params.register_low
        high = params.register_high

        # Chromatic passing tone insertion
        if random.random() < params.chromatic_prob and notes:
            approach = _approach_next_chord(state.pitch, chords, beat, low, high)
            if approach is not None:
                cdur = 0.25
                if beat + cdur <= phrase_end + 1e-9:
                    tick = _beat_to_tick(beat)
                    if swing:
                        tick = _apply_swing(tick)
                    tick = _humanize(tick)
                    notes.append(NoteEvent(
                        pitch=approach, start_tick=tick,
                        duration_ticks=max(30, int(cdur * TICKS_PER_QUARTER)),
                        velocity=max(1, _choose_velocity(params) - 10),
                    ))
                    state.pitch = approach
                    state.record_pitch(approach)
                    beat += cdur
                    continue

        strategy = random.choices(strategies, weights=weights, k=1)[0]

        if strategy == "enclosure":
            # Target next chord's guide tone if near a change
            approach = _approach_next_chord(state.pitch, chords, beat, low, high)
            if approach is not None:
                target = approach
            else:
                target = _choose_target_pitch(
                    state.pitch, chord.root_pc, chord.quality,
                    low, high, tension, beat % 4.0)
            enc_notes = _chromatic_enclosure(target)
            for note_midi, dur in enc_notes:
                if beat + dur > phrase_end + 1e-9:
                    break
                note_midi = max(low, min(high, note_midi))
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick)
                dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                vel = _choose_velocity(params)
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                state.pitch = note_midi
                state.record_pitch(note_midi)
                beat += dur

        elif strategy == "scalar_run":
            direction = 1 if random.random() < 0.5 else -1
            if _contour_check(state.recent_pitches):
                direction = -1 if state.recent_pitches[-1] > state.recent_pitches[-3] else 1
            length = random.randint(3, 6)
            run = _scalar_run(state.pitch, direction, chord.root_pc, chord.quality, length, low, high)
            for note_midi, dur in run:
                if beat + dur > phrase_end + 1e-9:
                    break
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick)
                dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                vel = _choose_velocity(params)
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                state.pitch = note_midi
                state.record_pitch(note_midi)
                beat += dur

        elif strategy == "arpeggio":
            direction = 1 if random.random() < 0.5 else -1
            if _contour_check(state.recent_pitches):
                direction = -1 if state.recent_pitches[-1] > state.recent_pitches[-3] else 1
            run = _arpeggio_run(state.pitch, direction, chord.root_pc, chord.quality, low, high)
            for note_midi, dur in run:
                if beat + dur > phrase_end + 1e-9:
                    break
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick)
                dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                vel = _choose_velocity(params)
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                state.pitch = note_midi
                state.record_pitch(note_midi)
                beat += dur

        elif strategy == "digital_pattern":
            fragment = _digital_pattern_fragment(chord.root_pc, state.pitch, low, high)
            for note_midi, dur in fragment:
                if beat + dur > phrase_end + 1e-9:
                    break
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick)
                dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                vel = _choose_velocity(params)
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                state.pitch = note_midi
                state.record_pitch(note_midi)
                beat += dur

        elif strategy == "direct":
            target = _choose_target_pitch(
                state.pitch, chord.root_pc, chord.quality,
                low, high, tension, beat % 4.0)
            cell = _choose_rhythmic_cell(tension)
            dur = cell[0]
            if beat + dur > phrase_end + 1e-9:
                dur = phrase_end - beat
                if dur < 0.125:
                    beat = phrase_end
                    break
            tick = _beat_to_tick(beat)
            if swing:
                tick = _apply_swing(tick)
            tick = _humanize(tick)
            dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
            vel = _choose_velocity(params)
            notes.append(NoteEvent(
                pitch=target, start_tick=tick,
                duration_ticks=dur_ticks, velocity=vel,
            ))
            state.pitch = target
            state.record_pitch(target)
            beat += dur

        # Stall guard: if no progress was made, advance to end
        if beat <= beat_before + 1e-9:
            beat = phrase_end

    return notes, beat


# ---------------------------------------------------------------------------
# Tier 3: Sheets of Sound -- rapid notes, register shifts, digital patterns
# ---------------------------------------------------------------------------


def _generate_tier3_phrase(current_beat: float, phrase_beats: float,
                           chords: List[ChordEvent], params: MusicParams,
                           swing: bool, coltrane: bool,
                           state: _SoloState,
                           tension: float) -> Tuple[List[NoteEvent], float]:
    notes = []
    beat = current_beat
    phrase_end = current_beat + phrase_beats

    strategies = ["digital", "rapid_arpeggio", "enclosure_run"]
    weights = [0.40, 0.30, 0.30]
    if coltrane:
        weights = [0.50, 0.30, 0.20]

    vel_base = max(90, params.velocity_base)
    vel_range = params.velocity_range

    while beat < phrase_end - 1e-9:
        # Guard: need at least a 16th note
        if phrase_end - beat < 0.25 - 1e-9:
            break

        beat_before = beat  # Stall guard

        chord = _get_chord_at_beat(chords, beat)
        if chord is None:
            beat += 0.25
            continue

        low = params.register_low
        high = params.register_high

        # Occasional dramatic register shift (15%)
        if random.random() < 0.15:
            mid = (low + high) // 2
            if state.pitch < mid:
                state.pitch = min(high, state.pitch + random.randint(12, 19))
            else:
                state.pitch = max(low, state.pitch - random.randint(12, 19))

        # Choose rhythmic cell for this burst
        cell = _choose_rhythmic_cell(tension)
        strategy = random.choices(strategies, weights=weights, k=1)[0]

        if strategy == "digital":
            fragment = _digital_pattern_fragment(chord.root_pc, state.pitch, low, high)
            for (note_midi, _), dur in zip(fragment, cell):
                if beat + dur > phrase_end + 1e-9:
                    break
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick, amount=5)
                dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                vel = min(127, max(1, vel_base + random.randint(-vel_range, vel_range)))
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                state.pitch = note_midi
                state.record_pitch(note_midi)
                beat += dur

        elif strategy == "rapid_arpeggio":
            chord_notes = _chord_tones_in_range(chord.root_pc, chord.quality, low, high)
            ext_notes = _extensions_in_range(chord.root_pc, chord.quality, low, high)
            all_tones = sorted(set(chord_notes + ext_notes))
            if not all_tones:
                all_tones = [state.pitch]
            start_note = min(all_tones, key=lambda t: abs(t - state.pitch))
            idx = all_tones.index(start_note)
            direction = 1 if random.random() < 0.5 else -1
            if _contour_check(state.recent_pitches, max_monotonic=4):
                direction = -direction
            run_length = random.randint(4, 8)
            for i in range(run_length):
                dur = cell[i % len(cell)]
                if beat + dur > phrase_end + 1e-9:
                    break
                if 0 <= idx < len(all_tones):
                    note_midi = all_tones[idx]
                else:
                    direction = -direction
                    idx = max(0, min(len(all_tones) - 1, idx))
                    note_midi = all_tones[idx]
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick, amount=5)
                dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                vel = min(127, max(1, vel_base + random.randint(-vel_range, vel_range)))
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                state.pitch = note_midi
                state.record_pitch(note_midi)
                beat += dur
                idx += direction

        elif strategy == "enclosure_run":
            chord_notes = _chord_tones_in_range(chord.root_pc, chord.quality, low, high)
            if not chord_notes:
                chord_notes = [state.pitch]
            targets = sorted(chord_notes, key=lambda t: abs(t - state.pitch))[:3]
            random.shuffle(targets)
            for target in targets:
                if beat >= phrase_end - 1e-9:
                    break
                enc_pitches = [target + 1, target - 1, target]
                for j, ep in enumerate(enc_pitches):
                    dur = cell[j % len(cell)] if j < len(cell) else 0.25
                    if beat + dur > phrase_end + 1e-9:
                        break
                    ep_clamped = max(low, min(high, ep))
                    tick = _beat_to_tick(beat)
                    if swing:
                        tick = _apply_swing(tick)
                    tick = _humanize(tick, amount=5)
                    dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                    vel = min(127, max(1, vel_base + random.randint(-vel_range, vel_range)))
                    notes.append(NoteEvent(
                        pitch=ep_clamped, start_tick=tick,
                        duration_ticks=dur_ticks, velocity=vel,
                    ))
                    state.pitch = ep_clamped
                    state.record_pitch(ep_clamped)
                    beat += dur

        # Stall guard: if no progress was made, advance to end
        if beat <= beat_before + 1e-9:
            beat = phrase_end

    return notes, beat


# ---------------------------------------------------------------------------
# Extended phrase generators
# ---------------------------------------------------------------------------


def _generate_pentatonic_super_phrase(current_beat: float, phrase_beats: float,
                                      chords: List[ChordEvent], params: MusicParams,
                                      swing: bool, coltrane: bool,
                                      state: _SoloState,
                                      tension: float) -> Tuple[List[NoteEvent], float]:
    notes = []
    beat = current_beat
    phrase_end = current_beat + phrase_beats

    chord = _get_chord_at_beat(chords, beat)
    if chord is None:
        return notes, beat

    low = params.register_low
    high = params.register_high

    root = chord.root_pc
    q = chord.quality
    if q in ("maj7", "maj", "6"):
        penta_root = (root + 2) % 12
    elif q in ("dom7", "7"):
        penta_root = (root + random.choice([2, 10])) % 12
    elif q in ("min7", "min", "min6"):
        penta_root = (root + 3) % 12
    else:
        penta_root = root

    penta_intervals = (0, 2, 4, 7, 9)
    penta_notes = [m for m in range(low, high + 1)
                   if (m % 12 - penta_root) % 12 in penta_intervals]

    if not penta_notes:
        return _generate_tier2_phrase(current_beat, phrase_beats, chords,
                                      params, swing, coltrane, state, tension)

    start_note = min(penta_notes, key=lambda t: abs(t - state.pitch))
    idx = penta_notes.index(start_note)
    direction = random.choice([-1, 1])

    num_notes = random.randint(6, 10)
    for i in range(num_notes):
        if beat >= phrase_end - 1e-9:
            break
        if 0 <= idx < len(penta_notes):
            note_midi = penta_notes[idx]
        else:
            direction = -direction
            idx = max(0, min(len(penta_notes) - 1, idx))
            note_midi = penta_notes[idx]

        cell = _choose_rhythmic_cell(tension)
        dur = cell[0]
        if beat + dur > phrase_end + 1e-9:
            remaining = phrase_end - beat
            if remaining < 0.125:
                break
            dur = remaining

        tick = _beat_to_tick(beat)
        if swing:
            tick = _apply_swing(tick)
        tick = _humanize(tick)
        dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
        vel = _choose_velocity(params)
        notes.append(NoteEvent(
            pitch=note_midi, start_tick=tick,
            duration_ticks=dur_ticks, velocity=vel,
        ))
        state.pitch = note_midi
        state.record_pitch(note_midi)
        beat += dur
        idx += direction

    return notes, beat


def _generate_call_response_phrase(current_beat: float, phrase_beats: float,
                                    chords: List[ChordEvent], params: MusicParams,
                                    swing: bool, coltrane: bool,
                                    state: _SoloState,
                                    tension: float) -> Tuple[List[NoteEvent], float]:
    notes = []
    beat = current_beat
    phrase_end = current_beat + phrase_beats
    half_beats = phrase_beats / 2.0

    # --- Call ---
    call_notes_data = []
    call_end = beat + half_beats
    call_note_count = random.randint(3, 6)
    call_played = 0

    while beat < call_end - 1e-9 and call_played < call_note_count:
        chord = _get_chord_at_beat(chords, beat)
        if chord is None:
            beat += 0.5
            continue

        low = params.register_low
        high = params.register_high
        target = _choose_target_pitch(
            state.pitch, chord.root_pc, chord.quality,
            low, high, tension, beat % 4.0)

        cell = _choose_rhythmic_cell(tension)
        dur = cell[0]
        if beat + dur > call_end + 1e-9:
            remaining = call_end - beat
            if remaining < 0.125:
                break
            dur = remaining

        tick = _beat_to_tick(beat)
        if swing:
            tick = _apply_swing(tick)
        tick = _humanize(tick)
        dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
        vel = _choose_velocity(params)
        notes.append(NoteEvent(
            pitch=target, start_tick=tick,
            duration_ticks=dur_ticks, velocity=vel,
        ))
        call_notes_data.append((target, dur))
        state.pitch = target
        state.record_pitch(target)
        beat += dur
        call_played += 1

    # --- Response ---
    if len(call_notes_data) < 2:
        return notes, beat

    call_pitches = [p for p, d in call_notes_data]
    call_durs = [d for p, d in call_notes_data]
    call_intervals = [call_pitches[i + 1] - call_pitches[i]
                      for i in range(len(call_pitches) - 1)]

    transform_roll = random.random()
    response_end = min(beat + half_beats, phrase_end)

    if transform_roll < 0.4:
        shift = random.choice([-5, -4, -3, -2, 2, 3, 4, 5])
        resp_pitches = [max(MELODY_LOW, min(MELODY_HIGH, p + shift)) for p in call_pitches]
        resp_durs = list(call_durs)
    elif transform_roll < 0.7:
        inverted_intervals = [-iv for iv in call_intervals]
        resp_pitches = [call_pitches[0]]
        for iv in inverted_intervals:
            resp_pitches.append(max(MELODY_LOW, min(MELODY_HIGH, resp_pitches[-1] + iv)))
        resp_durs = list(call_durs)
    else:
        resp_pitches = list(call_pitches)
        resp_durs = list(call_durs)
        beat += 0.5

    for resp_pitch, dur in zip(resp_pitches, resp_durs):
        if beat >= response_end - 1e-9:
            break
        if beat + dur > response_end + 1e-9:
            remaining = response_end - beat
            if remaining < 0.125:
                break
            dur = remaining
        tick = _beat_to_tick(beat)
        if swing:
            tick = _apply_swing(tick)
        tick = _humanize(tick)
        dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
        vel = _choose_velocity(params)
        notes.append(NoteEvent(
            pitch=resp_pitch, start_tick=tick,
            duration_ticks=dur_ticks, velocity=vel,
        ))
        state.pitch = resp_pitch
        state.record_pitch(resp_pitch)
        beat += dur

    return notes, beat


def _generate_triplet_phrase(current_beat: float, phrase_beats: float,
                              chords: List[ChordEvent], params: MusicParams,
                              swing: bool, coltrane: bool,
                              state: _SoloState,
                              tension: float) -> Tuple[List[NoteEvent], float]:
    notes = []
    beat = current_beat
    phrase_end = current_beat + phrase_beats
    triplet_dur_ticks = TICKS_PER_QUARTER // 3
    triplet_dur_beats = 1.0 / 3.0
    num_groups = random.randint(2, 4)

    for group_idx in range(num_groups):
        if beat >= phrase_end - 1e-9:
            break
        chord = _get_chord_at_beat(chords, beat)
        if chord is None:
            beat += 1.0
            continue
        low = params.register_low
        high = params.register_high
        chord_notes = _chord_tones_in_range(chord.root_pc, chord.quality, low, high)
        if not chord_notes:
            beat += 1.0
            continue

        start_note = min(chord_notes, key=lambda t: abs(t - state.pitch))
        idx = chord_notes.index(start_note)
        direction = random.choice([-1, 1])

        for note_i in range(3):
            if beat + triplet_dur_beats > phrase_end + 1e-9:
                break
            if 0 <= idx < len(chord_notes):
                note_midi = chord_notes[idx]
            else:
                direction = -direction
                idx = max(0, min(len(chord_notes) - 1, idx))
                note_midi = chord_notes[idx]
            tick = _beat_to_tick(beat)
            if swing:
                tick = _apply_swing(tick)
            tick = _humanize(tick)
            vel = _choose_velocity(params)
            notes.append(NoteEvent(
                pitch=note_midi, start_tick=tick,
                duration_ticks=triplet_dur_ticks, velocity=vel,
            ))
            state.pitch = note_midi
            state.record_pitch(note_midi)
            beat += triplet_dur_beats
            idx += direction

        if group_idx < num_groups - 1 and random.random() < 0.5:
            beat += 0.5

    return notes, beat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_head_melody(chords: List[ChordEvent], total_beats: float,
                         swing: bool = True) -> List[NoteEvent]:
    """Generate a composed head melody -- simpler, thematic feel."""
    if not chords:
        return []

    notes: List[NoteEvent] = []
    beat = 0.0
    mid_pitch = (MELODY_LOW_BASE + MELODY_HIGH_BASE) // 2
    pitch = _nearest_chord_tone(mid_pitch, chords[0].root_pc, chords[0].quality,
                                MELODY_LOW_BASE, MELODY_HIGH_BASE)
    recent_pitches = [pitch]

    while beat < total_beats:
        # Varied phrase lengths (1-8 bars)
        phrase_bars = random.choice([1, 2, 2, 3, 3, 4, 4, 6, 8])
        phrase_beats = float(phrase_bars * 4)
        phrase_end = min(beat + phrase_beats, total_beats)

        while beat < phrase_end - 1e-9:
            chord = _get_chord_at_beat(chords, beat)
            if chord is None:
                beat += 1.0
                continue

            # Harmonic anticipation
            approach = _approach_next_chord(pitch, chords, beat,
                                            MELODY_LOW_BASE, MELODY_HIGH_BASE)
            if approach is not None and random.random() < 0.4:
                target = approach
            else:
                target = _choose_target_pitch(
                    pitch, chord.root_pc, chord.quality,
                    MELODY_LOW_BASE, MELODY_HIGH_BASE,
                    tension=0.15, beat_in_bar=beat % 4.0)

            # Contour check
            if _contour_check(recent_pitches):
                direction = -1 if recent_pitches[-1] > recent_pitches[-max(2, len(recent_pitches))] else 1
                candidates = _chord_tones_in_range(chord.root_pc, chord.quality,
                                                   MELODY_LOW_BASE, MELODY_HIGH_BASE)
                if direction > 0:
                    above = [t for t in candidates if t > pitch]
                    if above:
                        target = above[0]
                else:
                    below = [t for t in candidates if t < pitch]
                    if below:
                        target = below[-1]

            # Syncopated rhythm via cells
            cell = _choose_rhythmic_cell(tension=0.15)
            dur = cell[0]  # Use first duration from the cell
            if beat + dur > phrase_end:
                remaining = phrase_end - beat
                if remaining < 0.25:
                    break
                dur = remaining

            tick = _beat_to_tick(beat)
            if swing:
                tick = _apply_swing(tick)
            tick = _humanize(tick)
            dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
            vel = random.randint(75, 90)

            notes.append(NoteEvent(
                pitch=target, start_tick=tick,
                duration_ticks=dur_ticks, velocity=vel,
            ))

            pitch = target
            recent_pitches.append(pitch)
            if len(recent_pitches) > 20:
                recent_pitches = recent_pitches[-20:]
            beat += dur

        # Breathing -- varied rest lengths
        rest_dur = random.choice([0.5, 1.0, 1.0, 1.5, 2.0, 4.0])
        beat += rest_dur

    notes.sort(key=lambda n: n.start_tick)
    return notes


def generate_solo(chords: List[ChordEvent], total_beats: float,
                  tension_curve: str = "arc", swing: bool = True,
                  coltrane: bool = False, seed: Optional[int] = None) -> List[NoteEvent]:
    """Generate an improvised jazz solo with tension-driven phrasing."""
    if seed is not None:
        random.seed(seed)
    if not chords:
        return []

    curve = TensionCurve(tension_curve)
    notes: List[NoteEvent] = []

    mid_pitch = (MELODY_LOW_BASE + MELODY_HIGH_BASE) // 2
    state = _SoloState(
        pitch=_nearest_chord_tone(mid_pitch, chords[0].root_pc, chords[0].quality,
                                  MELODY_LOW_BASE, MELODY_HIGH_BASE),
    )
    state.record_pitch(state.pitch)

    beat = 0.0
    while beat < total_beats:
        progress = beat / total_beats if total_beats > 0 else 0.0
        tension = curve(progress)
        params = interpolate_params(tension)

        # Smooth tier selection with probability blending
        tier = _select_tier(tension)

        # Phrase length -- responds to harmonic rhythm and tier
        beats_until_change = _beats_until_chord_change(chords, beat)
        harmonic_rhythm_fast = beats_until_change <= 2.0

        if tier == 1:
            if harmonic_rhythm_fast:
                phrase_beats = float(random.randint(2, 4))
            else:
                phrase_beats = float(random.choice([4, 4, 8, 8, 12, 16]))
        elif tier == 2:
            phrase_beats = float(random.randint(3, 8))
        else:
            phrase_beats = float(random.randint(2, 6))

        # Short stab phrases (10% chance, tier 1-2 only)
        if random.random() < 0.10 and tier <= 2:
            phrase_beats = float(random.choice([1, 1, 1.5, 2]))

        # Strategic silence (8% chance in tier 1)
        if random.random() < 0.08 and tier == 1:
            beat += 4.0
            continue

        # Clamp to remaining beats
        phrase_beats = min(phrase_beats, total_beats - beat)
        if phrase_beats < 0.5:
            break

        # Coltrane mode: key center awareness
        if coltrane:
            chord = _get_chord_at_beat(chords, beat)
            if chord is not None and chord.key_center_pc != chord.root_pc:
                _substitute_key_tones(chord.key_center_pc,
                                      params.register_low,
                                      params.register_high)

        # Generate phrase based on tier
        if tier == 1:
            phrase_notes, beat = _generate_tier1_phrase(
                beat, phrase_beats, chords, params, swing, state, tension)
        elif tier == 2:
            sub_type = random.choices(
                ["motivic", "pentatonic", "call_response", "triplet"],
                weights=[0.35, 0.25, 0.20, 0.20], k=1)[0]
            if sub_type == "pentatonic" and tension > 0.5:
                phrase_notes, beat = _generate_pentatonic_super_phrase(
                    beat, phrase_beats, chords, params, swing, coltrane, state, tension)
            elif sub_type == "call_response":
                phrase_notes, beat = _generate_call_response_phrase(
                    beat, phrase_beats, chords, params, swing, coltrane, state, tension)
            elif sub_type == "triplet" and tension > 0.5:
                phrase_notes, beat = _generate_triplet_phrase(
                    beat, phrase_beats, chords, params, swing, coltrane, state, tension)
            else:
                phrase_notes, beat = _generate_tier2_phrase(
                    beat, phrase_beats, chords, params, swing, coltrane, state, tension)
        else:
            phrase_notes, beat = _generate_tier3_phrase(
                beat, phrase_beats, chords, params, swing, coltrane, state, tension)

        notes.extend(phrase_notes)
        state.phrase_count += 1

        # Ghost notes (Coltrane mode)
        if coltrane and 0.3 < tension < 0.6 and random.random() < 0.4:
            num_ghost = random.randint(1, 3)
            ghost_beat = beat
            ghost_pitch = state.pitch
            for _ in range(num_ghost):
                if ghost_beat >= total_beats:
                    break
                direction = random.choice([-1, 1])
                ghost_pitch = max(MELODY_LOW, min(MELODY_HIGH, ghost_pitch + direction))
                ghost_tick = _beat_to_tick(ghost_beat)
                if swing:
                    ghost_tick = _apply_swing(ghost_tick)
                ghost_dur = TICKS_PER_16TH
                ghost_vel = random.randint(20, 35)
                notes.append(NoteEvent(
                    pitch=ghost_pitch, start_tick=ghost_tick,
                    duration_ticks=ghost_dur, velocity=ghost_vel,
                ))
                ghost_beat += 0.25

        # Rest between phrases -- varied by tier
        if random.random() < params.rest_prob or tier == 1:
            if tier == 3:
                rest_dur = random.choice([0.25, 0.5])
            elif tier == 2:
                rest_dur = random.choice([0.5, 1.0, 1.5])
            else:
                rest_dur = random.choice([1.0, 1.5, 2.0, 4.0])
            beat += rest_dur

    notes.sort(key=lambda n: n.start_tick)
    return notes


def generate_trading_fours(chords: List[ChordEvent], total_beats: float,
                           intensity: float = 0.6) -> List[NoteEvent]:
    """Generate a trading-fours section (4 bars melody, 4 bars drums)."""
    if not chords:
        return []

    notes: List[NoteEvent] = []
    beat = 0.0
    phrase_size = 32.0
    melody_beats = 16.0

    mid_pitch = (MELODY_LOW_BASE + MELODY_HIGH_BASE) // 2
    pitch = _nearest_chord_tone(mid_pitch, chords[0].root_pc, chords[0].quality,
                                MELODY_LOW_BASE, MELODY_HIGH_BASE)
    recent_pitches = [pitch]
    params = interpolate_params(min(1.0, intensity + 0.15))

    while beat < total_beats:
        melody_end = min(beat + melody_beats, total_beats)
        melody_section_beats = melody_end - beat
        if melody_section_beats <= 0:
            break

        current_beat = beat
        while current_beat < melody_end - 1e-9:
            chord = _get_chord_at_beat(chords, current_beat)
            if chord is None:
                current_beat += 0.5
                continue

            low = params.register_low
            high = params.register_high

            # Use _choose_target_pitch for variety
            target = _choose_target_pitch(
                pitch, chord.root_pc, chord.quality,
                low, high, tension=intensity, beat_in_bar=current_beat % 4.0)

            # Use rhythmic cells
            cell = _choose_rhythmic_cell(tension=intensity)
            dur = cell[0]

            if _contour_check(recent_pitches):
                direction = -1 if len(recent_pitches) >= 2 and recent_pitches[-1] > recent_pitches[-2] else 1
                candidates = _chord_tones_in_range(chord.root_pc, chord.quality, low, high)
                if direction > 0:
                    above = [t for t in candidates if t > pitch]
                    if above:
                        target = above[0]
                else:
                    below = [t for t in candidates if t < pitch]
                    if below:
                        target = below[-1]

            if current_beat + dur > melody_end:
                remaining = melody_end - current_beat
                if remaining < 0.125:
                    break
                dur = remaining

            tick = _beat_to_tick(current_beat)
            tick = _apply_swing(tick)
            tick = _humanize(tick)
            dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
            vel = _choose_velocity(params)

            notes.append(NoteEvent(
                pitch=target, start_tick=tick,
                duration_ticks=dur_ticks, velocity=vel,
            ))

            pitch = target
            recent_pitches.append(pitch)
            if len(recent_pitches) > 20:
                recent_pitches = recent_pitches[-20:]
            current_beat += dur

        beat += phrase_size

    notes.sort(key=lambda n: n.start_tick)
    return notes
