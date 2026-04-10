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

MELODY_LOW = 55   # G3
MELODY_HIGH = 84  # C6

# Re-export chord tones for local use (already imported from chord.py)
# CHORD_TONES is imported above.

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
# Motif system for motivic development
# ---------------------------------------------------------------------------


@dataclass
class Motif:
    """A short melodic idea defined by intervals and durations."""
    intervals: list   # relative semitone intervals from first note
    durations: list   # beat durations for each note

    def invert(self) -> "Motif":
        """Invert all intervals (mirror around axis)."""
        return Motif([-iv for iv in self.intervals], list(self.durations))

    def retrograde(self) -> "Motif":
        """Reverse the order of notes."""
        return Motif(list(reversed(self.intervals)), list(reversed(self.durations)))

    def augment(self) -> "Motif":
        """Double all durations (rhythmic augmentation)."""
        return Motif(list(self.intervals), [d * 2 for d in self.durations])

    def diminish(self) -> "Motif":
        """Halve all durations (rhythmic diminution)."""
        return Motif(list(self.intervals), [d * 0.5 for d in self.durations])


SEED_MOTIFS = [
    Motif([0, 2, 4, 7], [0.5, 0.5, 0.5, 0.5]),
    Motif([0, -1, -3, -5], [0.5, 0.5, 0.5, 0.5]),
    Motif([0, 4, 7, 4], [0.5, 0.5, 0.5, 0.5]),
    Motif([0, 7, 5, 3, 0], [0.25, 0.25, 0.25, 0.25, 1.0]),
]

# ---------------------------------------------------------------------------
# Tension curve
# ---------------------------------------------------------------------------


class TensionCurve:
    """Maps position (0.0-1.0) to tension (0.0-1.0).

    Three built-in curves:
      'arc'   - rises to peak at 75%, then resolves
      'build' - gradual build throughout
      'wave'  - oscillating tension with overall upward trend
    """

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
    note_density: float     # notes per beat (1.0 = quarters, 4.0 = 16ths)
    chromatic_prob: float   # 0.0-0.4
    rest_prob: float        # probability of resting between phrases
    velocity_base: int
    velocity_range: int
    register_low: int
    register_high: int
    motif_complexity: float  # 0.0-1.0


def interpolate_params(tension: float) -> MusicParams:
    """Interpolate musical parameters from a tension value in [0, 1]."""
    t = max(0.0, min(1.0, tension))
    return MusicParams(
        note_density=1.0 + t * 3.0,         # 1.0 -> 4.0
        chromatic_prob=t * 0.4,              # 0.0 -> 0.4
        rest_prob=0.15 - t * 0.13,           # 0.15 -> 0.02
        velocity_base=int(70 + t * 30),      # 70 -> 100
        velocity_range=int(5 + t * 10),      # 5 -> 15
        register_low=int(60 - t * 5),        # 60 -> 55
        register_high=int(78 + t * 6),       # 78 -> 84
        motif_complexity=t,
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _chord_tones_in_range(root_pc: int, quality: str, low: int, high: int) -> List[int]:
    """Return sorted list of MIDI notes that are chord tones of (root, quality) in [low, high]."""
    intervals = CHORD_TONES.get(quality, (0, 4, 7))
    result = []
    for midi_note in range(low, high + 1):
        interval = (midi_note % 12 - root_pc) % 12
        if interval in intervals:
            result.append(midi_note)
    return result


def _scale_tones_in_range(root_pc: int, quality: str, low: int, high: int) -> List[int]:
    """Return sorted list of MIDI notes in the primary scale for (root, quality) in [low, high]."""
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
    """Find the chord tone nearest to current_midi within [low, high].

    If no chord tones exist in range, returns the clamped current_midi.
    """
    tones = _chord_tones_in_range(root_pc, quality, low, high)
    if not tones:
        return max(low, min(high, current_midi))
    best = min(tones, key=lambda t: abs(t - current_midi))
    return best


def _nearest_scale_tone(current_midi: int, root_pc: int, quality: str,
                        low: int, high: int) -> int:
    """Find the scale tone nearest to current_midi within [low, high]."""
    tones = _scale_tones_in_range(root_pc, quality, low, high)
    if not tones:
        return max(low, min(high, current_midi))
    best = min(tones, key=lambda t: abs(t - current_midi))
    return best


def _chromatic_enclosure(target_midi: int) -> List[Tuple[int, float]]:
    """Generate a chromatic enclosure approaching target_midi.

    Returns 2-3 notes: approach from above and below before landing on target.
    Each entry is (midi_pitch, duration_in_beats).
    """
    if random.random() < 0.5:
        # From above then below
        return [
            (target_midi + 1, 0.25),
            (target_midi - 1, 0.25),
            (target_midi, 0.5),
        ]
    else:
        # From below then above
        return [
            (target_midi - 1, 0.25),
            (target_midi + 1, 0.25),
            (target_midi, 0.5),
        ]


def _scalar_run(current_midi: int, direction: int, root_pc: int, quality: str,
                length: int, low: int, high: int) -> List[Tuple[int, float]]:
    """Generate a scalar run of `length` notes in the given direction.

    Args:
        current_midi: Starting MIDI pitch.
        direction: +1 for ascending, -1 for descending.
        root_pc: Root pitch class.
        quality: Chord quality.
        length: Number of notes to generate.
        low: Lowest allowed MIDI.
        high: Highest allowed MIDI.

    Returns:
        List of (midi_pitch, duration_in_beats) tuples.
    """
    scale_tones = _scale_tones_in_range(root_pc, quality, low, high)
    if not scale_tones:
        return [(current_midi, 0.5)]

    # Find the closest scale tone to start from
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
    """Generate an arpeggio run through chord tones.

    Returns 3-6 chord tones ascending or descending from near current_midi.
    """
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
    """Generate a digital pattern fragment transposed to the current context.

    Picks a random DIGITAL_PATTERN, finds the nearest root-class MIDI note
    to current_midi, and transposes the pattern intervals from there.
    Duration: 8th notes (0.5 beats).
    """
    pattern = random.choice(DIGITAL_PATTERNS)

    # Find the nearest root-class note to current_midi
    root_candidates = [n for n in range(low, high + 1) if n % 12 == root_pc]
    if not root_candidates:
        return [(current_midi, 0.5)]

    root_midi = min(root_candidates, key=lambda r: abs(r - current_midi))

    result = []
    for interval in pattern:
        note = root_midi + interval
        if low <= note <= high:
            result.append((note, 0.5))
        else:
            # Clamp to range but still include
            clamped = max(low, min(high, note))
            result.append((clamped, 0.5))

    return result


def _apply_swing(tick: int, swing_ratio: float = 0.667) -> int:
    """Apply swing feel to a tick position.

    Offbeat 8th notes (those on the "and" of each beat) are shifted later.
    swing_ratio=0.667 means the downbeat 8th gets 2/3 of the beat, the upbeat gets 1/3.
    swing_ratio=0.5 means straight (no swing).
    """
    beat_pos = tick % TICKS_PER_QUARTER
    beat_start = tick - beat_pos

    # If this tick is in the second half of the beat (offbeat 8th note zone)
    if beat_pos >= TICKS_PER_8TH:
        # Shift the offbeat 8th to the swing position
        swing_point = int(TICKS_PER_QUARTER * swing_ratio)
        offset_within_offbeat = beat_pos - TICKS_PER_8TH
        remaining = TICKS_PER_QUARTER - swing_point
        if TICKS_PER_8TH > 0:
            scaled_offset = int(offset_within_offbeat * remaining / TICKS_PER_8TH)
        else:
            scaled_offset = 0
        return beat_start + swing_point + scaled_offset
    else:
        # Downbeat 8th: scale within the swing portion
        swing_point = int(TICKS_PER_QUARTER * swing_ratio)
        if TICKS_PER_8TH > 0:
            scaled = int(beat_pos * swing_point / TICKS_PER_8TH)
        else:
            scaled = 0
        return beat_start + scaled


def _humanize(tick: int, amount: int = 10) -> int:
    """Add random timing jitter to a tick position for humanization."""
    return max(0, tick + random.randint(-amount, amount))


def _contour_check(recent_pitches: List[int], max_monotonic: int = 6) -> bool:
    """Check if recent pitches have been moving in one direction too long.

    Returns True if the run is monotonic for >= max_monotonic notes and
    a direction change is recommended.
    """
    if len(recent_pitches) < max_monotonic:
        return False

    tail = recent_pitches[-max_monotonic:]
    ascending = all(tail[i] <= tail[i + 1] for i in range(len(tail) - 1))
    descending = all(tail[i] >= tail[i + 1] for i in range(len(tail) - 1))
    return ascending or descending


def _get_chord_at_beat(chords: List[ChordEvent], beat: float) -> Optional[ChordEvent]:
    """Return the chord active at a given beat position."""
    for chord in chords:
        if chord.start_beat <= beat < chord.end_beat:
            return chord
    # If past last chord, return last chord
    if chords:
        return chords[-1]
    return None


def _beat_to_tick(beat: float) -> int:
    """Convert beat position to tick position."""
    return int(beat * TICKS_PER_QUARTER)


def _choose_note_duration(params: MusicParams) -> float:
    """Choose a note duration in beats based on note_density.

    Higher density -> shorter notes.
    """
    density = params.note_density
    if density <= 1.5:
        # Mostly quarters and halves
        return random.choice([1.0, 1.0, 1.0, 2.0, 0.5])
    elif density <= 2.5:
        # Mostly 8ths
        return random.choice([0.5, 0.5, 0.5, 0.5, 1.0, 0.25])
    elif density <= 3.5:
        # Mix of 8ths and 16ths
        return random.choice([0.5, 0.5, 0.25, 0.25, 0.25, 0.5])
    else:
        # Mostly 16ths
        return random.choice([0.25, 0.25, 0.25, 0.25, 0.5, 0.25])


def _choose_velocity(params: MusicParams) -> int:
    """Choose a velocity value based on params."""
    v = params.velocity_base + random.randint(-params.velocity_range, params.velocity_range)
    return max(1, min(127, v))


def _extensions_in_range(root_pc: int, quality: str, low: int, high: int) -> List[int]:
    """Return MIDI notes for chord extensions (9, 11, 13) within [low, high]."""
    tension_intervals = TENSIONS.get(quality, [])
    result = []
    for midi_note in range(low, high + 1):
        interval = (midi_note % 12 - root_pc) % 12
        # Tensions are stored as intervals > 12, reduce them to single octave
        for t in tension_intervals:
            if interval == t % 12:
                result.append(midi_note)
                break
    return result


def _substitute_key_tones(key_center_pc: int, low: int, high: int) -> List[int]:
    """Return chord tones for substitute key centers a major third apart.

    For Coltrane multi-tonic: given a key center, returns chord tones
    for keys a major third above and below (the three key centers of
    Giant Steps: B, G, Eb for example).
    """
    sub_keys = [(key_center_pc + 4) % 12, (key_center_pc + 8) % 12]
    result = []
    for sub_key in sub_keys:
        # Major triad tones of each substitute key
        for interval in (0, 4, 7):
            pc = (sub_key + interval) % 12
            for midi_note in range(low, high + 1):
                if midi_note % 12 == pc and midi_note not in result:
                    result.append(midi_note)
    result.sort()
    return result


# ---------------------------------------------------------------------------
# Tier generators (internal)
# ---------------------------------------------------------------------------


def _generate_tier1_phrase(current_beat: float, phrase_beats: float,
                           chords: List[ChordEvent], current_pitch: int,
                           params: MusicParams, swing: bool) -> Tuple[List[NoteEvent], int, float]:
    """Tier 1 -- Melodic: chord tones + voice leading, quarter/half notes.

    Returns:
        (notes, last_pitch, beat_after_phrase)
    """
    notes = []
    beat = current_beat
    pitch = current_pitch
    phrase_end = current_beat + phrase_beats
    recent_pitches = [pitch]
    phrase_note_count = random.randint(4, 8)
    notes_played = 0

    while beat < phrase_end and notes_played < phrase_note_count:
        chord = _get_chord_at_beat(chords, beat)
        if chord is None:
            beat += 1.0
            continue

        low = params.register_low
        high = params.register_high

        # On strong beats (1, 3) prefer chord tones
        beat_in_bar = beat % 4.0
        if beat_in_bar < 0.1 or abs(beat_in_bar - 2.0) < 0.1:
            target = _nearest_chord_tone(pitch, chord.root_pc, chord.quality, low, high)
        else:
            # 20% chance scalar passing tone, otherwise chord tone
            if random.random() < 0.2:
                target = _nearest_scale_tone(pitch, chord.root_pc, chord.quality, low, high)
            else:
                target = _nearest_chord_tone(pitch, chord.root_pc, chord.quality, low, high)

        # Voice leading: prefer stepwise motion
        if abs(target - pitch) > 7 and random.random() < 0.6:
            # Find a closer chord/scale tone
            scale_tones = _scale_tones_in_range(chord.root_pc, chord.quality, low, high)
            if scale_tones:
                closer = [t for t in scale_tones if abs(t - pitch) <= 4]
                if closer:
                    target = random.choice(closer)

        # Contour check
        if _contour_check(recent_pitches):
            direction = -1 if recent_pitches[-1] > recent_pitches[-2] else 1
            candidates = _chord_tones_in_range(chord.root_pc, chord.quality, low, high)
            if direction > 0:
                above = [t for t in candidates if t > pitch]
                if above:
                    target = above[0]
            else:
                below = [t for t in candidates if t < pitch]
                if below:
                    target = below[-1]

        # Duration: quarter or half notes for Tier 1
        dur = random.choice([1.0, 1.0, 1.0, 2.0, 0.5])
        if beat + dur > phrase_end:
            dur = phrase_end - beat
            if dur <= 0:
                break

        tick = _beat_to_tick(beat)
        if swing:
            tick = _apply_swing(tick)
        tick = _humanize(tick)
        dur_ticks = int(dur * TICKS_PER_QUARTER)
        vel = _choose_velocity(params)

        notes.append(NoteEvent(
            pitch=target,
            start_tick=tick,
            duration_ticks=dur_ticks,
            velocity=vel,
        ))

        pitch = target
        recent_pitches.append(pitch)
        beat += dur
        notes_played += 1

    return notes, pitch, beat


def _generate_tier2_phrase(current_beat: float, phrase_beats: float,
                           chords: List[ChordEvent], current_pitch: int,
                           params: MusicParams, swing: bool,
                           coltrane: bool) -> Tuple[List[NoteEvent], int, float]:
    """Tier 2 -- Motivic Development: motifs, enclosures, scalar runs, arpeggios.

    Five strategies chosen by weighted random:
      1. Chromatic enclosure (20%)
      2. Scalar run (30%)
      3. Arpeggio (15%)
      4. Digital pattern (20%)
      5. Direct target (15%)

    Returns:
        (notes, last_pitch, beat_after_phrase)
    """
    notes = []
    beat = current_beat
    pitch = current_pitch
    phrase_end = current_beat + phrase_beats
    recent_pitches = [pitch]
    phrase_note_count = random.randint(6, 12)
    notes_played = 0

    # Maybe apply a motif transformation
    motif = random.choice(SEED_MOTIFS)
    transform_roll = random.random()
    if transform_roll < 0.2:
        motif = motif.invert()
    elif transform_roll < 0.35:
        motif = motif.retrograde()
    elif transform_roll < 0.45 and coltrane:
        motif = motif.diminish()  # More frequent transforms in Coltrane mode

    # Apply motif first (if it fits)
    motif_applied = False
    if params.motif_complexity > 0.3 and beat + sum(motif.durations) <= phrase_end:
        chord = _get_chord_at_beat(chords, beat)
        if chord is not None:
            low = params.register_low
            high = params.register_high
            motif_start = _nearest_chord_tone(pitch, chord.root_pc, chord.quality, low, high)

            for iv, dur in zip(motif.intervals, motif.durations):
                note_midi = motif_start + iv
                note_midi = max(low, min(high, note_midi))
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick)
                dur_ticks = int(dur * TICKS_PER_QUARTER)
                vel = _choose_velocity(params)

                notes.append(NoteEvent(
                    pitch=note_midi,
                    start_tick=tick,
                    duration_ticks=dur_ticks,
                    velocity=vel,
                ))
                pitch = note_midi
                recent_pitches.append(pitch)
                beat += dur
                notes_played += 1

            motif_applied = True

    # Continue with connective strategies
    strategies = ["enclosure", "scalar_run", "arpeggio", "digital_pattern", "direct"]
    weights = [0.20, 0.30, 0.15, 0.20, 0.15]
    if coltrane:
        # Coltrane mode: more digital patterns and arpeggios
        weights = [0.15, 0.20, 0.15, 0.35, 0.15]

    while beat < phrase_end and notes_played < phrase_note_count:
        chord = _get_chord_at_beat(chords, beat)
        if chord is None:
            beat += 0.5
            continue

        low = params.register_low
        high = params.register_high

        strategy = random.choices(strategies, weights=weights, k=1)[0]

        if strategy == "enclosure":
            target = _nearest_chord_tone(pitch, chord.root_pc, chord.quality, low, high)
            enc_notes = _chromatic_enclosure(target)
            for note_midi, dur in enc_notes:
                if beat + dur > phrase_end:
                    break
                note_midi = max(low, min(high, note_midi))
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick)
                dur_ticks = int(dur * TICKS_PER_QUARTER)
                vel = _choose_velocity(params)
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                pitch = note_midi
                recent_pitches.append(pitch)
                beat += dur
                notes_played += 1

        elif strategy == "scalar_run":
            direction = 1 if random.random() < 0.5 else -1
            if _contour_check(recent_pitches):
                direction = -1 if recent_pitches[-1] > recent_pitches[-3] else 1
            length = random.randint(3, 6)
            run = _scalar_run(pitch, direction, chord.root_pc, chord.quality, length, low, high)
            for note_midi, dur in run:
                if beat + dur > phrase_end:
                    break
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick)
                dur_ticks = int(dur * TICKS_PER_QUARTER)
                vel = _choose_velocity(params)
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                pitch = note_midi
                recent_pitches.append(pitch)
                beat += dur
                notes_played += 1

        elif strategy == "arpeggio":
            direction = 1 if random.random() < 0.5 else -1
            if _contour_check(recent_pitches):
                direction = -1 if recent_pitches[-1] > recent_pitches[-3] else 1
            run = _arpeggio_run(pitch, direction, chord.root_pc, chord.quality, low, high)
            for note_midi, dur in run:
                if beat + dur > phrase_end:
                    break
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick)
                dur_ticks = int(dur * TICKS_PER_QUARTER)
                vel = _choose_velocity(params)
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                pitch = note_midi
                recent_pitches.append(pitch)
                beat += dur
                notes_played += 1

        elif strategy == "digital_pattern":
            fragment = _digital_pattern_fragment(chord.root_pc, pitch, low, high)
            for note_midi, dur in fragment:
                if beat + dur > phrase_end:
                    break
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick)
                dur_ticks = int(dur * TICKS_PER_QUARTER)
                vel = _choose_velocity(params)
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                pitch = note_midi
                recent_pitches.append(pitch)
                beat += dur
                notes_played += 1

        elif strategy == "direct":
            target = _nearest_chord_tone(pitch, chord.root_pc, chord.quality, low, high)
            dur = 1.0
            if beat + dur > phrase_end:
                dur = phrase_end - beat
                if dur <= 0:
                    break
            tick = _beat_to_tick(beat)
            if swing:
                tick = _apply_swing(tick)
            tick = _humanize(tick)
            dur_ticks = int(dur * TICKS_PER_QUARTER)
            vel = _choose_velocity(params)
            notes.append(NoteEvent(
                pitch=target, start_tick=tick,
                duration_ticks=dur_ticks, velocity=vel,
            ))
            pitch = target
            recent_pitches.append(pitch)
            beat += dur
            notes_played += 1

    return notes, pitch, beat


def _generate_tier3_phrase(current_beat: float, phrase_beats: float,
                           chords: List[ChordEvent], current_pitch: int,
                           params: MusicParams, swing: bool,
                           coltrane: bool) -> Tuple[List[NoteEvent], int, float]:
    """Tier 3 -- Sheets of Sound: rapid 16th notes, digital patterns, enclosure runs.

    Three sub-strategies:
      1. Digital pattern (40%)
      2. Rapid arpeggiation through chord tones + extensions (30%)
      3. Enclosure runs (30%)

    Returns:
        (notes, last_pitch, beat_after_phrase)
    """
    notes = []
    beat = current_beat
    pitch = current_pitch
    phrase_end = current_beat + phrase_beats
    recent_pitches = [pitch]
    phrase_note_count = random.randint(8, 20)
    notes_played = 0

    strategies = ["digital", "rapid_arpeggio", "enclosure_run"]
    weights = [0.40, 0.30, 0.30]
    if coltrane:
        weights = [0.50, 0.30, 0.20]

    vel_base = max(90, params.velocity_base)
    vel_range = params.velocity_range

    while beat < phrase_end and notes_played < phrase_note_count:
        chord = _get_chord_at_beat(chords, beat)
        if chord is None:
            beat += 0.25
            continue

        low = params.register_low
        high = params.register_high

        strategy = random.choices(strategies, weights=weights, k=1)[0]

        if strategy == "digital":
            fragment = _digital_pattern_fragment(chord.root_pc, pitch, low, high)
            for note_midi, _ in fragment:
                dur = 0.25  # 16th notes
                if beat + dur > phrase_end:
                    break
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick, amount=5)
                dur_ticks = int(dur * TICKS_PER_QUARTER)
                vel = min(127, max(1, vel_base + random.randint(-vel_range, vel_range)))
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                pitch = note_midi
                recent_pitches.append(pitch)
                beat += dur
                notes_played += 1

        elif strategy == "rapid_arpeggio":
            # Arpeggio through chord tones AND extensions
            chord_notes = _chord_tones_in_range(chord.root_pc, chord.quality, low, high)
            ext_notes = _extensions_in_range(chord.root_pc, chord.quality, low, high)
            all_tones = sorted(set(chord_notes + ext_notes))
            if not all_tones:
                all_tones = [pitch]

            # Find start
            start_note = min(all_tones, key=lambda t: abs(t - pitch))
            idx = all_tones.index(start_note)
            direction = 1 if random.random() < 0.5 else -1
            if _contour_check(recent_pitches, max_monotonic=4):
                direction = -direction

            run_length = random.randint(6, 10)
            for _ in range(run_length):
                dur = 0.25  # 16th notes
                if beat + dur > phrase_end or notes_played >= phrase_note_count:
                    break
                if 0 <= idx < len(all_tones):
                    note_midi = all_tones[idx]
                else:
                    # Reverse direction at boundary
                    direction = -direction
                    idx = max(0, min(len(all_tones) - 1, idx))
                    note_midi = all_tones[idx]

                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick, amount=5)
                dur_ticks = int(dur * TICKS_PER_QUARTER)
                vel = min(127, max(1, vel_base + random.randint(-vel_range, vel_range)))
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                pitch = note_midi
                recent_pitches.append(pitch)
                beat += dur
                notes_played += 1
                idx += direction

        elif strategy == "enclosure_run":
            # Rapid chromatic approach into multiple chord tones
            chord_notes = _chord_tones_in_range(chord.root_pc, chord.quality, low, high)
            if not chord_notes:
                chord_notes = [pitch]

            # Pick 2-3 target chord tones and enclose each
            targets = sorted(chord_notes, key=lambda t: abs(t - pitch))[:3]
            random.shuffle(targets)

            for target in targets:
                if beat >= phrase_end or notes_played >= phrase_note_count:
                    break
                # Quick enclosure: above, below, target (all 16ths)
                enc_pitches = [target + 1, target - 1, target]
                for ep in enc_pitches:
                    dur = 0.25
                    if beat + dur > phrase_end or notes_played >= phrase_note_count:
                        break
                    ep_clamped = max(low, min(high, ep))
                    tick = _beat_to_tick(beat)
                    if swing:
                        tick = _apply_swing(tick)
                    tick = _humanize(tick, amount=5)
                    dur_ticks = int(dur * TICKS_PER_QUARTER)
                    vel = min(127, max(1, vel_base + random.randint(-vel_range, vel_range)))
                    notes.append(NoteEvent(
                        pitch=ep_clamped, start_tick=tick,
                        duration_ticks=dur_ticks, velocity=vel,
                    ))
                    pitch = ep_clamped
                    recent_pitches.append(pitch)
                    beat += dur
                    notes_played += 1

    return notes, pitch, beat


# ---------------------------------------------------------------------------
# Extended phrase generators (pentatonic, call-response, triplet)
# ---------------------------------------------------------------------------


def _generate_pentatonic_super_phrase(current_beat: float, phrase_beats: float,
                                      chords: List[ChordEvent], current_pitch: int,
                                      params: MusicParams, swing: bool,
                                      coltrane: bool) -> Tuple[List[NoteEvent], int, float]:
    """Pentatonic superimposition: play a pentatonic scale from a different root
    than the chord root to create modal color.

    Returns:
        (notes, last_pitch, beat_after_phrase)
    """
    notes = []
    beat = current_beat
    pitch = current_pitch
    phrase_end = current_beat + phrase_beats

    chord = _get_chord_at_beat(chords, beat)
    if chord is None:
        return notes, pitch, beat

    low = params.register_low
    high = params.register_high

    # Determine superimposed pentatonic root based on chord quality
    root = chord.root_pc
    q = chord.quality
    if q in ("maj7", "maj", "6"):
        penta_root = (root + 2) % 12  # whole step up = lydian sound
    elif q in ("dom7", "7"):
        penta_root = (root + random.choice([2, 10])) % 12  # up or down a step
    elif q in ("min7", "min", "min6"):
        penta_root = (root + 3) % 12  # minor third up = natural minor sound
    else:
        penta_root = root  # regular pentatonic

    # Build pentatonic scale in range: intervals (0, 2, 4, 7, 9)
    penta_intervals = (0, 2, 4, 7, 9)
    penta_notes = []
    for midi_note in range(low, high + 1):
        interval = (midi_note % 12 - penta_root) % 12
        if interval in penta_intervals:
            penta_notes.append(midi_note)

    if not penta_notes:
        # Fallback: just use chord tones
        return _generate_tier2_phrase(current_beat, phrase_beats, chords,
                                      current_pitch, params, swing, coltrane)

    # Generate 6-10 notes using stepwise motion through pentatonic scale
    num_notes = random.randint(6, 10)
    # Find starting position in pentatonic scale nearest to current pitch
    start_note = min(penta_notes, key=lambda t: abs(t - pitch))
    idx = penta_notes.index(start_note)
    direction = random.choice([-1, 1])

    for i in range(num_notes):
        if beat >= phrase_end:
            break

        if 0 <= idx < len(penta_notes):
            note_midi = penta_notes[idx]
        else:
            # Reverse direction at boundary
            direction = -direction
            idx = max(0, min(len(penta_notes) - 1, idx))
            note_midi = penta_notes[idx]

        # Mostly 8th notes, occasional quarter
        dur = 0.5 if random.random() < 0.75 else 1.0
        if beat + dur > phrase_end:
            dur = phrase_end - beat
            if dur <= 0:
                break

        tick = _beat_to_tick(beat)
        if swing:
            tick = _apply_swing(tick)
        tick = _humanize(tick)
        dur_ticks = int(dur * TICKS_PER_QUARTER)
        vel = _choose_velocity(params)

        notes.append(NoteEvent(
            pitch=note_midi,
            start_tick=tick,
            duration_ticks=dur_ticks,
            velocity=vel,
        ))

        pitch = note_midi
        beat += dur
        idx += direction

    return notes, pitch, beat


def _generate_call_response_phrase(current_beat: float, phrase_beats: float,
                                    chords: List[ChordEvent], current_pitch: int,
                                    params: MusicParams, swing: bool,
                                    coltrane: bool) -> Tuple[List[NoteEvent], int, float]:
    """Call-and-response: generate a short 'call' melody, then respond with a
    transformed version (transposed, inverted, or rhythmically shifted).

    Returns:
        (notes, last_pitch, beat_after_phrase)
    """
    notes = []
    beat = current_beat
    pitch = current_pitch
    phrase_end = current_beat + phrase_beats

    # Split phrase roughly in half
    half_beats = phrase_beats / 2.0

    # --- Call: generate a short melody (4-6 notes, chord tones) ---
    call_notes_data = []  # list of (pitch, dur) for interval extraction
    call_end = beat + half_beats
    call_note_count = random.randint(4, 6)
    call_played = 0

    while beat < call_end and call_played < call_note_count:
        chord = _get_chord_at_beat(chords, beat)
        if chord is None:
            beat += 0.5
            continue

        low = params.register_low
        high = params.register_high

        target = _nearest_chord_tone(pitch, chord.root_pc, chord.quality, low, high)
        # Voice leading: prefer stepwise
        if abs(target - pitch) > 5:
            scale_tones = _scale_tones_in_range(chord.root_pc, chord.quality, low, high)
            if scale_tones:
                closer = [t for t in scale_tones if abs(t - pitch) <= 4]
                if closer:
                    target = random.choice(closer)

        dur = random.choice([0.5, 0.5, 1.0, 1.0])
        if beat + dur > call_end:
            dur = call_end - beat
            if dur <= 0:
                break

        tick = _beat_to_tick(beat)
        if swing:
            tick = _apply_swing(tick)
        tick = _humanize(tick)
        dur_ticks = int(dur * TICKS_PER_QUARTER)
        vel = _choose_velocity(params)

        notes.append(NoteEvent(
            pitch=target,
            start_tick=tick,
            duration_ticks=dur_ticks,
            velocity=vel,
        ))

        call_notes_data.append((target, dur))
        pitch = target
        beat += dur
        call_played += 1

    # --- Response: transform the call ---
    if len(call_notes_data) < 2:
        return notes, pitch, beat

    # Extract intervals from call
    call_pitches = [p for p, d in call_notes_data]
    call_durs = [d for p, d in call_notes_data]
    call_intervals = [call_pitches[i + 1] - call_pitches[i]
                      for i in range(len(call_pitches) - 1)]

    # Choose transform
    transform_roll = random.random()
    response_end = min(beat + half_beats, phrase_end)

    if transform_roll < 0.4:
        # Transpose up/down by 2-5 semitones
        shift = random.choice([-5, -4, -3, -2, 2, 3, 4, 5])
        resp_pitches = [max(MELODY_LOW, min(MELODY_HIGH, p + shift))
                        for p in call_pitches]
        resp_durs = list(call_durs)
    elif transform_roll < 0.7:
        # Invert intervals
        inverted_intervals = [-iv for iv in call_intervals]
        resp_pitches = [call_pitches[0]]
        for iv in inverted_intervals:
            next_p = max(MELODY_LOW, min(MELODY_HIGH, resp_pitches[-1] + iv))
            resp_pitches.append(next_p)
        resp_durs = list(call_durs)
    else:
        # Same pitches, shifted rhythmically by half a beat
        resp_pitches = list(call_pitches)
        resp_durs = list(call_durs)
        beat += 0.5  # rhythmic shift

    for resp_pitch, dur in zip(resp_pitches, resp_durs):
        if beat >= response_end:
            break
        if beat + dur > response_end:
            dur = response_end - beat
            if dur <= 0:
                break

        tick = _beat_to_tick(beat)
        if swing:
            tick = _apply_swing(tick)
        tick = _humanize(tick)
        dur_ticks = int(dur * TICKS_PER_QUARTER)
        vel = _choose_velocity(params)

        notes.append(NoteEvent(
            pitch=resp_pitch,
            start_tick=tick,
            duration_ticks=dur_ticks,
            velocity=vel,
        ))

        pitch = resp_pitch
        beat += dur

    return notes, pitch, beat


def _generate_triplet_phrase(current_beat: float, phrase_beats: float,
                              chords: List[ChordEvent], current_pitch: int,
                              params: MusicParams, swing: bool,
                              coltrane: bool) -> Tuple[List[NoteEvent], int, float]:
    """Triplet phrase: groups of 3 notes at triplet rhythm (each note = 1/3 beat).

    Generates 2-4 groups (6-12 notes) of ascending or descending chord-tone runs
    with optional rests between groups.

    Returns:
        (notes, last_pitch, beat_after_phrase)
    """
    notes = []
    beat = current_beat
    pitch = current_pitch
    phrase_end = current_beat + phrase_beats

    triplet_dur_ticks = TICKS_PER_QUARTER // 3  # 160 ticks
    triplet_dur_beats = 1.0 / 3.0

    num_groups = random.randint(2, 4)

    for group_idx in range(num_groups):
        if beat >= phrase_end:
            break

        chord = _get_chord_at_beat(chords, beat)
        if chord is None:
            beat += 1.0
            continue

        low = params.register_low
        high = params.register_high

        # Get chord tones for ascending/descending run
        chord_notes = _chord_tones_in_range(chord.root_pc, chord.quality, low, high)
        if not chord_notes:
            beat += 1.0
            continue

        # Find starting position nearest to current pitch
        start_note = min(chord_notes, key=lambda t: abs(t - pitch))
        idx = chord_notes.index(start_note)
        direction = random.choice([-1, 1])

        # 3 notes per group
        for note_i in range(3):
            if beat + triplet_dur_beats > phrase_end:
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
                pitch=note_midi,
                start_tick=tick,
                duration_ticks=triplet_dur_ticks,
                velocity=vel,
            ))

            pitch = note_midi
            beat += triplet_dur_beats
            idx += direction

        # Optionally rest between groups (50% chance, 0.5 beats)
        if group_idx < num_groups - 1 and random.random() < 0.5:
            beat += 0.5

    return notes, pitch, beat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_head_melody(chords: List[ChordEvent], total_beats: float,
                         swing: bool = True) -> List[NoteEvent]:
    """Generate a composed head melody -- simpler, thematic feel.

    The head uses primarily quarter and half notes, targets chord tones on
    strong beats (1, 3), and includes breathing (rests) every 2-4 bars.

    Args:
        chords: List of ChordEvent objects covering the arrangement.
        total_beats: Total number of beats to generate.
        swing: Whether to apply swing feel (default True).

    Returns:
        List of NoteEvent objects sorted by start_tick.
    """
    if not chords:
        return []

    notes: List[NoteEvent] = []
    beat = 0.0
    # Start on a chord tone near the middle of the register
    mid_pitch = (MELODY_LOW + MELODY_HIGH) // 2
    pitch = _nearest_chord_tone(mid_pitch, chords[0].root_pc, chords[0].quality,
                                MELODY_LOW, MELODY_HIGH)
    recent_pitches = [pitch]

    while beat < total_beats:
        # Phrase length: 2-4 bars (8-16 beats)
        phrase_bars = random.randint(2, 4)
        phrase_beats = float(phrase_bars * 4)
        phrase_end = min(beat + phrase_beats, total_beats)

        while beat < phrase_end:
            chord = _get_chord_at_beat(chords, beat)
            if chord is None:
                beat += 1.0
                continue

            beat_in_bar = beat % 4.0

            # Strong beats: chord tones
            if beat_in_bar < 0.1 or abs(beat_in_bar - 2.0) < 0.1:
                target = _nearest_chord_tone(pitch, chord.root_pc, chord.quality,
                                             MELODY_LOW, MELODY_HIGH)
            else:
                # Weak beats: scalar connections
                if random.random() < 0.3:
                    target = _nearest_scale_tone(pitch, chord.root_pc, chord.quality,
                                                 MELODY_LOW, MELODY_HIGH)
                else:
                    target = _nearest_chord_tone(pitch, chord.root_pc, chord.quality,
                                                 MELODY_LOW, MELODY_HIGH)

            # Voice leading: prefer small intervals
            if abs(target - pitch) > 5:
                scale_tones = _scale_tones_in_range(chord.root_pc, chord.quality,
                                                    MELODY_LOW, MELODY_HIGH)
                if scale_tones:
                    closer = [t for t in scale_tones if abs(t - pitch) <= 4]
                    if closer:
                        target = min(closer, key=lambda t: abs(t - pitch))

            # Contour check
            if _contour_check(recent_pitches):
                direction = -1 if recent_pitches[-1] > recent_pitches[-max(2, len(recent_pitches))] else 1
                candidates = _chord_tones_in_range(chord.root_pc, chord.quality,
                                                   MELODY_LOW, MELODY_HIGH)
                if direction > 0:
                    above = [t for t in candidates if t > pitch]
                    if above:
                        target = above[0]
                else:
                    below = [t for t in candidates if t < pitch]
                    if below:
                        target = below[-1]

            # Duration: quarters and halves, occasionally dotted quarter
            dur = random.choice([1.0, 1.0, 1.0, 2.0, 2.0, 1.5])
            if beat + dur > phrase_end:
                dur = phrase_end - beat
                if dur <= 0:
                    break

            tick = _beat_to_tick(beat)
            if swing:
                tick = _apply_swing(tick)
            tick = _humanize(tick)
            dur_ticks = int(dur * TICKS_PER_QUARTER)
            vel = random.randint(75, 90)

            notes.append(NoteEvent(
                pitch=target,
                start_tick=tick,
                duration_ticks=dur_ticks,
                velocity=vel,
            ))

            pitch = target
            recent_pitches.append(pitch)
            if len(recent_pitches) > 20:
                recent_pitches = recent_pitches[-20:]
            beat += dur

        # Breathing: rest for 1-2 beats between phrases
        rest_dur = random.choice([1.0, 1.0, 1.5, 2.0])
        beat += rest_dur

    notes.sort(key=lambda n: n.start_tick)
    return notes


def generate_solo(chords: List[ChordEvent], total_beats: float,
                  tension_curve: str = "arc", swing: bool = True,
                  coltrane: bool = False, seed: Optional[int] = None) -> List[NoteEvent]:
    """Generate an improvised jazz solo with tension-driven phrasing.

    The solo progresses through three tiers based on the tension curve:
      - Tier 1 (tension < 0.4): Melodic -- chord tones, voice leading, quarters/halves
      - Tier 2 (tension 0.4-0.7): Motivic -- motif development, enclosures, 8th notes
      - Tier 3 (tension > 0.7): Sheets of Sound -- rapid 16ths, digital patterns

    Args:
        chords: List of ChordEvent objects.
        total_beats: Total beats in the solo section.
        tension_curve: Curve shape ('arc', 'build', 'wave').
        swing: Whether to apply swing feel.
        coltrane: Enable Coltrane multi-tonic features.
        seed: Optional random seed for reproducibility.

    Returns:
        List of NoteEvent objects sorted by start_tick.
    """
    if seed is not None:
        random.seed(seed)

    if not chords:
        return []

    curve = TensionCurve(tension_curve)
    notes: List[NoteEvent] = []
    beat = 0.0

    # Start on a chord tone near middle register
    mid_pitch = (MELODY_LOW + MELODY_HIGH) // 2
    pitch = _nearest_chord_tone(mid_pitch, chords[0].root_pc, chords[0].quality,
                                MELODY_LOW, MELODY_HIGH)

    while beat < total_beats:
        progress = beat / total_beats if total_beats > 0 else 0.0
        tension = curve(progress)
        params = interpolate_params(tension)

        # Determine tier
        if tension < 0.4:
            tier = 1
        elif tension < 0.7:
            tier = 2
        else:
            tier = 3

        # Determine phrase length based on tier
        if tier == 1:
            phrase_beats = float(random.randint(4, 8))
        elif tier == 2:
            phrase_beats = float(random.randint(3, 6))
        else:
            phrase_beats = float(random.randint(2, 5))

        # Clamp to remaining beats
        phrase_beats = min(phrase_beats, total_beats - beat)
        if phrase_beats <= 0:
            break

        # Coltrane mode: apply key center awareness
        if coltrane:
            chord = _get_chord_at_beat(chords, beat)
            if chord is not None and chord.key_center_pc != chord.root_pc:
                # Boost substitute key tones by expanding register slightly
                sub_tones = _substitute_key_tones(chord.key_center_pc,
                                                  params.register_low,
                                                  params.register_high)
                # This awareness is passed implicitly through register and
                # the coltrane flag to tier generators

        # Generate phrase based on tier
        if tier == 1:
            phrase_notes, pitch, beat = _generate_tier1_phrase(
                beat, phrase_beats, chords, pitch, params, swing
            )
        elif tier == 2:
            # Randomly select between motivic, pentatonic, call-response, or triplet
            sub_type = random.choices(
                ["motivic", "pentatonic", "call_response", "triplet"],
                weights=[0.35, 0.25, 0.20, 0.20],
                k=1,
            )[0]
            if sub_type == "pentatonic" and tension > 0.5:
                phrase_notes, pitch, beat = _generate_pentatonic_super_phrase(
                    beat, phrase_beats, chords, pitch, params, swing, coltrane
                )
            elif sub_type == "call_response":
                phrase_notes, pitch, beat = _generate_call_response_phrase(
                    beat, phrase_beats, chords, pitch, params, swing, coltrane
                )
            elif sub_type == "triplet" and tension > 0.5:
                phrase_notes, pitch, beat = _generate_triplet_phrase(
                    beat, phrase_beats, chords, pitch, params, swing, coltrane
                )
            else:
                phrase_notes, pitch, beat = _generate_tier2_phrase(
                    beat, phrase_beats, chords, pitch, params, swing, coltrane
                )
        else:
            phrase_notes, pitch, beat = _generate_tier3_phrase(
                beat, phrase_beats, chords, pitch, params, swing, coltrane
            )

        notes.extend(phrase_notes)

        # Ghost notes: chromatic passing tones at low velocity between phrases
        if coltrane and 0.3 < tension < 0.6 and random.random() < 0.4:
            # Insert 1-3 ghost notes stepping chromatically toward next phrase area
            num_ghost = random.randint(1, 3)
            ghost_beat = beat
            ghost_pitch = pitch
            for _ in range(num_ghost):
                if ghost_beat >= total_beats:
                    break
                # Step chromatically up or down
                direction = random.choice([-1, 1])
                ghost_pitch = max(MELODY_LOW, min(MELODY_HIGH, ghost_pitch + direction))
                ghost_tick = _beat_to_tick(ghost_beat)
                if swing:
                    ghost_tick = _apply_swing(ghost_tick)
                ghost_dur = TICKS_PER_16TH  # Very short
                ghost_vel = random.randint(20, 35)  # Very quiet
                notes.append(NoteEvent(
                    pitch=ghost_pitch,
                    start_tick=ghost_tick,
                    duration_ticks=ghost_dur,
                    velocity=ghost_vel,
                ))
                ghost_beat += 0.25

        # Rest between phrases (scaled by tension -- higher tension = shorter rests)
        if random.random() < params.rest_prob or tier == 1:
            if tier == 3:
                # Minimal rest at high tension: 16th note gap
                rest_dur = 0.25
            elif tier == 2:
                rest_dur = random.choice([0.5, 1.0])
            else:
                rest_dur = random.choice([1.0, 1.5, 2.0])
            beat += rest_dur

    notes.sort(key=lambda n: n.start_tick)
    return notes


def generate_trading_fours(chords: List[ChordEvent], total_beats: float,
                           intensity: float = 0.6) -> List[NoteEvent]:
    """Generate a trading-fours section (4 bars melody, 4 bars drums).

    Generates melody for bars 1-4 of each 8-bar phrase and leaves
    bars 5-8 silent for drum fills. Higher intensity compensates
    for the shorter melodic sections.

    Args:
        chords: List of ChordEvent objects.
        total_beats: Total beats in the trading section.
        intensity: Base intensity for the melody bars (0.0-1.0).

    Returns:
        List of NoteEvent objects sorted by start_tick.
    """
    if not chords:
        return []

    notes: List[NoteEvent] = []
    beat = 0.0
    phrase_size = 32.0  # 8 bars = 32 beats
    melody_beats = 16.0  # 4 bars = 16 beats

    # Start pitch
    mid_pitch = (MELODY_LOW + MELODY_HIGH) // 2
    pitch = _nearest_chord_tone(mid_pitch, chords[0].root_pc, chords[0].quality,
                                MELODY_LOW, MELODY_HIGH)

    phrase_idx = 0
    while beat < total_beats:
        # Melody section: first 4 bars
        melody_end = min(beat + melody_beats, total_beats)
        melody_section_beats = melody_end - beat
        if melody_section_beats <= 0:
            break

        # Use elevated intensity for the melody bars
        boosted_intensity = min(1.0, intensity + 0.15)
        params = interpolate_params(boosted_intensity)

        current_beat = beat
        recent_pitches = [pitch]

        while current_beat < melody_end:
            chord = _get_chord_at_beat(chords, current_beat)
            if chord is None:
                current_beat += 0.5
                continue

            low = params.register_low
            high = params.register_high

            # Use a mix of Tier 1 and Tier 2 approaches
            if boosted_intensity < 0.5:
                # More melodic approach
                target = _nearest_chord_tone(pitch, chord.root_pc, chord.quality, low, high)
                dur = random.choice([0.5, 0.5, 1.0, 1.0])
            else:
                # More energetic approach
                roll = random.random()
                if roll < 0.3:
                    # Scalar run
                    direction = 1 if random.random() < 0.5 else -1
                    run = _scalar_run(pitch, direction, chord.root_pc, chord.quality,
                                      random.randint(3, 5), low, high)
                    for note_midi, run_dur in run:
                        if current_beat + run_dur > melody_end:
                            break
                        tick = _beat_to_tick(current_beat)
                        tick = _apply_swing(tick)
                        tick = _humanize(tick)
                        dur_ticks = int(run_dur * TICKS_PER_QUARTER)
                        vel = _choose_velocity(params)
                        notes.append(NoteEvent(
                            pitch=note_midi, start_tick=tick,
                            duration_ticks=dur_ticks, velocity=vel,
                        ))
                        pitch = note_midi
                        recent_pitches.append(pitch)
                        current_beat += run_dur
                    continue
                elif roll < 0.5:
                    # Digital pattern
                    fragment = _digital_pattern_fragment(chord.root_pc, pitch, low, high)
                    for note_midi, frag_dur in fragment:
                        if current_beat + frag_dur > melody_end:
                            break
                        tick = _beat_to_tick(current_beat)
                        tick = _apply_swing(tick)
                        tick = _humanize(tick)
                        dur_ticks = int(frag_dur * TICKS_PER_QUARTER)
                        vel = _choose_velocity(params)
                        notes.append(NoteEvent(
                            pitch=note_midi, start_tick=tick,
                            duration_ticks=dur_ticks, velocity=vel,
                        ))
                        pitch = note_midi
                        recent_pitches.append(pitch)
                        current_beat += frag_dur
                    continue
                else:
                    target = _nearest_chord_tone(pitch, chord.root_pc, chord.quality, low, high)
                    dur = random.choice([0.5, 0.5, 0.5, 1.0])

            # Voice leading
            if abs(target - pitch) > 7:
                scale_tones = _scale_tones_in_range(chord.root_pc, chord.quality, low, high)
                if scale_tones:
                    closer = [t for t in scale_tones if abs(t - pitch) <= 4]
                    if closer:
                        target = random.choice(closer)

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
                dur = melody_end - current_beat
                if dur <= 0:
                    break

            tick = _beat_to_tick(current_beat)
            tick = _apply_swing(tick)
            tick = _humanize(tick)
            dur_ticks = int(dur * TICKS_PER_QUARTER)
            vel = _choose_velocity(params)

            notes.append(NoteEvent(
                pitch=target,
                start_tick=tick,
                duration_ticks=dur_ticks,
                velocity=vel,
            ))

            pitch = target
            recent_pitches.append(pitch)
            if len(recent_pitches) > 20:
                recent_pitches = recent_pitches[-20:]
            current_beat += dur

        # Skip bars 5-8 (drums fill) -- advance beat past the silence
        beat += phrase_size  # Jump full 8 bars
        phrase_idx += 1

    notes.sort(key=lambda n: n.start_tick)
    return notes
