"""Piano comping generator for Coltrain.

Generates idiomatic jazz piano voicings with rhythmic comping patterns,
voice leading, and intensity-dependent pattern selection.
"""

import math
import random
from typing import List, Optional, Tuple

from . import NoteEvent, TICKS_PER_QUARTER, TICKS_PER_BAR, TICKS_PER_8TH, TICKS_PER_16TH

# ---------------------------------------------------------------------------
# Chord tone intervals (self-contained)
# ---------------------------------------------------------------------------

CHORD_TONES = {
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "dom7": (0, 4, 7, 10),
    "7": (0, 4, 7, 10),
    "min7b5": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9),
    "aug7": (0, 4, 8, 10),
    "minmaj7": (0, 3, 7, 11),
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus4": (0, 5, 7, 10),
    "sus2": (0, 2, 7, 10),
    "min6": (0, 3, 7, 9),
    "6": (0, 4, 7, 9),
}

# Extensions to add to voicings when quality supports them
EXTENSIONS = {
    "maj7": (14,),         # 9th
    "min7": (14,),         # 9th
    "dom7": (14,),         # 9th
    "7": (14,),            # 9th
    "min7b5": (14,),       # 9th (b9 sometimes, but we keep it diatonic)
    "dim7": (),
    "aug7": (14,),         # 9th
    "minmaj7": (14,),      # 9th
    "maj": (),
    "min": (),
    "dim": (),
    "aug": (),
    "sus4": (14,),         # 9th
    "sus2": (),
    "min6": (14,),         # 9th
    "6": (14,),            # 9th
}

# Piano comping range
COMP_LOW = 48    # C3
COMP_HIGH = 72   # C5

# ---------------------------------------------------------------------------
# Rhythmic patterns
# Each pattern is a list of (beat_offset_within_bar, duration_in_beats)
# ---------------------------------------------------------------------------

COMPING_PATTERNS = [
    # Pattern 0 - Basic 2&4
    [(1.0, 1.0), (3.0, 1.0)],
    # Pattern 1 - Charleston
    [(0.0, 1.0), (1.5, 0.5)],
    # Pattern 2 - Anticipation
    [(0.0, 1.5), (2.5, 1.5)],
    # Pattern 3 - Syncopated
    [(0.5, 1.0), (2.0, 0.5), (3.5, 0.5)],
    # Pattern 4 - Sparse
    [(1.0, 2.0)],
    # Pattern 5 - Dense
    [(0.0, 0.5), (1.0, 0.5), (2.0, 0.5), (3.0, 0.5)],
    # Pattern 6 - Freddie Green (4-on-the-floor quarters)
    [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0)],
    # Pattern 7 - Anticipation push (hits before barline area)
    [(3.5, 1.5), (1.5, 0.5)],
    # Pattern 8 - Dotted rhythm
    [(0.0, 1.5), (1.5, 0.5), (3.0, 1.0)],
    # Pattern 9 - Space (single hit on beat 3)
    [(2.0, 2.0)],
]

# Which patterns to use at each intensity range
PATTERNS_BY_INTENSITY = {
    "low": [0, 4, 9],           # sparse, spacious
    "medium": [0, 1, 2, 7, 8],  # standard, varied
    "high": [1, 3, 5, 6, 8],    # dense, syncopated, driving
}


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _swing_tick(bar_start_tick: int, beat_offset: float, swing: bool) -> int:
    """Convert beat offset within a bar to absolute tick with swing.

    Swing shifts offbeat 8ths (beat + 0.5) to the 2/3 position.
    """
    beat_num = int(beat_offset)
    frac = beat_offset - beat_num

    if swing and abs(frac - 0.5) < 0.01:
        # Offbeat 8th — shift to 2/3 of the beat (triplet feel)
        tick = bar_start_tick + beat_num * TICKS_PER_QUARTER + int(TICKS_PER_QUARTER * 2 / 3)
    else:
        tick = bar_start_tick + int(beat_offset * TICKS_PER_QUARTER)

    return tick


def _humanize_tick(tick: int, amount: int = 10) -> int:
    """Add slight random timing variation."""
    return max(0, tick + random.randint(-amount, amount))


def _humanize_velocity(vel: int, amount: int = 5) -> int:
    """Add slight random velocity variation, clamped to 1-127."""
    return max(1, min(127, vel + random.randint(-amount, amount)))


# ---------------------------------------------------------------------------
# Voicing builders
# ---------------------------------------------------------------------------


def _get_intervals(quality: str) -> Tuple[int, ...]:
    """Get chord tone intervals for a quality, with fallback."""
    return CHORD_TONES.get(quality, (0, 4, 7))


def _get_3rd(quality: str) -> int:
    """Return the 3rd interval for a chord quality."""
    intervals = _get_intervals(quality)
    for iv in intervals:
        if 3 <= iv <= 5:  # minor 3rd, major 3rd, or sus4 (5)
            return iv
    return 4  # default major 3rd


def _get_5th(quality: str) -> int:
    """Return the 5th interval for a chord quality."""
    intervals = _get_intervals(quality)
    for iv in intervals:
        if 6 <= iv <= 8:  # dim5, P5, aug5
            return iv
    return 7  # default perfect 5th


def _get_7th(quality: str) -> int:
    """Return the 7th interval for a chord quality (or best substitute)."""
    intervals = _get_intervals(quality)
    for iv in intervals:
        if 9 <= iv <= 11:  # 6th, min7th, maj7th
            return iv
    # No 7th in chord — triads. Return 10 (min7) as default extension
    return 10


def _get_9th(quality: str) -> int:
    """Return the 9th (as semitones from root, mod 12 = 2 usually)."""
    exts = EXTENSIONS.get(quality, ())
    if 14 in exts:
        return 14 % 12  # = 2
    return 2  # natural 9th


def _build_voicing(root_pc: int, quality: str, voicing_type: str,
                    center_midi: int = 60) -> List[int]:
    """Build a chord voicing as a list of MIDI pitches.

    Args:
        root_pc: Root pitch class (0-11).
        quality: Chord quality string.
        voicing_type: One of 'shell', 'rootless_a', 'rootless_b', 'drop2'.
        center_midi: Target center of the voicing for octave placement.

    Returns:
        Sorted list of MIDI pitches within COMP_LOW-COMP_HIGH.
    """
    third = _get_3rd(quality)
    fifth = _get_5th(quality)
    seventh = _get_7th(quality)
    ninth = _get_9th(quality)

    if voicing_type == "shell":
        # Root + 3rd + 7th (3 notes)
        pcs = [0, third, seventh]
    elif voicing_type == "rootless_a":
        # 3rd + 5th + 7th + 9th (4 notes, no root)
        pcs = [third, fifth, seventh, ninth]
    elif voicing_type == "rootless_b":
        # 7th + 9th + 3rd + 5th (4 notes, no root, different ordering/inversion)
        pcs = [seventh, ninth, third, fifth]
    elif voicing_type == "drop2":
        # Close position (root, 3rd, 5th, 7th) but 2nd-from-top dropped an octave
        # Close position bottom-up: root, 3rd, 5th, 7th
        # 2nd from top is 5th, drop it an octave
        pcs = [0, third, fifth, seventh]
    elif voicing_type == "quartal":
        # McCoy Tyner-style stacked perfect 4ths from the chord's 3rd
        # Creates open, modern voicings characteristic of Coltrane quartet
        pcs = [third, third + 5, third + 10, third + 15]
    else:
        pcs = [0, third, seventh]

    # Convert pitch classes to MIDI notes near center_midi
    midi_notes = []
    for pc_offset in pcs:
        target_pc = (root_pc + pc_offset) % 12
        # Find the nearest MIDI note with this pitch class to center_midi
        candidates = []
        for midi in range(COMP_LOW, COMP_HIGH + 1):
            if midi % 12 == target_pc:
                candidates.append(midi)
        if candidates:
            best = min(candidates, key=lambda m: abs(m - center_midi))
            midi_notes.append(best)

    # For drop2: take the 2nd-from-top note and drop it an octave
    if voicing_type == "drop2" and len(midi_notes) >= 4:
        midi_notes.sort()
        # 2nd from top
        second_from_top = midi_notes[-2]
        dropped = second_from_top - 12
        if dropped >= COMP_LOW:
            midi_notes[-2] = dropped
        midi_notes.sort()

    # Remove duplicates and sort
    midi_notes = sorted(set(midi_notes))

    # Ensure all notes are in range
    midi_notes = [n for n in midi_notes if COMP_LOW <= n <= COMP_HIGH]

    return midi_notes


def _voice_leading_cost(voicing_a: List[int], voicing_b: List[int]) -> int:
    """Calculate the total semitone distance between two voicings.

    Uses minimum-distance matching between notes. Lower cost = smoother
    voice leading.
    """
    if not voicing_a or not voicing_b:
        return 0

    # Simple approach: sum of distances between corresponding notes
    # Pad shorter voicing if needed
    a = sorted(voicing_a)
    b = sorted(voicing_b)

    # Match by position (bottom-up pairing)
    total = 0
    min_len = min(len(a), len(b))
    for i in range(min_len):
        total += abs(a[i] - b[i])

    # Penalize different voicing sizes
    total += abs(len(a) - len(b)) * 6

    return total


def _smooth_voicing(root_pc: int, quality: str, prev_voicing: List[int],
                     voicing_type: str) -> List[int]:
    """Build a voicing that voice-leads smoothly from prev_voicing.

    Tries multiple center points and picks the one with lowest voice-leading cost.

    Args:
        root_pc: Root pitch class.
        quality: Chord quality.
        prev_voicing: Previous voicing's MIDI pitches.
        voicing_type: Voicing type string.

    Returns:
        Best-voiced list of MIDI pitches.
    """
    if not prev_voicing:
        return _build_voicing(root_pc, quality, voicing_type, center_midi=60)

    prev_center = sum(prev_voicing) // len(prev_voicing)

    # Try building voicings centered at different positions
    best_voicing = None
    best_cost = float("inf")

    for center_offset in range(-6, 7, 2):
        candidate_center = prev_center + center_offset
        candidate = _build_voicing(root_pc, quality, voicing_type, candidate_center)
        if not candidate:
            continue
        cost = _voice_leading_cost(prev_voicing, candidate)
        if cost < best_cost:
            best_cost = cost
            best_voicing = candidate

    if best_voicing is None:
        return _build_voicing(root_pc, quality, voicing_type, center_midi=60)

    return best_voicing


# ---------------------------------------------------------------------------
# Chord lookup by beat
# ---------------------------------------------------------------------------


def _chord_at_beat(chords, beat: float):
    """Return the chord active at the given beat."""
    for chord in chords:
        if chord.start_beat <= beat < chord.end_beat:
            return chord
    return chords[-1] if chords else None


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate_comping(chords, total_beats: int, intensity: float = 0.5,
                      swing: bool = True, coltrane: bool = False) -> List[NoteEvent]:
    """Generate piano comping voicings over a chord progression.

    Args:
        chords: List of ChordEvent objects.
        total_beats: Total number of beats to generate.
        intensity: Comping intensity 0.0-1.0. Affects pattern density,
                   voicing type, and velocity.
        swing: Whether to apply swing feel.

    Returns:
        List of NoteEvent objects (multiple notes per chord hit).
    """
    if not chords:
        return []

    intensity = max(0.0, min(1.0, intensity))
    total_bars = total_beats // 4
    notes: List[NoteEvent] = []

    # Select voicing type based on intensity
    if intensity < 0.3:
        voicing_type = "shell"
    elif intensity < 0.7:
        if coltrane and intensity > 0.5:
            voicing_type = random.choice(["rootless_a", "rootless_b", "quartal"])
        else:
            voicing_type = random.choice(["rootless_a", "rootless_b"])
    else:
        if coltrane:
            voicing_type = random.choice(["drop2", "quartal"])
        else:
            voicing_type = "drop2"

    # Select pattern pool based on intensity
    if intensity < 0.3:
        pattern_pool = PATTERNS_BY_INTENSITY["low"]
    elif intensity <= 0.6:
        pattern_pool = PATTERNS_BY_INTENSITY["medium"]
    else:
        pattern_pool = PATTERNS_BY_INTENSITY["high"]

    prev_voicing: List[int] = []
    prev_pattern_idx = -1
    pattern_hold_bars = 0  # How many more bars to hold the current pattern
    phrase_length = random.choice([4, 6, 8])  # Bars per dynamic phrase

    for bar_idx in range(total_bars):
        bar_start_beat = bar_idx * 4.0
        bar_start_tick = bar_idx * TICKS_PER_BAR

        # Lay-out (rest) bars: skip this bar entirely for breathing room
        if intensity < 0.4:
            layoff_prob = 0.28
        elif intensity < 0.7:
            layoff_prob = 0.12
        else:
            layoff_prob = 0.05
        if random.random() < layoff_prob:
            continue

        # Re-evaluate voicing type every 8 bars for gradual evolution
        if bar_idx > 0 and bar_idx % 8 == 0:
            if intensity < 0.3:
                voicing_type = "shell"
            elif intensity < 0.7:
                if coltrane and intensity > 0.5:
                    voicing_type = random.choice(["rootless_a", "rootless_b", "quartal"])
                else:
                    voicing_type = random.choice(["rootless_a", "rootless_b"])
            else:
                if coltrane:
                    voicing_type = random.choice(["drop2", "quartal"])
                else:
                    voicing_type = "drop2"

        # Choose a rhythm pattern — phrase continuity (hold for 2-3 bars)
        if pattern_hold_bars > 0:
            pattern_hold_bars -= 1
            pattern_idx = prev_pattern_idx
        else:
            available = [p for p in pattern_pool if p != prev_pattern_idx]
            if not available:
                available = pattern_pool
            pattern_idx = random.choice(available)
            prev_pattern_idx = pattern_idx
            if random.random() < 0.40:
                pattern_hold_bars = random.randint(1, 2)
        pattern = COMPING_PATTERNS[pattern_idx]

        # Phrase-level dynamic contour
        phrase_pos = (bar_idx % phrase_length) / max(1, phrase_length - 1)
        vel_phrase_mult = 0.85 + 0.30 * math.sin(phrase_pos * math.pi)

        for beat_offset, duration_beats in pattern:
            abs_beat = bar_start_beat + beat_offset

            chord = _chord_at_beat(chords, abs_beat)
            if chord is None:
                continue

            # Build voicing with voice leading from previous
            voicing = _smooth_voicing(
                chord.root_pc, chord.quality, prev_voicing, voicing_type
            )
            if not voicing:
                continue
            prev_voicing = voicing

            # Calculate tick position with swing
            tick = _swing_tick(bar_start_tick, beat_offset, swing)
            tick = _humanize_tick(tick, amount=20 if swing else 8)

            # Duration in ticks
            dur_ticks = int(duration_beats * TICKS_PER_QUARTER)
            # Slight shortening for articulation
            dur_ticks = max(TICKS_PER_16TH, dur_ticks - 30)

            # Velocity: intensity-scaled ranges with phrase arc
            if intensity < 0.3:
                vel_lo, vel_hi = 50, 65
            elif intensity < 0.6:
                vel_lo, vel_hi = 60, 75
            else:
                vel_lo, vel_hi = 70, 90

            beat_in_bar = beat_offset
            if abs(beat_in_bar - 1.0) < 0.1 or abs(beat_in_bar - 3.0) < 0.1:
                # On beat 2 or 4 — accent
                base_vel = random.randint(vel_lo + 5, vel_hi)
            elif abs(beat_in_bar - 0.0) < 0.1 or abs(beat_in_bar - 2.0) < 0.1:
                # On beat 1 or 3 — softer
                base_vel = random.randint(max(1, vel_lo - 5), max(2, vel_hi - 8))
            else:
                # Syncopated position
                base_vel = random.randint(vel_lo, max(vel_lo + 1, vel_hi - 3))

            # Apply phrase-level velocity arc
            base_vel = max(1, min(127, round(base_vel * vel_phrase_mult)))

            # Emit one NoteEvent per note in the voicing, with chord spread
            # Stagger notes bottom-to-top to simulate hand roll
            spread_amount = random.randint(0, 18) if len(voicing) > 1 else 0
            for note_idx, midi_pitch in enumerate(sorted(voicing)):
                note_tick = tick + (note_idx * spread_amount // max(1, len(voicing) - 1))
                vel = _humanize_velocity(base_vel, amount=4)
                notes.append(NoteEvent(
                    pitch=midi_pitch,
                    start_tick=note_tick,
                    duration_ticks=dur_ticks,
                    velocity=vel,
                    channel=0,
                ))

    return notes
