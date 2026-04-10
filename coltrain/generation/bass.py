"""Walking bass line generator for Coltrain.

Generates idiomatic jazz walking bass lines using chord-tone targeting,
chromatic approach notes, and voice-leading principles.
"""

import random
from typing import List, Optional

from . import NoteEvent, TICKS_PER_QUARTER, TICKS_PER_BAR, TICKS_PER_8TH

# ---------------------------------------------------------------------------
# Chord tone intervals (self-contained — mirrors theory.chord.CHORD_TONES)
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

# Scale tones per quality — used for scalar passing tones
# These are the full scales typically associated with each quality.
SCALE_TONES = {
    "maj7": (0, 2, 4, 5, 7, 9, 11),       # Ionian
    "min7": (0, 2, 3, 5, 7, 9, 10),        # Dorian
    "dom7": (0, 2, 4, 5, 7, 9, 10),        # Mixolydian
    "7": (0, 2, 4, 5, 7, 9, 10),           # Mixolydian
    "min7b5": (0, 2, 3, 5, 6, 8, 10),      # Locrian
    "dim7": (0, 2, 3, 5, 6, 8, 9, 11),     # Diminished (whole-half)
    "aug7": (0, 2, 4, 6, 8, 10),           # Whole tone
    "minmaj7": (0, 2, 3, 5, 7, 9, 11),     # Melodic minor
    "maj": (0, 2, 4, 5, 7, 9, 11),         # Ionian
    "min": (0, 2, 3, 5, 7, 9, 10),         # Dorian
    "dim": (0, 2, 3, 5, 6, 8, 9, 11),      # Diminished
    "aug": (0, 2, 4, 6, 8, 10),            # Whole tone
    "sus4": (0, 2, 4, 5, 7, 9, 10),        # Mixolydian
    "sus2": (0, 2, 4, 5, 7, 9, 10),        # Mixolydian
    "min6": (0, 2, 3, 5, 7, 9, 10),        # Dorian
    "6": (0, 2, 4, 5, 7, 9, 11),           # Ionian
}

# Bass range limits
BASS_LOW = 28   # E1
BASS_HIGH = 55  # G3


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _nearest_bass_note(target_pc: int, current_midi: int,
                       low: int = BASS_LOW, high: int = BASS_HIGH) -> int:
    """Find the nearest MIDI note with the given pitch class to current_midi,
    within the bass range [low, high].

    Args:
        target_pc: Target pitch class (0-11).
        current_midi: Current MIDI note to stay close to.
        low: Lowest allowed MIDI note.
        high: Highest allowed MIDI note.

    Returns:
        MIDI note number.
    """
    candidates = []
    for midi_note in range(low, high + 1):
        if midi_note % 12 == target_pc:
            candidates.append(midi_note)

    if not candidates:
        # Fallback: just clamp the obvious candidate
        base = (current_midi // 12) * 12 + target_pc
        return max(low, min(high, base))

    # Pick the candidate closest to current_midi
    return min(candidates, key=lambda n: abs(n - current_midi))


def _chromatic_approach(target_midi: int) -> int:
    """Return a chromatic approach note to target_midi (one semitone above or below).

    Chosen randomly. Clamped to bass range.
    """
    direction = random.choice([-1, 1])
    approach = target_midi + direction
    return max(BASS_LOW, min(BASS_HIGH, approach))


def _chord_tones_in_range(root_pc: int, quality: str,
                           low: int = BASS_LOW, high: int = BASS_HIGH) -> List[int]:
    """Return a sorted list of all MIDI notes that are chord tones of the given
    chord within [low, high] inclusive.
    """
    intervals = CHORD_TONES.get(quality, (0, 4, 7))
    result = []
    for midi_note in range(low, high + 1):
        pc = midi_note % 12
        interval = (pc - root_pc) % 12
        if interval in intervals:
            result.append(midi_note)
    return sorted(result)


def _scale_tones_in_range(root_pc: int, quality: str,
                           low: int = BASS_LOW, high: int = BASS_HIGH) -> List[int]:
    """Return all MIDI notes from the chord's associated scale within range."""
    intervals = SCALE_TONES.get(quality, (0, 2, 4, 5, 7, 9, 11))
    result = []
    for midi_note in range(low, high + 1):
        pc = midi_note % 12
        interval = (pc - root_pc) % 12
        if interval in intervals:
            result.append(midi_note)
    return sorted(result)


def _scale_step_toward(current_midi: int, target_midi: int,
                       root_pc: int, quality: str) -> int:
    """Take one scale step from current_midi toward target_midi.

    Uses the scale tones of the chord quality.
    """
    scale_notes = _scale_tones_in_range(root_pc, quality)
    if not scale_notes:
        # Fallback: chromatic step
        return current_midi + (1 if target_midi > current_midi else -1)

    if target_midi > current_midi:
        # Step up: find the next scale note above current
        candidates = [n for n in scale_notes if n > current_midi]
        if candidates:
            return candidates[0]
        return current_midi + 1
    elif target_midi < current_midi:
        # Step down: find the next scale note below current
        candidates = [n for n in scale_notes if n < current_midi]
        if candidates:
            return candidates[-1]
        return current_midi - 1
    else:
        return current_midi


def _clamp(value: int, low: int = BASS_LOW, high: int = BASS_HIGH) -> int:
    """Clamp a MIDI note to the bass range."""
    return max(low, min(high, value))


def _apply_swing_tick(beat_in_bar: int, bar_start_tick: int, swing: bool) -> int:
    """Convert a beat number (0-3) within a bar to an absolute tick position,
    applying swing feel to offbeat 8ths.

    For walking bass, every beat is on a downbeat, so swing mostly affects
    the feel subtly. The real swing is on 8th-note subdivisions.
    """
    tick = bar_start_tick + beat_in_bar * TICKS_PER_QUARTER
    return tick


def _humanize_tick(tick: int, amount: int = 10) -> int:
    """Add slight random timing variation."""
    return max(0, tick + random.randint(-amount, amount))


def _humanize_velocity(vel: int, amount: int = 5) -> int:
    """Add slight random velocity variation, clamped to 1-127."""
    return max(1, min(127, vel + random.randint(-amount, amount)))


# ---------------------------------------------------------------------------
# Chord lookup by beat position
# ---------------------------------------------------------------------------


def _chord_at_beat(chords, beat: float):
    """Return the chord active at the given beat, or the last chord."""
    for i, chord in enumerate(chords):
        if chord.start_beat <= beat < chord.end_beat:
            return chord
    # Past last chord — return the last one
    return chords[-1] if chords else None


def _next_chord_at_bar(chords, bar_start_beat: float):
    """Return the chord active at the start of the NEXT bar (bar_start_beat + 4).
    If none, return the current chord.
    """
    next_bar_beat = bar_start_beat + 4.0
    return _chord_at_beat(chords, next_bar_beat)


# ---------------------------------------------------------------------------
# Main generators
# ---------------------------------------------------------------------------


def generate_walking_bass(chords, total_beats: int, swing: bool = True) -> List[NoteEvent]:
    """Generate a walking bass line over the given chord progression.

    Args:
        chords: List of ChordEvent objects.
        total_beats: Total number of beats to generate.
        swing: Whether to apply swing feel (affects humanization).

    Returns:
        List of NoteEvent objects forming the bass line.
    """
    if not chords:
        return []

    notes: List[NoteEvent] = []
    total_bars = total_beats // 4

    # Start on the root of the first chord, in a comfortable range
    first_chord = chords[0]
    current_midi = _nearest_bass_note(first_chord.root_pc, 40)  # Start around E2

    for bar_idx in range(total_bars):
        bar_start_beat = bar_idx * 4.0
        bar_start_tick = bar_idx * TICKS_PER_BAR

        chord = _chord_at_beat(chords, bar_start_beat)
        if chord is None:
            continue

        # Get the chord active at beat 3 (might be different if chord changes mid-bar)
        chord_beat3 = _chord_at_beat(chords, bar_start_beat + 2.0)
        if chord_beat3 is None:
            chord_beat3 = chord

        # Look ahead to the next bar's chord for approach note on beat 4
        next_chord = _next_chord_at_bar(chords, bar_start_beat)

        root_pc = chord.root_pc
        quality = chord.quality

        chord_tone_list = _chord_tones_in_range(root_pc, quality)
        if not chord_tone_list:
            chord_tone_list = [_nearest_bass_note(root_pc, current_midi)]

        # ---- BEAT 1: Root (80%), 5th (15%), or 3rd (5%) ----
        r = random.random()
        if r < 0.80:
            beat1_pc = root_pc
        elif r < 0.95:
            # 5th
            intervals = CHORD_TONES.get(quality, (0, 4, 7))
            fifth_interval = 7  # Perfect 5th
            # Find the actual 5th in the chord tones
            for iv in intervals:
                if 6 <= iv <= 8:  # tritone, P5, or aug5
                    fifth_interval = iv
                    break
            beat1_pc = (root_pc + fifth_interval) % 12
        else:
            # 3rd
            intervals = CHORD_TONES.get(quality, (0, 4, 7))
            third_interval = 4  # Major 3rd default
            for iv in intervals:
                if 3 <= iv <= 4:  # minor or major 3rd
                    third_interval = iv
                    break
            beat1_pc = (root_pc + third_interval) % 12

        beat1_midi = _nearest_bass_note(beat1_pc, current_midi)
        # Voice leading constraint: keep within +-7 semitones of previous note
        if abs(beat1_midi - current_midi) > 7:
            beat1_midi = _nearest_bass_note(beat1_pc, current_midi)
            # If still too far, try adjusting octave
            if beat1_midi - current_midi > 7:
                alt = beat1_midi - 12
                if alt >= BASS_LOW:
                    beat1_midi = alt
            elif current_midi - beat1_midi > 7:
                alt = beat1_midi + 12
                if alt <= BASS_HIGH:
                    beat1_midi = alt

        tick1 = _humanize_tick(bar_start_tick, amount=8 if swing else 3)
        vel1 = _humanize_velocity(random.randint(80, 100))
        notes.append(NoteEvent(
            pitch=beat1_midi,
            start_tick=tick1,
            duration_ticks=TICKS_PER_QUARTER - 20,  # Slight space between notes
            velocity=vel1,
            channel=0,
        ))

        # ---- BEAT 2: Chord tone (40%), scale run (30%), chromatic (30%) ----
        pattern_choice = random.random()
        if pattern_choice < 0.40:
            # Chord tone: 3rd or 5th
            ct_candidates = [n for n in chord_tone_list
                             if n != beat1_midi and abs(n - beat1_midi) <= 7]
            if ct_candidates:
                beat2_midi = random.choice(ct_candidates)
            else:
                beat2_midi = random.choice(chord_tone_list)
        elif pattern_choice < 0.70:
            # Scale run: step from beat 1 toward a target (beat 3 area)
            # Aim for a chord tone on beat 3
            beat3_target = _nearest_bass_note(
                chord_beat3.root_pc, beat1_midi
            )
            beat2_midi = _scale_step_toward(beat1_midi, beat3_target,
                                            root_pc, quality)
        else:
            # Chromatic step from beat 1
            direction = random.choice([-1, 1])
            beat2_midi = _clamp(beat1_midi + direction)

        beat2_midi = _clamp(beat2_midi)
        tick2 = _humanize_tick(bar_start_tick + TICKS_PER_QUARTER, amount=8 if swing else 3)
        vel2 = _humanize_velocity(random.randint(70, 85))
        notes.append(NoteEvent(
            pitch=beat2_midi,
            start_tick=tick2,
            duration_ticks=TICKS_PER_QUARTER - 20,
            velocity=vel2,
            channel=0,
        ))

        # ---- BEAT 3: Another chord tone or scale passing tone ----
        beat3_root_pc = chord_beat3.root_pc
        beat3_quality = chord_beat3.quality
        beat3_ct = _chord_tones_in_range(beat3_root_pc, beat3_quality)

        r3 = random.random()
        if r3 < 0.55:
            # Chord tone
            ct3_candidates = [n for n in beat3_ct
                              if n != beat2_midi and abs(n - beat2_midi) <= 7]
            if ct3_candidates:
                beat3_midi = random.choice(ct3_candidates)
            elif beat3_ct:
                beat3_midi = min(beat3_ct, key=lambda n: abs(n - beat2_midi))
            else:
                beat3_midi = _nearest_bass_note(beat3_root_pc, beat2_midi)
        else:
            # Scale passing tone
            next_root_midi = _nearest_bass_note(
                next_chord.root_pc if next_chord else root_pc, beat2_midi
            )
            beat3_midi = _scale_step_toward(beat2_midi, next_root_midi,
                                            beat3_root_pc, beat3_quality)

        beat3_midi = _clamp(beat3_midi)
        tick3 = _humanize_tick(bar_start_tick + 2 * TICKS_PER_QUARTER, amount=8 if swing else 3)
        vel3 = _humanize_velocity(random.randint(70, 85))
        notes.append(NoteEvent(
            pitch=beat3_midi,
            start_tick=tick3,
            duration_ticks=TICKS_PER_QUARTER - 20,
            velocity=vel3,
            channel=0,
        ))

        # ---- BEAT 4: Chromatic approach to next chord's root ----
        if next_chord is not None:
            approach_target_pc = next_chord.root_pc
        else:
            approach_target_pc = root_pc

        approach_target_midi = _nearest_bass_note(approach_target_pc, beat3_midi)
        beat4_midi = _chromatic_approach(approach_target_midi)

        # Make sure approach is actually a half step away — if the target is
        # the same as beat3, pick a chromatic neighbor of the target instead
        if beat4_midi == beat3_midi:
            beat4_midi = _clamp(beat4_midi + random.choice([-1, 1]))

        beat4_midi = _clamp(beat4_midi)
        tick4 = _humanize_tick(bar_start_tick + 3 * TICKS_PER_QUARTER, amount=8 if swing else 3)
        vel4 = _humanize_velocity(random.randint(70, 85))
        notes.append(NoteEvent(
            pitch=beat4_midi,
            start_tick=tick4,
            duration_ticks=TICKS_PER_QUARTER - 20,
            velocity=vel4,
            channel=0,
        ))

        # Update current position for voice-leading continuity
        current_midi = beat4_midi

    return notes


def generate_two_feel_bass(chords, total_beats: int, swing: bool = True) -> List[NoteEvent]:
    """Generate a two-feel bass line (half notes on beats 1 and 3).

    Used for intros, endings, ballads, or lower-intensity sections.

    Args:
        chords: List of ChordEvent objects.
        total_beats: Total number of beats to generate.
        swing: Whether to apply swing feel.

    Returns:
        List of NoteEvent objects.
    """
    if not chords:
        return []

    notes: List[NoteEvent] = []
    total_bars = total_beats // 4
    current_midi = _nearest_bass_note(chords[0].root_pc, 40)

    for bar_idx in range(total_bars):
        bar_start_beat = bar_idx * 4.0
        bar_start_tick = bar_idx * TICKS_PER_BAR

        chord = _chord_at_beat(chords, bar_start_beat)
        if chord is None:
            continue

        chord_beat3 = _chord_at_beat(chords, bar_start_beat + 2.0)
        if chord_beat3 is None:
            chord_beat3 = chord

        # ---- Beat 1: Root ----
        beat1_midi = _nearest_bass_note(chord.root_pc, current_midi)
        # Voice leading: keep stepwise
        if abs(beat1_midi - current_midi) > 7:
            if beat1_midi > current_midi:
                alt = beat1_midi - 12
                if alt >= BASS_LOW:
                    beat1_midi = alt
            else:
                alt = beat1_midi + 12
                if alt <= BASS_HIGH:
                    beat1_midi = alt

        tick1 = _humanize_tick(bar_start_tick, amount=5)
        vel1 = _humanize_velocity(random.randint(80, 95))
        notes.append(NoteEvent(
            pitch=beat1_midi,
            start_tick=tick1,
            duration_ticks=2 * TICKS_PER_QUARTER - 40,  # Half note, slight gap
            velocity=vel1,
            channel=0,
        ))

        # ---- Beat 3: 5th or root of chord at beat 3 ----
        intervals = CHORD_TONES.get(chord_beat3.quality, (0, 4, 7))
        r = random.random()
        if r < 0.6:
            # 5th
            fifth_interval = 7
            for iv in intervals:
                if 6 <= iv <= 8:
                    fifth_interval = iv
                    break
            beat3_pc = (chord_beat3.root_pc + fifth_interval) % 12
        else:
            # Root
            beat3_pc = chord_beat3.root_pc

        beat3_midi = _nearest_bass_note(beat3_pc, beat1_midi)
        beat3_midi = _clamp(beat3_midi)

        tick3 = _humanize_tick(bar_start_tick + 2 * TICKS_PER_QUARTER, amount=5)
        vel3 = _humanize_velocity(random.randint(75, 90))
        notes.append(NoteEvent(
            pitch=beat3_midi,
            start_tick=tick3,
            duration_ticks=2 * TICKS_PER_QUARTER - 40,
            velocity=vel3,
            channel=0,
        ))

        current_midi = beat3_midi

    return notes
