"""Walking bass line generator for Coltrain.

Generates idiomatic jazz walking bass lines using chord-tone targeting,
chromatic approach notes, and voice-leading principles.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import NoteEvent, CCEvent, PitchBendEvent, BarFeel, TICKS_PER_QUARTER, TICKS_PER_BAR, TICKS_PER_8TH, ticks_per_bar

# Module-level rhythmic feel state (set per bar in generator loop)
_current_feel: Optional[BarFeel] = None

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


def _chromatic_enclosure(target_midi: int) -> tuple:
    """Return a 2-note chromatic enclosure targeting *target_midi*.

    Returns (above, below) or (below, above) — two 8th notes that
    surround the resolution note chromatically.
    """
    above = _clamp(target_midi + 1)
    below = _clamp(target_midi - 1)
    if random.random() < 0.5:
        return (above, below)
    return (below, above)


def _is_dominant(quality: str) -> bool:
    """True if the quality string represents a dominant 7th chord."""
    return quality in ("dom7", "7", "aug7", "sus4")


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
    applying swing feel to beats 2 and 4.

    For walking bass, quarter-note beats get a subtle swing displacement
    on beats 2 and 4 to match the drummer's triplet feel.
    """
    tick = bar_start_tick + beat_in_bar * TICKS_PER_QUARTER
    if swing and beat_in_bar in (1, 3):
        depth = _current_feel.swing_depth if _current_feel else 1.0
        tick += int(random.randint(8, 22) * depth)
    # Apply push/pull offset
    if _current_feel and _current_feel.offset_bias != 0.0:
        tick += int(_current_feel.offset_bias * 15)
    return tick


def _humanize_tick(tick: int, amount: int = 10) -> int:
    """Add slight random timing variation, scaled by current feel."""
    spread = _current_feel.timing_spread if _current_feel else 1.0
    scaled = max(1, int(amount * spread))
    return max(0, tick + random.randint(-scaled, scaled))


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


def _next_chord_at_bar(chords, bar_start_beat: float, beats_per_bar: int = 4):
    """Return the chord active at the start of the NEXT bar.
    If none, return the current chord.
    """
    next_bar_beat = bar_start_beat + float(beats_per_bar)
    return _chord_at_beat(chords, next_bar_beat)


# ---------------------------------------------------------------------------
# Accent style helpers
# ---------------------------------------------------------------------------


def _bass_accent_style(intensity: float, bar_in_phrase: int) -> str:
    """Choose a per-bar accent style for walking bass velocity shaping.

    Styles:
      standard  - beat 1 loud, 2&3 soft, 4 medium (traditional)
      even      - all beats roughly equal
      accent_3  - beat 3 accented (pulling toward next chord)
      accent_4  - beat 4 accented (approach note emphasis)
      soft      - everything quieter (breathing room)
    """
    if intensity < 0.3:
        weights = [3, 2, 1, 0, 4]  # standard, even, accent_3, accent_4, soft
    elif intensity < 0.6:
        weights = [3, 3, 2, 1, 1]
    else:
        weights = [2, 3, 3, 2, 0]

    # Bias by position in 4-bar phrase
    if bar_in_phrase == 3:
        weights[3] += 3   # accent_4 on turnaround bars
    elif bar_in_phrase in (1, 2):
        weights[2] += 2   # accent_3 in middle of phrase

    styles = ["standard", "even", "accent_3", "accent_4", "soft"]
    return random.choices(styles, weights=weights, k=1)[0]


def _bass_velocity_for_beat(beat_idx: int, style: str, intensity: float) -> int:
    """Return a velocity for a bass note based on accent style and intensity.

    Uses intensity-scaled floor and ceiling with per-style accent fractions.
    """
    vel_floor = int(55 + intensity * 20)   # 55 at low, 75 at high
    vel_ceil = int(80 + intensity * 30)    # 80 at low, 110 at high

    _PATTERNS = {
        "standard":  {0: (0.9, 1.0), 1: (0.65, 0.80), 2: (0.65, 0.80), 3: (0.70, 0.85)},
        "even":      {0: (0.75, 0.90), 1: (0.75, 0.90), 2: (0.75, 0.90), 3: (0.75, 0.90)},
        "accent_3":  {0: (0.70, 0.85), 1: (0.60, 0.75), 2: (0.85, 1.0), 3: (0.70, 0.85)},
        "accent_4":  {0: (0.70, 0.85), 1: (0.60, 0.75), 2: (0.65, 0.80), 3: (0.85, 1.0)},
        "soft":      {0: (0.60, 0.75), 1: (0.55, 0.70), 2: (0.55, 0.70), 3: (0.60, 0.75)},
    }

    lo_frac, hi_frac = _PATTERNS[style][beat_idx]
    vel_lo = int(vel_floor + (vel_ceil - vel_floor) * lo_frac)
    vel_hi = int(vel_floor + (vel_ceil - vel_floor) * hi_frac)
    return random.randint(vel_lo, max(vel_lo, vel_hi))


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------

TICKS_PER_16TH = TICKS_PER_QUARTER // 4


def _bass_duration_beat1() -> int:
    """Beat 1: mostly legato, occasionally staccato."""
    if random.random() < 0.70:
        return TICKS_PER_QUARTER - random.randint(5, 15)
    return TICKS_PER_QUARTER - random.randint(40, 80)


def _bass_duration_mid() -> int:
    """Beats 2/3: variety from legato to detached."""
    r = random.random()
    if r < 0.50:
        dur = TICKS_PER_QUARTER - random.randint(10, 30)
    elif r < 0.85:
        dur = TICKS_PER_QUARTER - random.randint(30, 60)
    else:
        dur = TICKS_PER_QUARTER - random.randint(80, 140)
    return max(TICKS_PER_16TH, dur)


def _bass_duration_beat4() -> int:
    """Beat 4: mostly short approach, occasionally legato."""
    if random.random() < 0.80:
        return TICKS_PER_QUARTER - random.randint(20, 50)
    return TICKS_PER_QUARTER - random.randint(5, 20)


# ---------------------------------------------------------------------------
# Texture selection — conversational bass (LaFaro / Evans trio style)
# ---------------------------------------------------------------------------

MELODIC_HIGH = 62  # D4 — LaFaro thumb-position territory


@dataclass
class _TextureState:
    """Tracks multi-bar texture coherence to prevent jittery switching."""
    current_texture: str = "walk"
    bars_in_texture: int = 0
    min_bars: int = 1
    recent_textures: List[str] = field(default_factory=list)

    def commit(self, texture: str) -> None:
        """Commit to a new texture, resetting the hold counter."""
        if texture != self.current_texture:
            self.recent_textures.append(self.current_texture)
            if len(self.recent_textures) > 4:
                self.recent_textures.pop(0)
            self.current_texture = texture
            self.bars_in_texture = 0
            if texture == "two_feel":
                self.min_bars = random.randint(1, 2)
            elif texture == "melodic":
                self.min_bars = random.randint(1, 2)
            else:
                self.min_bars = 1
        else:
            self.bars_in_texture += 1


def _choose_bass_texture(state: _TextureState, intensity: float,
                         ctx, bar_idx: int, total_bars: int,
                         chord, next_chord) -> str:
    """Select a bass texture for the current bar.

    Returns one of: 'walk', 'two_feel', 'melodic', 'held', 'double_time'.
    """
    # If current texture hasn't reached its minimum hold, keep it
    if state.bars_in_texture < state.min_bars:
        return state.current_texture

    # Base weights by intensity tier
    if intensity < 0.3:
        weights = {"walk": 40, "two_feel": 30, "melodic": 10,
                   "held": 18, "double_time": 2}
    elif intensity < 0.6:
        weights = {"walk": 57, "two_feel": 18, "melodic": 12,
                   "held": 8, "double_time": 5}
    else:
        weights = {"walk": 55, "two_feel": 5, "melodic": 10,
                   "held": 3, "double_time": 27}

    # Context modifiers
    if ctx is not None:
        if ctx.density > 0.7:
            weights["two_feel"] += 15
            weights["held"] += 10
            weights["walk"] -= 15
        if ctx.has_silence:
            weights["melodic"] += 20
            weights["double_time"] += 5
        section = getattr(ctx, "form_section", None)
        if section == "B":
            weights["melodic"] += 8
        chord_count = getattr(ctx, "chord_count", 1)
        if chord_count >= 2:
            weights["walk"] += 10
            weights["double_time"] += 5

    # Form boundary anchoring: first/last 2 bars of section get walk bias
    if bar_idx < 2 or bar_idx >= total_bars - 2:
        weights["walk"] += 20

    # Anti-repetition: halve weight if texture appeared in last 2 selections
    last_two = state.recent_textures[-2:] if len(state.recent_textures) >= 2 else state.recent_textures
    for tex in last_two:
        if tex in weights:
            weights[tex] = max(1, weights[tex] // 2)

    # Walk floor: never below 30
    weights["walk"] = max(30, weights["walk"])

    # Clamp all to >= 0
    for k in weights:
        weights[k] = max(0, weights[k])

    textures = list(weights.keys())
    w = [weights[t] for t in textures]
    chosen = random.choices(textures, weights=w, k=1)[0]
    state.commit(chosen)
    return chosen


# ---------------------------------------------------------------------------
# Texture generators — melodic fragment, held note, double-time burst
# ---------------------------------------------------------------------------


def _generate_melodic_fragment(
    bar_start_tick: int, chord, next_chord, current_midi: int,
    intensity: float, swing: bool, accent_style: str,
) -> Tuple[List[NoteEvent], int]:
    """Generate a 2-4 note melodic response in the bar.

    Returns (notes, final_pitch) for voice-leading continuity.
    """
    root_pc = chord.root_pc
    quality = chord.quality
    # Extended range up to thumb position
    ct = _chord_tones_in_range(root_pc, quality, BASS_LOW, MELODIC_HIGH)
    if not ct:
        ct = [_nearest_bass_note(root_pc, current_midi)]

    notes: List[NoteEvent] = []
    num_notes = random.randint(2, 4)
    vel_base = int(65 + intensity * 25)

    # Guide tone bias: pick targets from 3rd/7th (40%) or root/5th (60%)
    intervals = CHORD_TONES.get(quality, (0, 4, 7))
    guide_intervals = [iv for iv in intervals if iv in (3, 4, 10, 11)]  # 3rds and 7ths
    anchor_intervals = [iv for iv in intervals if iv in (0, 7)]  # root and 5th

    # Start on beat 1, 1-and, or 2
    start_beats = [0.0, 0.5, 1.0]
    beat_pos = random.choice(start_beats)
    pitch = current_midi

    for j in range(num_notes):
        if beat_pos >= 4.0:
            break

        # Target selection: guide tone or anchor
        if random.random() < 0.40 and guide_intervals:
            target_iv = random.choice(guide_intervals)
        elif anchor_intervals:
            target_iv = random.choice(anchor_intervals)
        else:
            target_iv = 0
        target_pc = (root_pc + target_iv) % 12
        target_midi = _nearest_bass_note(target_pc, pitch, BASS_LOW, MELODIC_HIGH)

        # Move stepwise toward target if far
        if abs(target_midi - pitch) > 4:
            pitch = _scale_step_toward(pitch, target_midi, root_pc, quality)
        else:
            pitch = target_midi
        pitch = _clamp(pitch, BASS_LOW, MELODIC_HIGH)

        # Pull back on last note to stay in comfortable range
        if j == num_notes - 1 and pitch > BASS_HIGH:
            pitch = _clamp(pitch, BASS_LOW, BASS_HIGH)

        # Timing: mix of quarter and 8th notes
        is_eighth = random.random() < 0.4
        dur_beats = 0.5 if is_eighth else 1.0

        tick = _swing_tick_bass(bar_start_tick, beat_pos, swing)
        tick = _humanize_tick(tick, 12)
        dur_ticks = int(dur_beats * TICKS_PER_QUARTER) - random.randint(15, 35)
        dur_ticks = max(TICKS_PER_16TH, dur_ticks)

        vel = _humanize_velocity(vel_base + random.randint(-8, 8))
        notes.append(NoteEvent(
            pitch=pitch, start_tick=tick,
            duration_ticks=dur_ticks, velocity=vel, channel=0,
        ))
        beat_pos += dur_beats

    return notes, pitch


def _generate_held_note(
    bar_start_tick: int, chord, current_midi: int, intensity: float,
    swing: bool, beats_per_bar: int = 4,
) -> Tuple[List[NoteEvent], int]:
    """Generate a single held chord tone for half or whole bar.

    Creates exposed piano space. Returns (notes, pitch).
    """
    root_pc = chord.root_pc
    quality = chord.quality
    intervals = CHORD_TONES.get(quality, (0, 4, 7))

    # Root preferred (60%), 5th (25%), 3rd (15%)
    r = random.random()
    if r < 0.60:
        target_iv = 0
    elif r < 0.85:
        fifth = 7
        for iv in intervals:
            if 6 <= iv <= 8:
                fifth = iv
                break
        target_iv = fifth
    else:
        third = 4
        for iv in intervals:
            if 3 <= iv <= 4:
                third = iv
                break
        target_iv = third

    target_pc = (root_pc + target_iv) % 12
    pitch = _nearest_bass_note(target_pc, current_midi)
    # Voice-leading: keep close
    if abs(pitch - current_midi) > 7:
        alt = pitch + (-12 if pitch > current_midi else 12)
        if BASS_LOW <= alt <= BASS_HIGH:
            pitch = alt

    # Half bar or whole bar
    if random.random() < 0.45:
        dur = ticks_per_bar(beats_per_bar) - random.randint(30, 60)
    else:
        dur = 2 * TICKS_PER_QUARTER - random.randint(20, 40)

    tick = _humanize_tick(bar_start_tick, amount=8)
    vel = random.randint(60, 80)
    vel = _humanize_velocity(vel)

    notes = [NoteEvent(
        pitch=pitch, start_tick=tick,
        duration_ticks=dur, velocity=vel, channel=0,
    )]
    return notes, pitch


def _generate_double_time_burst(
    bar_start_tick: int, chord, next_chord, current_midi: int,
    intensity: float, swing: bool, accent_style: str,
) -> Tuple[List[NoteEvent], int]:
    """Generate half-bar walking + half-bar 8th-note scalar burst.

    50/50 whether burst is on beats 1-2 or beats 3-4.
    Returns (notes, final_pitch).
    """
    root_pc = chord.root_pc
    quality = chord.quality
    notes: List[NoteEvent] = []
    pitch = current_midi

    # Determine which half gets the burst
    burst_first_half = random.random() < 0.5
    vel_base = int(_bass_velocity_for_beat(0, accent_style, intensity))
    vel_burst = min(127, vel_base + random.randint(5, 10))

    if burst_first_half:
        # Beats 1-2: 4 eighth-note scalar run
        target = _nearest_bass_note(
            next_chord.root_pc if next_chord else root_pc, pitch)
        for ei in range(4):
            beat_off = ei * 0.5
            tick = _swing_tick_bass(bar_start_tick, beat_off, swing)
            tick = _humanize_tick(tick, 8)
            pitch = _scale_step_toward(pitch, target, root_pc, quality)
            pitch = _clamp(pitch)
            vel = _humanize_velocity(vel_burst + random.randint(-5, 5))
            notes.append(NoteEvent(
                pitch=pitch, start_tick=tick,
                duration_ticks=TICKS_PER_8TH - random.randint(10, 25),
                velocity=vel, channel=0,
            ))
        # Beats 3-4: normal walking quarter notes
        for beat_idx in (2, 3):
            tick = _apply_swing_tick(beat_idx, bar_start_tick, swing)
            tick = _humanize_tick(tick, 15)
            ct = _chord_tones_in_range(root_pc, quality)
            nearby = [n for n in ct if abs(n - pitch) <= 5]
            if nearby:
                pitch = random.choice(nearby)
            elif ct:
                pitch = min(ct, key=lambda n: abs(n - pitch))
            vel = _humanize_velocity(
                _bass_velocity_for_beat(beat_idx, accent_style, intensity))
            dur = _bass_duration_mid() if beat_idx == 2 else _bass_duration_beat4()
            notes.append(NoteEvent(
                pitch=pitch, start_tick=tick,
                duration_ticks=dur, velocity=vel, channel=0,
            ))
    else:
        # Beats 1-2: normal walking quarter notes
        for beat_idx in (0, 1):
            tick = _apply_swing_tick(beat_idx, bar_start_tick, swing)
            tick = _humanize_tick(tick, 15)
            if beat_idx == 0:
                pitch = _nearest_bass_note(root_pc, pitch)
                if abs(pitch - current_midi) > 7:
                    alt = pitch + (-12 if pitch > current_midi else 12)
                    if BASS_LOW <= alt <= BASS_HIGH:
                        pitch = alt
            else:
                ct = _chord_tones_in_range(root_pc, quality)
                nearby = [n for n in ct if n != pitch and abs(n - pitch) <= 5]
                if nearby:
                    pitch = random.choice(nearby)
            vel = _humanize_velocity(
                _bass_velocity_for_beat(beat_idx, accent_style, intensity))
            dur = _bass_duration_beat1() if beat_idx == 0 else _bass_duration_mid()
            notes.append(NoteEvent(
                pitch=pitch, start_tick=tick,
                duration_ticks=dur, velocity=vel, channel=0,
            ))
        # Beats 3-4: 4 eighth-note scalar run toward next bar root
        target = _nearest_bass_note(
            next_chord.root_pc if next_chord else root_pc, pitch)
        for ei in range(4):
            beat_off = 2.0 + ei * 0.5
            tick = _swing_tick_bass(bar_start_tick, beat_off, swing)
            tick = _humanize_tick(tick, 8)
            pitch = _scale_step_toward(pitch, target, root_pc, quality)
            pitch = _clamp(pitch)
            vel = _humanize_velocity(vel_burst + random.randint(-5, 5))
            notes.append(NoteEvent(
                pitch=pitch, start_tick=tick,
                duration_ticks=TICKS_PER_8TH - random.randint(10, 25),
                velocity=vel, channel=0,
            ))

    return notes, pitch


# ---------------------------------------------------------------------------
# Main generators
# ---------------------------------------------------------------------------


def generate_walking_bass(chords, total_beats: int, swing: bool = True,
                          intensity: float = 0.5,
                          bar_intensities: Optional[List[float]] = None,
                          bar_context: Optional[list] = None,
                          bar_feel: Optional[list] = None,
                          beats_per_bar: int = 4) -> List[NoteEvent]:
    """Generate a walking bass line over the given chord progression.

    Args:
        chords: List of ChordEvent objects.
        total_beats: Total number of beats to generate.
        swing: Whether to apply swing feel (affects humanization).
        intensity: 0.0-1.0. Low (<0.3) mixes in two-feel bars; high (>0.7)
                   adds 8th-note pickups.

    Returns:
        List of NoteEvent objects forming the bass line.
    """
    if not chords:
        return []

    notes: List[NoteEvent] = []
    tpb = ticks_per_bar(beats_per_bar)
    total_bars = total_beats // beats_per_bar

    # Start on the root of the first chord, in a comfortable range
    first_chord = chords[0]
    current_midi = _nearest_bass_note(first_chord.root_pc, 40)  # Start around E2

    texture_state = _TextureState()

    for bar_idx in range(total_bars):
        global _current_feel
        _current_feel = bar_feel[bar_idx] if bar_feel and bar_idx < len(bar_feel) else None

        bar_start_beat = bar_idx * float(beats_per_bar)
        bar_start_tick = bar_idx * tpb

        chord = _chord_at_beat(chords, bar_start_beat)
        if chord is None:
            continue

        # Get the chord active at mid-bar (might be different if chord changes mid-bar)
        mid_beat = beats_per_bar // 2
        chord_beat3 = _chord_at_beat(chords, bar_start_beat + mid_beat)
        if chord_beat3 is None:
            chord_beat3 = chord

        # Look ahead to the next bar's chord for approach note on last beat
        next_chord = _next_chord_at_bar(chords, bar_start_beat, beats_per_bar)

        root_pc = chord.root_pc
        quality = chord.quality

        # Per-bar reactive intensity (falls back to section-level value)
        local_intensity = (bar_intensities[bar_idx]
                           if bar_intensities and bar_idx < len(bar_intensities)
                           else intensity)

        # Inter-instrument reactivity: adjust based on what lead plays
        ctx = bar_context[bar_idx] if bar_context and bar_idx < len(bar_context) else None
        react_drop_boost = 0.0
        if ctx is not None:
            if ctx.density > 0.7:
                # Lead is busy — simplify bass
                react_drop_boost = 0.15
            elif ctx.has_silence:
                # Lead is laying out — bass can be more active
                react_drop_boost = -0.08

        # Per-bar accent style and rest decisions
        accent_style = _bass_accent_style(local_intensity, bar_idx % 4)
        rest_prob = max(0.02, 0.18 - local_intensity * 0.15) + react_drop_boost
        rest_prob = max(0.0, min(0.5, rest_prob))
        drop_beat_2 = random.random() < rest_prob
        drop_beat_3 = random.random() < rest_prob * 0.8

        # --- Texture selection: conversational bass ---
        texture = _choose_bass_texture(texture_state, local_intensity, ctx,
                                       bar_idx, total_bars, chord, next_chord)

        if texture == "two_feel":
            tf_mid = beats_per_bar // 2 + (1 if beats_per_bar % 2 else 0)
            beat1_midi = _nearest_bass_note(root_pc, current_midi)
            if abs(beat1_midi - current_midi) > 7:
                alt = beat1_midi + (-12 if beat1_midi > current_midi else 12)
                if BASS_LOW <= alt <= BASS_HIGH:
                    beat1_midi = alt
            tick1 = _humanize_tick(bar_start_tick, amount=22 if swing else 8)
            vel1 = _humanize_velocity(_bass_velocity_for_beat(0, accent_style, local_intensity))
            notes.append(NoteEvent(
                pitch=beat1_midi,
                start_tick=tick1,
                duration_ticks=tf_mid * TICKS_PER_QUARTER - random.randint(20, 40),
                velocity=vel1,
                channel=0,
            ))
            # Mid-bar: 5th or root
            fifth_int = 7
            for iv in CHORD_TONES.get(chord_beat3.quality, (0, 4, 7)):
                if 6 <= iv <= 8:
                    fifth_int = iv
                    break
            b3_pc = (chord_beat3.root_pc + fifth_int) % 12 if random.random() < 0.5 else chord_beat3.root_pc
            beat3_midi = _nearest_bass_note(b3_pc, beat1_midi)
            tick3 = _humanize_tick(bar_start_tick + tf_mid * TICKS_PER_QUARTER, amount=22 if swing else 8)
            vel3 = _humanize_velocity(_bass_velocity_for_beat(2, accent_style, local_intensity))
            notes.append(NoteEvent(
                pitch=beat3_midi,
                start_tick=tick3,
                duration_ticks=(beats_per_bar - tf_mid) * TICKS_PER_QUARTER - random.randint(20, 40),
                velocity=vel3,
                channel=0,
            ))
            current_midi = beat3_midi
            continue

        if texture == "melodic":
            frag_notes, frag_pitch = _generate_melodic_fragment(
                bar_start_tick, chord, next_chord, current_midi,
                local_intensity, swing, accent_style)
            notes.extend(frag_notes)
            current_midi = frag_pitch
            continue

        if texture == "held":
            held_notes, held_pitch = _generate_held_note(
                bar_start_tick, chord, current_midi, local_intensity, swing,
                beats_per_bar=beats_per_bar)
            notes.extend(held_notes)
            current_midi = held_pitch
            continue

        if texture == "double_time":
            dt_notes, dt_pitch = _generate_double_time_burst(
                bar_start_tick, chord, next_chord, current_midi,
                local_intensity, swing, accent_style)
            notes.extend(dt_notes)
            current_midi = dt_pitch
            continue

        # else: texture == "walk" — fall through to existing beat 1-4 code

        chord_tone_list = _chord_tones_in_range(root_pc, quality)
        if not chord_tone_list:
            chord_tone_list = [_nearest_bass_note(root_pc, current_midi)]

        # ---- Detect ii-V resolution: was previous beat's chord dominant? ----
        prev_chord = _chord_at_beat(chords, bar_start_beat - 1.0) if bar_idx > 0 else None
        is_resolution = (prev_chord is not None
                         and _is_dominant(prev_chord.quality)
                         and prev_chord.root_pc != root_pc)

        # ---- Form section awareness for beat 1 weights ----
        section = chord.form_section

        # ---- BEAT 1: pitch class selection ----
        # On ii-V resolution: target guide tones (40% root, 30% 3rd, 30% 7th)
        # On bridge (B section): anchor with root (80% root, 10% 5th, 10% 3rd)
        # Normal: 65% root, 20% 5th, 15% 3rd
        intervals = CHORD_TONES.get(quality, (0, 4, 7))
        third_interval = 4
        for iv in intervals:
            if 3 <= iv <= 4:
                third_interval = iv
                break
        fifth_interval = 7
        for iv in intervals:
            if 6 <= iv <= 8:
                fifth_interval = iv
                break
        seventh_interval = 10
        for iv in intervals:
            if 9 <= iv <= 11:
                seventh_interval = iv
                break

        r = random.random()
        if is_resolution:
            # ii-V resolution: land on guide tones for harmonic clarity
            if r < 0.40:
                beat1_pc = root_pc
            elif r < 0.70:
                beat1_pc = (root_pc + third_interval) % 12
            else:
                beat1_pc = (root_pc + seventh_interval) % 12
        elif section == "B":
            # Bridge: anchor distant key centers with root
            if r < 0.80:
                beat1_pc = root_pc
            elif r < 0.90:
                beat1_pc = (root_pc + fifth_interval) % 12
            else:
                beat1_pc = (root_pc + third_interval) % 12
        else:
            # Normal A/C sections
            if r < 0.65:
                beat1_pc = root_pc
            elif r < 0.85:
                beat1_pc = (root_pc + fifth_interval) % 12
            else:
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

        tick1 = _humanize_tick(bar_start_tick, amount=22 if swing else 8)
        vel1 = _humanize_velocity(_bass_velocity_for_beat(0, accent_style, local_intensity))
        notes.append(NoteEvent(
            pitch=beat1_midi,
            start_tick=tick1,
            duration_ticks=_bass_duration_beat1(),
            velocity=vel1,
            channel=0,
        ))

        # ---- BEAT 2: context-sensitive passing tone ----
        # Bridge or fast changes: more chromatic (45% chromatic vs 30% normal)
        # Spacious chords (duration >= 4): favor diatonic scale runs
        fast_harmony = chord.duration_beats <= 2
        chromatic_bias = (section == "B" or fast_harmony)

        if drop_beat_2:
            beat2_midi = beat1_midi  # Fallback for voice-leading continuity
        else:
            pattern_choice = random.random()
            if chromatic_bias:
                # More chromatic passing during bridge / fast changes
                ct_thresh, scale_thresh = 0.30, 0.55
            else:
                ct_thresh, scale_thresh = 0.40, 0.70
            if pattern_choice < ct_thresh:
                # Chord tone: 3rd or 5th
                ct_candidates = [n for n in chord_tone_list
                                 if n != beat1_midi and abs(n - beat1_midi) <= 7]
                if ct_candidates:
                    beat2_midi = random.choice(ct_candidates)
                else:
                    beat2_midi = random.choice(chord_tone_list)
            elif pattern_choice < scale_thresh:
                # Scale run: step from beat 1 toward a target (beat 3 area)
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
            tick2 = _humanize_tick(bar_start_tick + TICKS_PER_QUARTER, amount=22 if swing else 8)
            vel2 = _humanize_velocity(_bass_velocity_for_beat(1, accent_style, local_intensity))
            notes.append(NoteEvent(
                pitch=beat2_midi,
                start_tick=tick2,
                duration_ticks=_bass_duration_mid(),
                velocity=vel2,
                channel=0,
            ))

        # ---- BEAT 3: Another chord tone or scale passing tone ----
        if drop_beat_3:
            beat3_midi = beat2_midi  # Fallback for voice-leading continuity
        else:
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
                    beat3_midi = _nearest_bass_note(chord_beat3.root_pc, beat2_midi)
            else:
                # Scale passing tone
                next_root_midi = _nearest_bass_note(
                    next_chord.root_pc if next_chord else root_pc, beat2_midi
                )
                beat3_midi = _scale_step_toward(beat2_midi, next_root_midi,
                                                chord_beat3.root_pc, chord_beat3.quality)

            beat3_midi = _clamp(beat3_midi)
            tick3 = _humanize_tick(bar_start_tick + 2 * TICKS_PER_QUARTER, amount=22 if swing else 8)
            vel3 = _humanize_velocity(_bass_velocity_for_beat(2, accent_style, local_intensity))

            # Occasional 8th-note pair on beat 3 (rhythmic variety)
            if local_intensity > 0.30 and random.random() < 0.20:
                # First 8th: beat3_midi
                notes.append(NoteEvent(
                    pitch=beat3_midi, start_tick=tick3,
                    duration_ticks=TICKS_PER_8TH - random.randint(10, 25),
                    velocity=vel3, channel=0,
                ))
                # Second 8th: scale step toward beat 4 target area
                b3_and = _scale_step_toward(
                    beat3_midi, _nearest_bass_note(
                        next_chord.root_pc if next_chord else root_pc, beat3_midi),
                    chord_beat3.root_pc, chord_beat3.quality)
                b3_and = _clamp(b3_and)
                tick3_and = _humanize_tick(
                    bar_start_tick + 2 * TICKS_PER_QUARTER + TICKS_PER_8TH,
                    amount=12)
                notes.append(NoteEvent(
                    pitch=b3_and, start_tick=tick3_and,
                    duration_ticks=TICKS_PER_8TH - random.randint(10, 25),
                    velocity=_humanize_velocity(vel3 - random.randint(3, 8)),
                    channel=0,
                ))
                beat3_midi = b3_and  # Update for voice-leading into beat 4
            else:
                notes.append(NoteEvent(
                    pitch=beat3_midi, start_tick=tick3,
                    duration_ticks=_bass_duration_mid(),
                    velocity=vel3, channel=0,
                ))

        # ---- EXTRA MID-BAR BEATS for odd meters (between beat 3 and approach) ----
        # In 5/4: beat 4 is a passing tone, beat 5 is approach
        # In 7/4: beats 4-6 are passing/chord tones, beat 7 is approach
        last_mid_midi = beat3_midi
        approach_beat_idx = beats_per_bar - 1  # Last beat = approach
        for extra_beat in range(3, approach_beat_idx):
            # Alternate between chord tones and scale steps toward approach target
            if next_chord is not None:
                target_pc = next_chord.root_pc
            else:
                target_pc = root_pc
            target_midi = _nearest_bass_note(target_pc, last_mid_midi)

            if extra_beat % 2 == 1:
                # Chord tone of current harmony
                ct_cands = [n for n in chord_tone_list
                            if n != last_mid_midi and abs(n - last_mid_midi) <= 7]
                if ct_cands:
                    extra_midi = random.choice(ct_cands)
                else:
                    extra_midi = _scale_step_toward(last_mid_midi, target_midi,
                                                    root_pc, quality)
            else:
                # Scale step toward approach target
                extra_midi = _scale_step_toward(last_mid_midi, target_midi,
                                                root_pc, quality)
            extra_midi = _clamp(extra_midi)
            tick_extra = _humanize_tick(
                bar_start_tick + extra_beat * TICKS_PER_QUARTER,
                amount=22 if swing else 8)
            vel_extra = _humanize_velocity(
                _bass_velocity_for_beat(1, accent_style, local_intensity))
            notes.append(NoteEvent(
                pitch=extra_midi, start_tick=tick_extra,
                duration_ticks=_bass_duration_mid(),
                velocity=vel_extra, channel=0,
            ))
            last_mid_midi = extra_midi

        # ---- LAST BEAT: Varied approach to next chord's root ----
        if next_chord is not None:
            approach_target_pc = next_chord.root_pc
        else:
            approach_target_pc = root_pc

        approach_target_midi = _nearest_bass_note(approach_target_pc, last_mid_midi)

        b4_roll = random.random()
        if b4_roll < 0.20:
            # Chromatic enclosure: two 8th notes surrounding the target
            enc_a, enc_b = _chromatic_enclosure(approach_target_midi)
            tick4a = _humanize_tick(bar_start_tick + approach_beat_idx * TICKS_PER_QUARTER, amount=12)
            tick4b = _humanize_tick(
                bar_start_tick + approach_beat_idx * TICKS_PER_QUARTER + TICKS_PER_8TH, amount=12)
            vel4 = _humanize_velocity(_bass_velocity_for_beat(3, accent_style, local_intensity))
            notes.append(NoteEvent(
                pitch=enc_a, start_tick=tick4a,
                duration_ticks=TICKS_PER_8TH - random.randint(10, 25),
                velocity=vel4, channel=0,
            ))
            notes.append(NoteEvent(
                pitch=enc_b, start_tick=tick4b,
                duration_ticks=TICKS_PER_8TH - random.randint(10, 25),
                velocity=_humanize_velocity(vel4 - random.randint(2, 6)),
                channel=0,
            ))
            beat4_midi = enc_b
        else:
            if b4_roll < 0.50:
                # Chromatic approach (half step above or below target)
                beat4_midi = _chromatic_approach(approach_target_midi)
            elif b4_roll < 0.75:
                # Diatonic scale step toward target
                beat4_midi = _scale_step_toward(
                    last_mid_midi, approach_target_midi,
                    chord_beat3.root_pc, chord_beat3.quality)
            elif b4_roll < 0.90:
                # Double chromatic — two semitones away for a longer run-in
                direction = 1 if approach_target_midi > last_mid_midi else -1
                beat4_midi = _clamp(approach_target_midi + direction * 2)
            else:
                # Anticipation — play the target note itself early
                beat4_midi = approach_target_midi

            # Avoid repeating previous pitch
            if beat4_midi == last_mid_midi:
                beat4_midi = _clamp(beat4_midi + random.choice([-1, 1]))

            beat4_midi = _clamp(beat4_midi)
            tick4 = _humanize_tick(bar_start_tick + approach_beat_idx * TICKS_PER_QUARTER,
                                  amount=22 if swing else 8)
            vel4 = _humanize_velocity(_bass_velocity_for_beat(3, accent_style, local_intensity))
            notes.append(NoteEvent(
                pitch=beat4_midi,
                start_tick=tick4,
                duration_ticks=_bass_duration_beat4(),
                velocity=vel4,
                channel=0,
            ))

        # High intensity: occasional 8th-note pickup into next bar's beat 1
        if local_intensity > 0.7 and random.random() < 0.20 and bar_idx < total_bars - 1:
            next_bar_chord = _chord_at_beat(chords, (bar_idx + 1) * float(beats_per_bar))
            if next_bar_chord is not None:
                pickup_target = _nearest_bass_note(next_bar_chord.root_pc, beat4_midi)
                pickup_midi = _chromatic_approach(pickup_target)
                pickup_tick = bar_start_tick + approach_beat_idx * TICKS_PER_QUARTER + TICKS_PER_8TH
                notes.append(NoteEvent(
                    pitch=_clamp(pickup_midi),
                    start_tick=_humanize_tick(pickup_tick, amount=12),
                    duration_ticks=TICKS_PER_8TH - 20,
                    velocity=_humanize_velocity(random.randint(60, 75)),
                    channel=0,
                ))

        # Update current position for voice-leading continuity
        current_midi = beat4_midi

    return notes


def generate_two_feel_bass(chords, total_beats: int, swing: bool = True,
                           beats_per_bar: int = 4) -> List[NoteEvent]:
    """Generate a two-feel bass line (half notes on beats 1 and mid-bar).

    Used for intros, endings, ballads, or lower-intensity sections.
    For 5/4: beat 1 (dotted half) + beat 4 (half). For 7/4: beat 1 + beat 4.
    """
    if not chords:
        return []

    notes: List[NoteEvent] = []
    tpb = ticks_per_bar(beats_per_bar)
    total_bars = total_beats // beats_per_bar
    current_midi = _nearest_bass_note(chords[0].root_pc, 40)

    # Two-feel split point: where the second note starts
    mid_beat = beats_per_bar // 2 + (1 if beats_per_bar % 2 else 0)  # 2 for 4/4, 3 for 5/4, 4 for 7/4

    for bar_idx in range(total_bars):
        bar_start_beat = bar_idx * float(beats_per_bar)
        bar_start_tick = bar_idx * tpb

        chord = _chord_at_beat(chords, bar_start_beat)
        if chord is None:
            continue

        chord_mid = _chord_at_beat(chords, bar_start_beat + mid_beat)
        if chord_mid is None:
            chord_mid = chord

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
        dur1 = mid_beat * TICKS_PER_QUARTER - 40
        notes.append(NoteEvent(
            pitch=beat1_midi,
            start_tick=tick1,
            duration_ticks=dur1,
            velocity=vel1,
            channel=0,
        ))

        # ---- Mid-bar beat: 5th or root of chord at mid-bar ----
        intervals = CHORD_TONES.get(chord_mid.quality, (0, 4, 7))
        r = random.random()
        if r < 0.6:
            # 5th
            fifth_interval = 7
            for iv in intervals:
                if 6 <= iv <= 8:
                    fifth_interval = iv
                    break
            beat3_pc = (chord_mid.root_pc + fifth_interval) % 12
        else:
            # Root
            beat3_pc = chord_mid.root_pc

        beat3_midi = _nearest_bass_note(beat3_pc, beat1_midi)
        beat3_midi = _clamp(beat3_midi)

        remaining_beats = beats_per_bar - mid_beat
        tick3 = _humanize_tick(bar_start_tick + mid_beat * TICKS_PER_QUARTER, amount=5)
        vel3 = _humanize_velocity(random.randint(75, 90))
        notes.append(NoteEvent(
            pitch=beat3_midi,
            start_tick=tick3,
            duration_ticks=remaining_beats * TICKS_PER_QUARTER - 40,
            velocity=vel3,
            channel=0,
        ))

        current_midi = beat3_midi

    return notes


def generate_modal_bass(chords, total_beats: int, swing: bool = True,
                        beats_per_bar: int = 4) -> List[NoteEvent]:
    """Generate a modal/pedal-point bass line — root and 5th, half/whole notes.

    Used for modal jazz (Impressions, So What, etc.) where the harmony is
    static and the bass provides a drone-like foundation.
    """
    if not chords:
        return []

    notes: List[NoteEvent] = []
    tpb = ticks_per_bar(beats_per_bar)
    total_bars = total_beats // beats_per_bar
    current_midi = _nearest_bass_note(chords[0].root_pc, 40)

    for bar_idx in range(total_bars):
        bar_start_beat = bar_idx * float(beats_per_bar)
        bar_start_tick = bar_idx * tpb

        chord = _chord_at_beat(chords, bar_start_beat)
        if chord is None:
            continue

        root_pc = chord.root_pc
        # Find the 5th interval
        intervals = CHORD_TONES.get(chord.quality, (0, 4, 7))
        fifth_interval = 7
        for iv in intervals:
            if 6 <= iv <= 8:
                fifth_interval = iv
                break
        fifth_pc = (root_pc + fifth_interval) % 12

        root_midi = _nearest_bass_note(root_pc, current_midi)
        fifth_midi = _nearest_bass_note(fifth_pc, root_midi)

        # Pattern varies per bar
        mid_beat = beats_per_bar // 2 + (1 if beats_per_bar % 2 else 0)
        first_half_ticks = mid_beat * TICKS_PER_QUARTER
        second_half_ticks = (beats_per_bar - mid_beat) * TICKS_PER_QUARTER
        pattern_roll = random.random()
        if pattern_roll < 0.5:
            # Whole note on root
            tick = _humanize_tick(bar_start_tick, amount=5)
            vel = _humanize_velocity(random.randint(80, 95))
            notes.append(NoteEvent(
                pitch=root_midi,
                start_tick=tick,
                duration_ticks=tpb - 40,
                velocity=vel,
                channel=0,
            ))
        elif pattern_roll < 0.8:
            # First half root, second half 5th
            tick1 = _humanize_tick(bar_start_tick, amount=5)
            vel1 = _humanize_velocity(random.randint(80, 95))
            notes.append(NoteEvent(
                pitch=root_midi,
                start_tick=tick1,
                duration_ticks=first_half_ticks - 40,
                velocity=vel1,
                channel=0,
            ))
            tick2 = _humanize_tick(bar_start_tick + first_half_ticks, amount=5)
            vel2 = _humanize_velocity(random.randint(75, 90))
            notes.append(NoteEvent(
                pitch=fifth_midi,
                start_tick=tick2,
                duration_ticks=second_half_ticks - 40,
                velocity=vel2,
                channel=0,
            ))
        else:
            # First half 5th, second half root (inverted)
            tick1 = _humanize_tick(bar_start_tick, amount=5)
            vel1 = _humanize_velocity(random.randint(75, 90))
            notes.append(NoteEvent(
                pitch=fifth_midi,
                start_tick=tick1,
                duration_ticks=first_half_ticks - 40,
                velocity=vel1,
                channel=0,
            ))
            tick2 = _humanize_tick(bar_start_tick + first_half_ticks, amount=5)
            vel2 = _humanize_velocity(random.randint(80, 95))
            notes.append(NoteEvent(
                pitch=root_midi,
                start_tick=tick2,
                duration_ticks=second_half_ticks - 40,
                velocity=vel2,
                channel=0,
            ))

        current_midi = root_midi

    return notes


# ---------------------------------------------------------------------------
# Bass solo generator
# ---------------------------------------------------------------------------

# Solo range: upper register of the bass (thumb position)
BASS_SOLO_LOW = 36   # C2
BASS_SOLO_HIGH = 62  # D4


def generate_bass_solo(chords, total_beats: int, swing: bool = True,
                       intensity: float = 0.7,
                       beats_per_bar: int = 4) -> List[NoteEvent]:
    """Generate a melodic jazz bass solo over chord changes.

    Uses the upper register of the bass with a mix of chord tones,
    scalar passages, and chromatic approach notes.  Phrasing alternates
    between 2-4 bar melodic ideas and breathing space.
    """
    if not chords:
        return []

    intensity = max(0.0, min(1.0, intensity))
    tpb = ticks_per_bar(beats_per_bar)
    total_bars = total_beats // beats_per_bar
    notes: List[NoteEvent] = []

    # Start near middle of solo range
    mid = (BASS_SOLO_LOW + BASS_SOLO_HIGH) // 2
    chord0 = _chord_at_beat(chords, 0.0)
    current_midi = _nearest_bass_note(
        chord0.root_pc if chord0 else 0, mid,
        low=BASS_SOLO_LOW, high=BASS_SOLO_HIGH)

    bar_idx = 0
    while bar_idx < total_bars:
        # Dynamic arc
        progress = bar_idx / max(1, total_bars - 1)
        arc = 0.4 + 0.6 * math.sin(progress * math.pi)
        local_int = min(1.0, intensity * arc)

        # Phrase length: 1-4 bars
        phrase_bars = min(random.choice([1, 2, 2, 3, 4]), total_bars - bar_idx)

        # Choose phrase style
        style = random.choices(
            ["chord_tone", "scalar_run", "motif", "held_note"],
            weights=[0.35, 0.25, 0.25, 0.15],
            k=1,
        )[0]

        for pb in range(phrase_bars):
            bi = bar_idx + pb
            bar_start_beat = bi * float(beats_per_bar)
            bar_start_tick = bi * tpb

            chord = _chord_at_beat(chords, bar_start_beat)
            if chord is None:
                continue

            root_pc = chord.root_pc
            quality = chord.quality
            vel_base = int(80 + 25 * local_int)

            if style == "chord_tone":
                # Quarter-note chord tones with passing tones
                ct = _chord_tones_in_range(root_pc, quality,
                                           BASS_SOLO_LOW, BASS_SOLO_HIGH)
                for beat_idx in range(4):
                    if random.random() < 0.15:
                        continue  # Skip some beats for space
                    tick = _apply_swing_tick(beat_idx, bar_start_tick, swing)
                    tick = _humanize_tick(tick, 8)
                    # On beats 1 and 3: chord tone; beats 2 and 4: passing tone
                    if beat_idx % 2 == 0 and ct:
                        target = min(ct, key=lambda n: abs(n - current_midi))
                        current_midi = target
                    else:
                        # Scale step toward next chord tone
                        current_midi = _scale_step_toward(
                            current_midi, current_midi + random.choice([-2, -1, 1, 2]),
                            root_pc, quality)
                    current_midi = _clamp(current_midi, BASS_SOLO_LOW, BASS_SOLO_HIGH)
                    vel = _humanize_velocity(vel_base + random.randint(-10, 5))
                    dur = TICKS_PER_QUARTER - random.randint(20, 60)
                    notes.append(NoteEvent(
                        pitch=current_midi, start_tick=tick,
                        duration_ticks=max(TICKS_PER_8TH, dur),
                        velocity=vel, channel=0,
                    ))

            elif style == "scalar_run":
                # 8th-note scalar run across the bar
                scale = _scale_tones_in_range(root_pc, quality,
                                              BASS_SOLO_LOW, BASS_SOLO_HIGH)
                if not scale:
                    continue
                direction = random.choice([-1, 1])
                for eighth_idx in range(8):
                    if random.random() < 0.10:
                        continue  # Occasional skip
                    beat_off = eighth_idx * 0.5
                    tick = _swing_tick_bass(bar_start_tick, beat_off, swing)
                    tick = _humanize_tick(tick, 6)
                    # Step through scale
                    if direction > 0:
                        above = [n for n in scale if n > current_midi]
                        current_midi = above[0] if above else scale[-1]
                    else:
                        below = [n for n in scale if n < current_midi]
                        current_midi = below[-1] if below else scale[0]
                    # Reverse direction at range limits
                    if current_midi >= BASS_SOLO_HIGH - 2:
                        direction = -1
                    elif current_midi <= BASS_SOLO_LOW + 2:
                        direction = 1
                    vel = _humanize_velocity(vel_base + random.randint(-15, 0))
                    notes.append(NoteEvent(
                        pitch=current_midi, start_tick=tick,
                        duration_ticks=TICKS_PER_8TH - 10,
                        velocity=vel, channel=0,
                    ))

            elif style == "motif":
                # Short rhythmic motif: 2-3 notes then space
                num_notes = random.randint(2, 4)
                beat_pos = random.choice([0.0, 0.5, 1.0])
                ct = _chord_tones_in_range(root_pc, quality,
                                           BASS_SOLO_LOW, BASS_SOLO_HIGH)
                for j in range(num_notes):
                    bp = beat_pos + j * random.choice([0.5, 0.75, 1.0])
                    if bp >= 4.0:
                        break
                    tick = _swing_tick_bass(bar_start_tick, bp, swing)
                    tick = _humanize_tick(tick, 8)
                    if ct:
                        target = min(ct, key=lambda n: abs(n - current_midi))
                        step = random.choice([-2, -1, 1, 2])
                        idx = ct.index(target) if target in ct else 0
                        idx = max(0, min(len(ct) - 1, idx + step))
                        current_midi = ct[idx]
                    vel = _humanize_velocity(vel_base + random.randint(-5, 10))
                    dur = random.choice([TICKS_PER_QUARTER, TICKS_PER_8TH,
                                         TICKS_PER_QUARTER + TICKS_PER_8TH])
                    notes.append(NoteEvent(
                        pitch=current_midi, start_tick=tick,
                        duration_ticks=dur, velocity=vel, channel=0,
                    ))

            else:  # held_note
                # Single held note (half or whole note) — space
                ct = _chord_tones_in_range(root_pc, quality,
                                           BASS_SOLO_LOW, BASS_SOLO_HIGH)
                if ct:
                    current_midi = min(ct, key=lambda n: abs(n - current_midi))
                tick = _humanize_tick(bar_start_tick, 5)
                dur = random.choice([2 * TICKS_PER_QUARTER, tpb - 40])
                vel = _humanize_velocity(vel_base)
                notes.append(NoteEvent(
                    pitch=current_midi, start_tick=tick,
                    duration_ticks=dur, velocity=vel, channel=0,
                ))

        bar_idx += phrase_bars

        # Breathing space between phrases (0-2 bars)
        if bar_idx < total_bars:
            silence = random.choice([0, 1, 1, 2])
            bar_idx = min(bar_idx + silence, total_bars)

    return notes


def _swing_tick_bass(bar_start_tick: int, beat_offset: float, swing: bool) -> int:
    """Apply swing to a beat offset within a bar (bass solo helper)."""
    straight_tick = bar_start_tick + int(beat_offset * TICKS_PER_QUARTER)
    if not swing:
        return straight_tick
    beat_in_pair = beat_offset % 1.0
    if abs(beat_in_pair - 0.5) < 0.01:
        pair_start = int(beat_offset) * TICKS_PER_QUARTER
        swing_pos = int(TICKS_PER_QUARTER * 2 / 3)
        return bar_start_tick + pair_start + swing_pos
    return straight_tick


# ---------------------------------------------------------------------------
# Expression: pitch bends (slides) and CC11 (expression) for bass
# ---------------------------------------------------------------------------

BASS_CHANNEL = 2


def generate_bass_expression(
    notes: List[NoteEvent],
    chords: list,
    channel: int = BASS_CHANNEL,
    beats_per_bar: int = 4,
) -> tuple:
    """Generate pitch bend slides and CC11 expression curves for bass notes.

    Returns:
        (List[PitchBendEvent], List[CCEvent])
    """
    bends: List[PitchBendEvent] = []
    ccs: List[CCEvent] = []

    if not notes:
        return bends, ccs

    # Build a quick chord lookup by tick for chromatic-approach detection
    chord_roots = set()
    for c in chords:
        chord_roots.add(getattr(c, "root_pc", -1))

    for i, note in enumerate(notes):
        dur = note.duration_ticks
        start = note.start_tick

        # --- Pitch bend slides ---
        # 20-40% of notes get a slide-up into the note (more on beat 1)
        beat_in_bar = (start % ticks_per_bar(beats_per_bar)) / TICKS_PER_QUARTER
        is_beat_one = beat_in_bar < 0.3
        slide_prob = 0.35 if is_beat_one else 0.22

        # Chromatic approach: if previous note is 1-2 semitones away, higher slide prob
        if i > 0:
            interval = abs(note.pitch - notes[i - 1].pitch)
            if interval in (1, 2):
                slide_prob += 0.15

        if random.random() < slide_prob:
            # Ramp from -2048 to 0 over ~50 ticks before note attack
            slide_ticks = min(50, max(20, dur // 8))
            ramp_start = max(0, start - slide_ticks)
            steps = max(2, slide_ticks // 10)
            for s in range(steps):
                t = ramp_start + int(s * slide_ticks / steps)
                val = int(-2048 * (1.0 - s / steps))
                bends.append(PitchBendEvent(value=val, start_tick=t, channel=channel))
            # Reset to center at note start
            bends.append(PitchBendEvent(value=0, start_tick=start, channel=channel))
            # Reset again at note end for safety
            bends.append(PitchBendEvent(value=0, start_tick=start + dur, channel=channel))

        # --- CC11 expression envelope ---
        # Only for notes >= 8th note duration
        if dur >= TICKS_PER_8TH:
            # 3-point envelope: attack → peak → decay
            base_expr = max(60, min(120, note.velocity + random.randint(-5, 10)))
            attack_val = max(40, base_expr - 25)
            peak_val = min(127, base_expr + 10)
            decay_val = max(50, base_expr - 15)

            attack_tick = start
            peak_tick = start + dur // 4
            decay_tick = start + dur * 3 // 4

            ccs.append(CCEvent(cc_number=11, value=attack_val, start_tick=attack_tick, channel=channel))
            ccs.append(CCEvent(cc_number=11, value=peak_val, start_tick=peak_tick, channel=channel))
            ccs.append(CCEvent(cc_number=11, value=decay_val, start_tick=decay_tick, channel=channel))

    return bends, ccs