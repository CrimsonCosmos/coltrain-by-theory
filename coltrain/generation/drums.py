"""Swing drum pattern generator for Coltrain.

Generates idiomatic jazz drum patterns with ride cymbal swing,
hi-hat pedal, feathered bass drum, snare comping, and fills.
"""

import math
import random
from typing import List

from . import NoteEvent, TICKS_PER_QUARTER, TICKS_PER_BAR, TICKS_PER_8TH, TICKS_PER_16TH

# ---------------------------------------------------------------------------
# MIDI drum map (General MIDI, channel 9)
# ---------------------------------------------------------------------------

KICK = 36
SNARE = 38
SIDE_STICK = 37
CLOSED_HH = 42
OPEN_HH = 46
RIDE = 51
CRASH = 49
HI_TOM = 50
MID_TOM = 47
LO_TOM = 45
FLOOR_TOM = 43

DRUM_CHANNEL = 9

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _swing_tick(bar_start_tick: int, beat_offset: float, swing: bool) -> int:
    """Convert a beat offset within a bar to an absolute tick, applying swing.

    Swing shifts the "and" of each beat (the offbeat 8th note) from the
    exact midpoint to approximately 2/3 of the way through the beat,
    creating a triplet-based feel.

    Args:
        bar_start_tick: Absolute tick at bar start.
        beat_offset: Beat offset within the bar (0.0 = beat 1, 0.5 = and-of-1, etc.)
        swing: Whether to apply swing displacement.

    Returns:
        Absolute tick position.
    """
    # Determine which beat this falls on and the subdivision
    beat_num = int(beat_offset)
    frac = beat_offset - beat_num

    if swing and abs(frac - 0.5) < 0.01:
        # This is an offbeat 8th — shift to ~2/3 position (swing feel)
        # Add per-note jitter so the swing ratio isn't robotic
        swing_ratio = 0.667 + random.gauss(0, 0.03)
        swing_ratio = max(0.55, min(0.75, swing_ratio))
        tick = bar_start_tick + beat_num * TICKS_PER_QUARTER + int(TICKS_PER_QUARTER * swing_ratio)
    else:
        tick = bar_start_tick + int(beat_offset * TICKS_PER_QUARTER)

    return tick


def _humanize_tick(tick: int, amount: int = 10) -> int:
    """Add slight random timing variation."""
    return max(0, tick + random.randint(-amount, amount))


def _humanize_velocity(vel: int, amount: int = 5) -> int:
    """Add slight random velocity variation, clamped to 1-127."""
    return max(1, min(127, vel + random.randint(-amount, amount)))


def _drum_note(pitch: int, tick: int, duration: int, velocity: int) -> NoteEvent:
    """Create a drum NoteEvent (always channel 9)."""
    return NoteEvent(
        pitch=pitch,
        start_tick=max(0, tick),
        duration_ticks=duration,
        velocity=max(1, min(127, velocity)),
        channel=DRUM_CHANNEL,
    )


# ---------------------------------------------------------------------------
# Pattern components
# ---------------------------------------------------------------------------


def _generate_ride(bar_start_tick: int, swing: bool,
                   intensity: float = 0.5, vel_mult: float = 1.0) -> List[NoteEvent]:
    """Generate the ride cymbal swing pattern for one bar.

    Core pattern:
    - Beats 1, 3: always hit (accented) — sparse bar may drop one
    - And-of-2, and-of-4: always hit (swing pattern)
    - Beats 2, 4: 70% chance (fill between swing hits)
    """
    notes = []

    # Sparse bar chance: inversely scaled by intensity
    sparse = random.random() < (0.20 - intensity * 0.12)

    # Beats 1 and 3 — usually hit, sparse bar may drop one
    for beat in [0.0, 2.0]:
        if sparse and random.random() < 0.4:
            continue  # Drop this hit for a sparser feel
        tick = _swing_tick(bar_start_tick, beat, swing)
        vel = _humanize_velocity(round(random.randint(80, 110) * vel_mult))
        notes.append(_drum_note(RIDE, _humanize_tick(tick, 10), TICKS_PER_8TH, vel))

    # And-of-2 and and-of-4 — the swing pattern
    for beat in [1.5, 3.5]:
        tick = _swing_tick(bar_start_tick, beat, swing)
        vel = _humanize_velocity(round(random.randint(60, 85) * vel_mult))
        notes.append(_drum_note(RIDE, _humanize_tick(tick, 18), TICKS_PER_8TH, vel))

    # Beats 2 and 4 — 70% each
    for beat in [1.0, 3.0]:
        if random.random() < 0.70:
            tick = _swing_tick(bar_start_tick, beat, swing)
            vel = _humanize_velocity(round(random.randint(70, 90) * vel_mult))
            notes.append(_drum_note(RIDE, _humanize_tick(tick, 10), TICKS_PER_8TH, vel))

    # Occasional skip note (16th before a downbeat) at higher intensity
    if intensity > 0.4 and random.random() < 0.10:
        skip_beat = random.choice([3.75, 1.75])
        tick = bar_start_tick + int(skip_beat * TICKS_PER_QUARTER)
        vel = _humanize_velocity(round(random.randint(55, 70) * vel_mult))
        notes.append(_drum_note(RIDE, _humanize_tick(tick, 8), TICKS_PER_16TH, vel))

    return notes


def _generate_hihat(bar_start_tick: int, swing: bool,
                    vel_mult: float = 1.0) -> List[NoteEvent]:
    """Generate hi-hat pedal on beats 2 and 4 (the standard jazz hi-hat pattern)."""
    notes = []
    for beat in [1.0, 3.0]:
        tick = _swing_tick(bar_start_tick, beat, swing)
        vel = _humanize_velocity(round(random.randint(50, 60) * vel_mult))
        notes.append(_drum_note(CLOSED_HH, _humanize_tick(tick, 8), TICKS_PER_8TH, vel))
    return notes


def _generate_kick(bar_start_tick: int, intensity: float, swing: bool) -> List[NoteEvent]:
    """Generate feathered bass drum pattern.

    Ghost notes on all 4 beats with intensity-dependent probability.
    Occasional accent on beat 1.
    """
    notes = []
    # Probability of ghost note scales with intensity: 40% at low, 80% at high
    ghost_prob = 0.4 + intensity * 0.4

    for beat_idx in range(4):
        beat = float(beat_idx)
        tick = _swing_tick(bar_start_tick, beat, swing)

        if beat_idx == 0 and random.random() < 0.30:
            # Accented beat 1
            vel = _humanize_velocity(random.randint(70, 80))
            notes.append(_drum_note(KICK, _humanize_tick(tick, 5), TICKS_PER_QUARTER, vel))
        elif random.random() < ghost_prob:
            # Ghost note — very soft
            vel = _humanize_velocity(random.randint(30, 45))
            notes.append(_drum_note(KICK, _humanize_tick(tick, 8), TICKS_PER_QUARTER, vel))

    return notes


def _generate_snare_comping(bar_start_tick: int, intensity: float,
                             swing: bool) -> List[NoteEvent]:
    """Generate syncopated snare/cross-stick comping hits.

    Number of hits per bar scales with intensity (1-4).
    Positions chosen from typical jazz syncopations.
    """
    notes = []
    num_hits = int(1 + intensity * 3)

    # Available syncopated positions (beat offsets within bar)
    positions = [0.5, 1.0, 1.5, 2.5, 3.0, 3.5]
    random.shuffle(positions)
    chosen_positions = sorted(positions[:num_hits])

    for beat in chosen_positions:
        tick = _swing_tick(bar_start_tick, beat, swing)

        # 30% chance of cross-stick instead of snare
        if random.random() < 0.30:
            pitch = SIDE_STICK
        else:
            pitch = SNARE

        vel = _humanize_velocity(random.randint(55, 75))
        notes.append(_drum_note(pitch, _humanize_tick(tick, 15), TICKS_PER_8TH, vel))

    return notes


# ---------------------------------------------------------------------------
# Fill patterns
# ---------------------------------------------------------------------------


def _fill_descending_toms(bar_start_tick: int, swing: bool) -> List[NoteEvent]:
    """Descending toms fill: HI_TOM -> MID_TOM -> LO_TOM -> FLOOR_TOM.

    8th notes over 4 beats, each tom gets 2 hits.
    """
    notes = []
    tom_sequence = [HI_TOM, HI_TOM, MID_TOM, MID_TOM,
                    LO_TOM, LO_TOM, FLOOR_TOM, FLOOR_TOM]

    for i, tom in enumerate(tom_sequence):
        beat_offset = i * 0.5
        tick = _swing_tick(bar_start_tick, beat_offset, swing)
        # Crescendo across the fill
        base_vel = 70 + i * 5
        vel = _humanize_velocity(min(110, base_vel))
        notes.append(_drum_note(tom, _humanize_tick(tick, 5), TICKS_PER_8TH, vel))

    return notes


def _fill_snare_roll(bar_start_tick: int, swing: bool) -> List[NoteEvent]:
    """16th note snare crescendo roll."""
    notes = []
    num_16ths = 16  # 4 beats * 4 sixteenths

    for i in range(num_16ths):
        beat_offset = i * 0.25
        tick = bar_start_tick + int(beat_offset * TICKS_PER_QUARTER)
        # Crescendo from 50 to 110
        vel_base = 50 + int((110 - 50) * (i / (num_16ths - 1)))
        vel = _humanize_velocity(vel_base)
        notes.append(_drum_note(SNARE, _humanize_tick(tick, 3), TICKS_PER_16TH, vel))

    return notes


def _fill_buildup(bar_start_tick: int, swing: bool) -> List[NoteEvent]:
    """Alternating snare and floor tom 8th notes with crescendo."""
    notes = []
    num_8ths = 8  # 4 beats * 2 eighths

    for i in range(num_8ths):
        beat_offset = i * 0.5
        tick = _swing_tick(bar_start_tick, beat_offset, swing)
        # Alternate between snare and floor tom
        pitch = SNARE if i % 2 == 0 else FLOOR_TOM
        # Crescendo
        vel_base = 60 + int((110 - 60) * (i / (num_8ths - 1)))
        vel = _humanize_velocity(vel_base)
        notes.append(_drum_note(pitch, _humanize_tick(tick, 5), TICKS_PER_8TH, vel))

    return notes


_FILL_FUNCTIONS = [_fill_descending_toms, _fill_snare_roll, _fill_buildup]


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate_drums(total_beats: int, intensity: float = 0.5,
                   swing: bool = True, fill_every: int = 8) -> List[NoteEvent]:
    """Generate a jazz swing drum pattern.

    Layers ride cymbal, hi-hat pedal, feathered bass drum, and snare comping.
    Inserts drum fills at regular intervals.

    Args:
        total_beats: Total number of beats to generate.
        intensity: Drumming intensity 0.0-1.0. Affects snare density,
                   kick probability, and overall energy.
        swing: Whether to apply swing feel to offbeat 8ths.
        fill_every: Insert a drum fill every N bars (0 = no fills).

    Returns:
        List of NoteEvent objects (all on channel 9).
    """
    intensity = max(0.0, min(1.0, intensity))
    total_bars = total_beats // 4
    notes: List[NoteEvent] = []
    need_crash = False  # Track if previous bar was a fill

    # Dynamic arc: sine-wave velocity multiplier over 8-16 bar cycles
    arc_period = random.choice([8, 12, 16])

    for bar_idx in range(total_bars):
        bar_start_tick = bar_idx * TICKS_PER_BAR

        # Dynamic arc: ±10% velocity wave
        arc_phase = (bar_idx % arc_period) / arc_period
        vel_mult = 1.0 + 0.10 * math.sin(arc_phase * 2 * math.pi)

        # Check if this bar should be a fill
        is_fill_bar = (
            fill_every > 0
            and bar_idx > 0
            and (bar_idx + 1) % fill_every == 0
            and bar_idx < total_bars - 1  # Don't fill on last bar
        )

        if need_crash:
            # Crash cymbal on beat 1 after a fill
            tick = _swing_tick(bar_start_tick, 0.0, swing)
            vel = _humanize_velocity(random.randint(100, 115))
            notes.append(_drum_note(CRASH, _humanize_tick(tick, 3), TICKS_PER_QUARTER, vel))
            need_crash = False

        if is_fill_bar:
            # Replace this bar with a fill
            fill_func = random.choice(_FILL_FUNCTIONS)
            notes.extend(fill_func(bar_start_tick, swing))
            need_crash = True
        else:
            # Normal bar: layer all components with dynamic arc
            notes.extend(_generate_ride(bar_start_tick, swing, intensity, vel_mult))
            notes.extend(_generate_hihat(bar_start_tick, swing, vel_mult))
            notes.extend(_generate_kick(bar_start_tick, intensity, swing))
            notes.extend(_generate_snare_comping(bar_start_tick, intensity, swing))

    return notes


def generate_modal_drums(total_beats: int, intensity: float = 0.5,
                         swing: bool = True, fill_every: int = 8) -> List[NoteEvent]:
    """Generate a modal/Elvin Jones-influenced drum pattern.

    Sparse ride (hit on ~70% of positions), floor tom coloring on beats 2&4
    instead of hi-hat, more open feel. Used for modal jazz.
    """
    intensity = max(0.0, min(1.0, intensity))
    total_bars = total_beats // 4
    notes: List[NoteEvent] = []
    need_crash = False

    for bar_idx in range(total_bars):
        bar_start_tick = bar_idx * TICKS_PER_BAR

        is_fill_bar = (
            fill_every > 0
            and bar_idx > 0
            and (bar_idx + 1) % fill_every == 0
            and bar_idx < total_bars - 1
        )

        if need_crash:
            tick = _swing_tick(bar_start_tick, 0.0, swing)
            vel = _humanize_velocity(random.randint(100, 115))
            notes.append(_drum_note(CRASH, _humanize_tick(tick, 3), TICKS_PER_QUARTER, vel))
            need_crash = False

        if is_fill_bar:
            fill_func = random.choice(_FILL_FUNCTIONS)
            notes.extend(fill_func(bar_start_tick, swing))
            need_crash = True
        else:
            # Sparse ride: only ~70% of standard positions
            for beat in [0.0, 2.0]:
                if random.random() < 0.70:
                    tick = _swing_tick(bar_start_tick, beat, swing)
                    vel = _humanize_velocity(random.randint(80, 95))
                    notes.append(_drum_note(RIDE, _humanize_tick(tick, 5), TICKS_PER_8TH, vel))

            for beat in [1.5, 3.5]:
                if random.random() < 0.70:
                    tick = _swing_tick(bar_start_tick, beat, swing)
                    vel = _humanize_velocity(random.randint(70, 80))
                    notes.append(_drum_note(RIDE, _humanize_tick(tick, 8), TICKS_PER_8TH, vel))

            # Floor tom on beats 2 and 4 (instead of hi-hat)
            for beat in [1.0, 3.0]:
                if random.random() < 0.60:
                    tick = _swing_tick(bar_start_tick, beat, swing)
                    vel = _humanize_velocity(random.randint(45, 60))
                    notes.append(_drum_note(FLOOR_TOM, _humanize_tick(tick, 8), TICKS_PER_8TH, vel))

            # Occasional kick — sparser than swing
            for beat_idx in range(4):
                if random.random() < 0.25:
                    tick = _swing_tick(bar_start_tick, float(beat_idx), swing)
                    vel = _humanize_velocity(random.randint(30, 50))
                    notes.append(_drum_note(KICK, _humanize_tick(tick, 8), TICKS_PER_QUARTER, vel))

            # Occasional snare ghost note — very sparse
            if random.random() < 0.3 * intensity:
                beat = random.choice([0.5, 1.5, 2.5, 3.5])
                tick = _swing_tick(bar_start_tick, beat, swing)
                vel = _humanize_velocity(random.randint(35, 50))
                notes.append(_drum_note(SNARE, _humanize_tick(tick, 10), TICKS_PER_8TH, vel))

    return notes