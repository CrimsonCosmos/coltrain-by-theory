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

    # And-of-2 and and-of-4 — the swing pattern (occasional drops for space)
    for beat in [1.5, 3.5]:
        if random.random() < 0.15:
            continue  # Drop this swing hit — creates tension/space
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
                    intensity: float = 0.5, vel_mult: float = 1.0) -> List[NoteEvent]:
    """Generate hi-hat pedal with musical variation.

    Base pattern: beats 2 and 4.
    Variations: skip (10%), open hat (5%), extra hit on and-of-2/4 (8%).
    """
    notes = []
    for beat in [1.0, 3.0]:
        # 10% chance to skip this hit
        if random.random() < 0.10:
            continue

        # 5% chance of open hi-hat instead of closed
        if random.random() < 0.05:
            pitch = OPEN_HH
            vel = _humanize_velocity(round(random.randint(55, 70) * vel_mult))
            dur = TICKS_PER_QUARTER
        else:
            pitch = CLOSED_HH
            vel = _humanize_velocity(round(random.randint(50, 60) * vel_mult))
            dur = TICKS_PER_8TH

        tick = _swing_tick(bar_start_tick, beat, swing)
        notes.append(_drum_note(pitch, _humanize_tick(tick, 8), dur, vel))

        # 8% chance of extra hit on the "and" (adds shuffle feel)
        if random.random() < 0.08:
            and_beat = beat + 0.5
            and_tick = _swing_tick(bar_start_tick, and_beat, swing)
            and_vel = _humanize_velocity(round(random.randint(35, 50) * vel_mult))
            notes.append(_drum_note(CLOSED_HH, _humanize_tick(and_tick, 10),
                                    TICKS_PER_8TH, and_vel))

    return notes


_KICK_PATTERNS = [
    [(0.0, True)],                                # Just beat 1
    [(0.0, True), (2.0, False)],                   # Beats 1 & 3
    [(0.0, True), (1.5, False)],                   # Beat 1 + and-of-2
    [(0.0, True), (2.5, False), (3.0, False)],     # Beat 1 + syncopated cluster
    [(0.0, True), (1.0, False), (3.0, False)],     # Beats 1, 2, 4
    [(2.0, False), (3.5, False)],                  # Anticipation pattern
    [(0.0, True), (1.5, False), (3.0, False)],     # Beat 1 + cross-rhythm
]


def _generate_kick(bar_start_tick: int, intensity: float, swing: bool,
                   vel_mult: float = 1.0) -> List[NoteEvent]:
    """Generate bass drum pattern using musical kick patterns.

    Instead of uniform probability on every beat, choose from pre-defined
    syncopated patterns that create musical groupings.
    """
    notes = []

    # At low intensity, pick sparser patterns; at high, denser
    if intensity < 0.3:
        pool = _KICK_PATTERNS[:2]
    elif intensity < 0.6:
        pool = _KICK_PATTERNS[:5]
    else:
        pool = _KICK_PATTERNS

    # Chance of no kick at all (more likely at low intensity)
    silence_prob = max(0.05, 0.35 - intensity * 0.35)
    if random.random() < silence_prob:
        return notes

    pattern = random.choice(pool)
    for beat_offset, is_accent in pattern:
        tick = _swing_tick(bar_start_tick, beat_offset, swing)
        if is_accent:
            vel = _humanize_velocity(round(random.randint(65, 80) * vel_mult))
        else:
            vel = _humanize_velocity(round(random.randint(30, 50) * vel_mult))
        notes.append(_drum_note(KICK, _humanize_tick(tick, 8), TICKS_PER_QUARTER, vel))

    return notes


def _generate_snare_comping(bar_start_tick: int, intensity: float,
                             swing: bool, prev_positions: list = None,
                             vel_mult: float = 1.0):
    """Generate syncopated snare/cross-stick comping hits with phrase memory.

    Number of hits per bar scales with intensity (1-4).
    40% chance to repeat similar placement from previous bar.

    Returns:
        Tuple of (notes, chosen_positions) for phrase memory.
    """
    notes = []
    num_hits = int(1 + intensity * 3)

    # Available syncopated positions (beat offsets within bar)
    all_positions = [0.5, 1.0, 1.5, 2.5, 3.0, 3.5]

    if prev_positions and random.random() < 0.40:
        # Repeat similar placement from last bar (phrase continuity)
        chosen_positions = []
        for pos in prev_positions:
            varied = pos + random.choice([-0.5, 0.0, 0.0, 0.5])
            if varied in all_positions and varied not in chosen_positions:
                chosen_positions.append(varied)
        # Fill remaining from pool
        remaining = [p for p in all_positions if p not in chosen_positions]
        while len(chosen_positions) < num_hits and remaining:
            pick = random.choice(remaining)
            chosen_positions.append(pick)
            remaining.remove(pick)
        chosen_positions = sorted(chosen_positions[:num_hits])
    else:
        random.shuffle(all_positions)
        chosen_positions = sorted(all_positions[:num_hits])

    for beat in chosen_positions:
        tick = _swing_tick(bar_start_tick, beat, swing)

        # 30% chance of cross-stick instead of snare
        if random.random() < 0.30:
            pitch = SIDE_STICK
        else:
            pitch = SNARE

        vel = _humanize_velocity(round(random.randint(55, 75) * vel_mult))
        notes.append(_drum_note(pitch, _humanize_tick(tick, 15), TICKS_PER_8TH, vel))

    return notes, chosen_positions


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


def _fill_triplet_toms(bar_start_tick: int, swing: bool) -> List[NoteEvent]:
    """Triplet fill across toms — jazz 12/8 feel."""
    notes = []
    toms = [HI_TOM, MID_TOM, LO_TOM]
    triplet_dur = TICKS_PER_QUARTER // 3
    for i in range(12):  # 4 beats x 3 triplets
        beat_offset = i / 3.0
        tick = bar_start_tick + int(beat_offset * TICKS_PER_QUARTER)
        tom = toms[i % 3]
        vel_base = 60 + int(45 * (i / 11))
        notes.append(_drum_note(tom, _humanize_tick(tick, 5), triplet_dur,
                                _humanize_velocity(vel_base)))
    return notes


def _fill_single_stroke_roll(bar_start_tick: int, swing: bool) -> List[NoteEvent]:
    """Alternating snare single-stroke roll: 8ths accelerating to 16ths."""
    notes = []
    # First 2 beats: 8th notes
    for i in range(4):
        tick = bar_start_tick + int(i * 0.5 * TICKS_PER_QUARTER)
        notes.append(_drum_note(SNARE, _humanize_tick(tick, 3), TICKS_PER_8TH,
                                _humanize_velocity(65 + i * 5)))
    # Last 2 beats: 16th notes
    for i in range(8):
        tick = bar_start_tick + 2 * TICKS_PER_QUARTER + int(i * 0.25 * TICKS_PER_QUARTER)
        notes.append(_drum_note(SNARE, _humanize_tick(tick, 3), TICKS_PER_16TH,
                                _humanize_velocity(80 + i * 4)))
    return notes


def _fill_sparse_accents(bar_start_tick: int, swing: bool) -> List[NoteEvent]:
    """Sparse fill: just 3-4 big accents with space. Musical restraint."""
    notes = []
    num_accents = random.randint(3, 4)
    accent_beats = sorted(random.sample([0.0, 1.0, 1.5, 2.0, 3.0, 3.5], k=num_accents))
    drums_pool = [SNARE, FLOOR_TOM, HI_TOM, MID_TOM]
    for beat in accent_beats:
        tick = _swing_tick(bar_start_tick, beat, swing)
        drum = random.choice(drums_pool)
        notes.append(_drum_note(drum, _humanize_tick(tick, 5), TICKS_PER_8TH,
                                _humanize_velocity(random.randint(85, 110))))
    return notes


_FILL_FUNCTIONS = [
    _fill_descending_toms, _fill_snare_roll, _fill_buildup,
    _fill_triplet_toms, _fill_single_stroke_roll, _fill_sparse_accents,
]


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
    prev_snare_positions: list = []

    for bar_idx in range(total_bars):
        bar_start_tick = bar_idx * TICKS_PER_BAR

        # Dynamic arc: ±20% velocity wave
        arc_phase = (bar_idx % arc_period) / arc_period
        vel_mult = 1.0 + 0.20 * math.sin(arc_phase * 2 * math.pi)

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
            notes.extend(_generate_hihat(bar_start_tick, swing, intensity, vel_mult))
            notes.extend(_generate_kick(bar_start_tick, intensity, swing, vel_mult))
            snare_notes, prev_snare_positions = _generate_snare_comping(
                bar_start_tick, intensity, swing, prev_snare_positions, vel_mult)
            notes.extend(snare_notes)

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