"""Swing drum pattern generator for Coltrain.

Generates idiomatic jazz drum patterns with ride cymbal swing,
hi-hat pedal, feathered bass drum, snare comping, and fills.
"""

import math
import random
from typing import List, Optional

from . import NoteEvent, CCEvent, BarFeel, TICKS_PER_QUARTER, TICKS_PER_BAR, TICKS_PER_8TH, TICKS_PER_16TH, ticks_per_bar

# Module-level rhythmic feel state (set per bar in generator loop)
_current_feel: Optional[BarFeel] = None

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

RIDE_BELL = 53
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
        depth = _current_feel.swing_depth if _current_feel else 1.0
        swing_ratio = 0.667 + random.gauss(0, 0.03)
        swing_ratio = max(0.55, min(0.75, swing_ratio))
        # Scale swing displacement by depth
        straight_tick = beat_num * TICKS_PER_QUARTER + TICKS_PER_8TH
        swung_tick = beat_num * TICKS_PER_QUARTER + int(TICKS_PER_QUARTER * swing_ratio)
        tick = bar_start_tick + straight_tick + int((swung_tick - straight_tick) * depth)
    else:
        tick = bar_start_tick + int(beat_offset * TICKS_PER_QUARTER)

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
                   intensity: float = 0.5, vel_mult: float = 1.0,
                   beats_per_bar: int = 4) -> List[NoteEvent]:
    """Generate the ride cymbal swing pattern for one bar.

    Core pattern (4/4):
    - Beats 1, 3: always hit (accented)
    - And-of-2, and-of-4: always hit (swing pattern)
    - Beats 2, 4: 70% chance
    For 5/4 (3+2): accents on 1, 3; swing on 1.75, 3.75; secondary on 2, 4
    For 7/4 (4+3): accents on 1, 3, 5; swing on 1.75, 3.75, 5.75; secondary on 2, 4, 6
    """
    notes = []
    sparse = random.random() < (0.20 - intensity * 0.12)

    # Build beat lists based on meter
    if beats_per_bar == 5:
        accent_beats = [0.0, 2.0, 3.0]       # 3+2 grouping accents
        swing_beats = [1.5, 3.5, 4.5]         # Swung offbeats
        secondary_beats = [1.0, 4.0]           # Fill beats
    elif beats_per_bar == 7:
        accent_beats = [0.0, 2.0, 4.0]       # 4+3 grouping accents
        swing_beats = [1.5, 3.5, 5.5]         # Swung offbeats
        secondary_beats = [1.0, 3.0, 5.0, 6.0]  # Fill beats
    else:
        accent_beats = [0.0, 2.0]
        swing_beats = [1.5, 3.5]
        secondary_beats = [1.0, 3.0]

    # Accent beats — usually hit, sparse bar may drop one
    for beat in accent_beats:
        if sparse and random.random() < 0.4:
            continue
        tick = _swing_tick(bar_start_tick, beat, swing)
        vel = _humanize_velocity(round(random.randint(80, 110) * vel_mult))
        notes.append(_drum_note(RIDE, _humanize_tick(tick, 10), TICKS_PER_8TH, vel))

    # Swing offbeats (occasional drops for space)
    for beat in swing_beats:
        if random.random() < 0.15:
            continue
        tick = _swing_tick(bar_start_tick, beat, swing)
        vel = _humanize_velocity(round(random.randint(60, 85) * vel_mult))
        notes.append(_drum_note(RIDE, _humanize_tick(tick, 18), TICKS_PER_8TH, vel))

    # Secondary beats — 70% each
    for beat in secondary_beats:
        if random.random() < 0.70:
            tick = _swing_tick(bar_start_tick, beat, swing)
            vel = _humanize_velocity(round(random.randint(70, 90) * vel_mult))
            notes.append(_drum_note(RIDE, _humanize_tick(tick, 10), TICKS_PER_8TH, vel))

    # Skip note at higher intensity
    if intensity > 0.4 and random.random() < 0.10:
        skip_beat = beats_per_bar - 0.25
        tick = bar_start_tick + int(skip_beat * TICKS_PER_QUARTER)
        vel = _humanize_velocity(round(random.randint(55, 70) * vel_mult))
        notes.append(_drum_note(RIDE, _humanize_tick(tick, 8), TICKS_PER_16TH, vel))

    return notes


def _generate_hihat(bar_start_tick: int, swing: bool,
                    intensity: float = 0.5, vel_mult: float = 1.0,
                    beats_per_bar: int = 4) -> List[NoteEvent]:
    """Generate hi-hat pedal with musical variation.

    Base pattern: beats 2 and 4 (4/4), beats 2, 4, 5 (5/4), beats 2, 4, 6 (7/4).
    """
    notes = []
    if beats_per_bar == 5:
        hh_beats = [1.0, 3.0, 4.0]
    elif beats_per_bar == 7:
        hh_beats = [1.0, 3.0, 5.0]
    else:
        hh_beats = [1.0, 3.0]

    for beat in hh_beats:
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

_KICK_PATTERNS_5 = [
    [(0.0, True)],                                 # Just beat 1
    [(0.0, True), (3.0, False)],                   # Beat 1 + 2-group start
    [(0.0, True), (4.0, False)],                   # Beat 1 + beat 5
    [(0.0, True), (2.0, False), (4.0, False)],     # Spread across bar
]

_KICK_PATTERNS_7 = [
    [(0.0, True)],                                           # Just beat 1
    [(0.0, True), (4.0, False)],                             # Beat 1 + 3-group start
    [(0.0, True), (3.0, False), (5.0, False)],               # Spread
    [(0.0, True), (2.0, False), (4.0, False), (6.0, False)], # Full spread
]


def _generate_kick(bar_start_tick: int, intensity: float, swing: bool,
                   vel_mult: float = 1.0, beats_per_bar: int = 4) -> List[NoteEvent]:
    """Generate bass drum pattern using musical kick patterns.

    Instead of uniform probability on every beat, choose from pre-defined
    syncopated patterns that create musical groupings.
    """
    notes = []

    # Select meter-specific pattern table
    if beats_per_bar == 5:
        patterns = _KICK_PATTERNS_5
    elif beats_per_bar == 7:
        patterns = _KICK_PATTERNS_7
    else:
        patterns = _KICK_PATTERNS

    # At low intensity, pick sparser patterns; at high, denser
    if intensity < 0.3:
        pool = patterns[:2]
    elif intensity < 0.6:
        pool = patterns[:max(2, len(patterns) - 1)]
    else:
        pool = patterns

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
                             vel_mult: float = 1.0,
                             beats_per_bar: int = 4):
    """Generate syncopated snare/cross-stick comping hits with phrase memory.

    Number of hits per bar scales with intensity (1-4).
    40% chance to repeat similar placement from previous bar.

    Returns:
        Tuple of (notes, chosen_positions) for phrase memory.
    """
    notes = []
    num_hits = int(1 + intensity * 3)

    # Available syncopated positions (beat offsets within bar)
    if beats_per_bar == 5:
        all_positions = [0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5]
    elif beats_per_bar == 7:
        all_positions = [0.5, 1.0, 1.5, 2.5, 3.0, 3.5, 4.5, 5.0, 5.5, 6.0, 6.5]
    else:
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


def _generate_ghost_notes(bar_start_tick: int, intensity: float,
                          swing: bool, vel_mult: float = 1.0,
                          beats_per_bar: int = 4) -> List[NoteEvent]:
    """Generate soft snare ghost notes on offbeat 16ths.

    Ghost notes are the quiet "chatter" between accents that gives jazz
    drumming its human feel. Density scales with intensity.
    """
    notes = []
    # Available ghost positions: offbeat 16ths (avoiding main beats and swing 8ths)
    ghost_positions = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75]
    if beats_per_bar >= 5:
        ghost_positions.extend([4.25, 4.75])
    if beats_per_bar >= 7:
        ghost_positions.extend([5.25, 5.75, 6.25, 6.75])
    # Probability of each ghost note scales with intensity
    ghost_prob = 0.08 + intensity * 0.18  # 8% at low, 26% at high

    for beat_offset in ghost_positions:
        if random.random() < ghost_prob:
            tick = bar_start_tick + int(beat_offset * TICKS_PER_QUARTER)
            vel = _humanize_velocity(round(random.randint(15, 30) * vel_mult), amount=4)
            notes.append(_drum_note(SNARE, _humanize_tick(tick, 12),
                                    TICKS_PER_16TH, vel))

    return notes


# ---------------------------------------------------------------------------
# Fill patterns
# ---------------------------------------------------------------------------


def _fill_descending_toms(bar_start_tick: int, swing: bool,
                          beats_per_bar: int = 4) -> List[NoteEvent]:
    """Descending toms fill: HI_TOM -> MID_TOM -> LO_TOM -> FLOOR_TOM.

    8th notes over the bar, each tom gets proportional hits.
    """
    notes = []
    num_8ths = beats_per_bar * 2
    toms = [HI_TOM, MID_TOM, LO_TOM, FLOOR_TOM]

    for i in range(num_8ths):
        beat_offset = i * 0.5
        tick = _swing_tick(bar_start_tick, beat_offset, swing)
        # Distribute toms evenly across the fill
        tom = toms[min(i * len(toms) // num_8ths, len(toms) - 1)]
        # Crescendo across the fill
        base_vel = 70 + i * (40 // max(1, num_8ths - 1))
        vel = _humanize_velocity(min(110, base_vel))
        notes.append(_drum_note(tom, _humanize_tick(tick, 5), TICKS_PER_8TH, vel))

    return notes


def _fill_snare_roll(bar_start_tick: int, swing: bool,
                     beats_per_bar: int = 4) -> List[NoteEvent]:
    """16th note snare crescendo roll."""
    notes = []
    num_16ths = beats_per_bar * 4

    for i in range(num_16ths):
        beat_offset = i * 0.25
        tick = bar_start_tick + int(beat_offset * TICKS_PER_QUARTER)
        # Crescendo from 50 to 110
        vel_base = 50 + int((110 - 50) * (i / max(1, num_16ths - 1)))
        vel = _humanize_velocity(vel_base)
        notes.append(_drum_note(SNARE, _humanize_tick(tick, 3), TICKS_PER_16TH, vel))

    return notes


def _fill_buildup(bar_start_tick: int, swing: bool,
                  beats_per_bar: int = 4) -> List[NoteEvent]:
    """Alternating snare and floor tom 8th notes with crescendo."""
    notes = []
    num_8ths = beats_per_bar * 2

    for i in range(num_8ths):
        beat_offset = i * 0.5
        tick = _swing_tick(bar_start_tick, beat_offset, swing)
        # Alternate between snare and floor tom
        pitch = SNARE if i % 2 == 0 else FLOOR_TOM
        # Crescendo
        vel_base = 60 + int((110 - 60) * (i / max(1, num_8ths - 1)))
        vel = _humanize_velocity(vel_base)
        notes.append(_drum_note(pitch, _humanize_tick(tick, 5), TICKS_PER_8TH, vel))

    return notes


def _fill_triplet_toms(bar_start_tick: int, swing: bool,
                       beats_per_bar: int = 4) -> List[NoteEvent]:
    """Triplet fill across toms — jazz 12/8 feel."""
    notes = []
    toms = [HI_TOM, MID_TOM, LO_TOM]
    triplet_dur = TICKS_PER_QUARTER // 3
    num_triplets = beats_per_bar * 3
    for i in range(num_triplets):
        beat_offset = i / 3.0
        tick = bar_start_tick + int(beat_offset * TICKS_PER_QUARTER)
        tom = toms[i % 3]
        vel_base = 60 + int(45 * (i / max(1, num_triplets - 1)))
        notes.append(_drum_note(tom, _humanize_tick(tick, 5), triplet_dur,
                                _humanize_velocity(vel_base)))
    return notes


def _fill_single_stroke_roll(bar_start_tick: int, swing: bool,
                              beats_per_bar: int = 4) -> List[NoteEvent]:
    """Alternating snare single-stroke roll: 8ths accelerating to 16ths."""
    notes = []
    half = beats_per_bar // 2
    # First half: 8th notes
    for i in range(half * 2):
        tick = bar_start_tick + int(i * 0.5 * TICKS_PER_QUARTER)
        notes.append(_drum_note(SNARE, _humanize_tick(tick, 3), TICKS_PER_8TH,
                                _humanize_velocity(65 + i * 5)))
    # Second half: 16th notes
    remaining = beats_per_bar - half
    for i in range(remaining * 4):
        tick = bar_start_tick + half * TICKS_PER_QUARTER + int(i * 0.25 * TICKS_PER_QUARTER)
        notes.append(_drum_note(SNARE, _humanize_tick(tick, 3), TICKS_PER_16TH,
                                _humanize_velocity(80 + i * 3)))
    return notes


def _fill_sparse_accents(bar_start_tick: int, swing: bool,
                          beats_per_bar: int = 4) -> List[NoteEvent]:
    """Sparse fill: just 3-4 big accents with space. Musical restraint."""
    notes = []
    # Build accent pool based on meter
    accent_pool = [0.0, 1.0, 1.5, 2.0, 3.0, 3.5]
    if beats_per_bar >= 5:
        accent_pool.extend([4.0, 4.5])
    if beats_per_bar >= 7:
        accent_pool.extend([5.0, 5.5, 6.0, 6.5])
    num_accents = random.randint(3, min(4, len(accent_pool)))
    accent_beats = sorted(random.sample(accent_pool, k=num_accents))
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
                   swing: bool = True, fill_every: int = 8,
                   bar_intensities: Optional[List[float]] = None,
                   bar_context: Optional[list] = None,
                   bar_feel: Optional[list] = None,
                   beats_per_bar: int = 4) -> List[NoteEvent]:
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
    tpb = ticks_per_bar(beats_per_bar)
    total_bars = total_beats // beats_per_bar
    notes: List[NoteEvent] = []
    need_crash = False  # Track if previous bar was a fill

    # Dynamic arc: sine-wave velocity multiplier over 8-16 bar cycles
    arc_period = random.choice([8, 12, 16])
    prev_snare_positions: list = []

    # Multi-bar kick phrasing: hold a kick pattern for 2-4 bars
    kick_pattern = None
    kick_hold = 0

    for bar_idx in range(total_bars):
        global _current_feel
        _current_feel = bar_feel[bar_idx] if bar_feel and bar_idx < len(bar_feel) else None

        bar_start_tick = bar_idx * tpb

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
            notes.extend(fill_func(bar_start_tick, swing, beats_per_bar=beats_per_bar))
            need_crash = True
            kick_hold = 0  # Reset kick pattern after fill
        else:
            # Per-bar reactive intensity
            local_intensity = (bar_intensities[bar_idx]
                               if bar_intensities and bar_idx < len(bar_intensities)
                               else intensity)

            # Inter-instrument reactivity
            ctx = bar_context[bar_idx] if bar_context and bar_idx < len(bar_context) else None
            react_vel_mult = 1.0
            react_ghost_mult = 1.0
            form_section = ""
            harmonic_chord_count = 1
            is_key_change = False
            if ctx is not None:
                if ctx.density > 0.7:
                    # Lead is busy — soften drums
                    react_vel_mult = 0.85
                elif ctx.has_silence:
                    # Lead laying out — drums can fill more
                    react_ghost_mult = 1.3
                form_section = ctx.form_section
                harmonic_chord_count = ctx.chord_count
                is_key_change = ctx.is_key_change

            # --- Form section awareness ---
            is_bridge = form_section == "B"
            if is_bridge:
                # Bridge: more open, colorful playing
                react_ghost_mult *= 1.2  # More conversational
            if harmonic_chord_count >= 2:
                # Fast harmonic rhythm: less ghost chatter, more precision
                react_ghost_mult *= 0.7

            # --- Key center arrival accent ---
            if is_key_change:
                # Mark modulation with ride bell + kick on beat 1
                tick = _swing_tick(bar_start_tick, 0.0, swing)
                vel = _humanize_velocity(round(random.randint(95, 115) * vel_mult))
                notes.append(_drum_note(RIDE_BELL, _humanize_tick(tick, 5),
                                        TICKS_PER_8TH, vel))
                if random.random() < 0.60:
                    notes.append(_drum_note(KICK, _humanize_tick(tick, 8),
                                            TICKS_PER_QUARTER,
                                            _humanize_velocity(round(random.randint(70, 85) * vel_mult))))
            else:
                # Ride bell accent on beat 1
                # Bridge: 30% probability (more open/colorful)
                # Normal: 15%
                bell_prob = 0.30 if is_bridge else 0.15
                if random.random() < bell_prob:
                    tick = _swing_tick(bar_start_tick, 0.0, swing)
                    vel = _humanize_velocity(round(random.randint(90, 110) * vel_mult))
                    notes.append(_drum_note(RIDE_BELL, _humanize_tick(tick, 5),
                                            TICKS_PER_8TH, vel))

            # Normal bar: layer all components with dynamic arc
            notes.extend(_generate_ride(bar_start_tick, swing, local_intensity,
                                        vel_mult * react_vel_mult,
                                        beats_per_bar=beats_per_bar))
            notes.extend(_generate_hihat(bar_start_tick, swing, local_intensity,
                                         vel_mult * react_vel_mult,
                                         beats_per_bar=beats_per_bar))

            # Multi-bar kick phrasing
            # Bridge: prefer sparser kick patterns
            kick_intensity = local_intensity * 0.7 if is_bridge else local_intensity
            if kick_hold <= 0:
                kick_pattern = _generate_kick(bar_start_tick, kick_intensity, swing,
                                              vel_mult, beats_per_bar=beats_per_bar)
                kick_hold = random.randint(1, 3)
            else:
                # Replay same pattern with slight variation
                kick_pattern = _replay_kick_pattern(
                    kick_pattern, bar_start_tick, vel_mult,
                    beats_per_bar=beats_per_bar)
            kick_hold -= 1
            notes.extend(kick_pattern)

            # Snare comping: bridge gets 1 extra hit for busier feel
            snare_intensity = min(1.0, local_intensity + 0.15) if is_bridge else local_intensity
            snare_notes, prev_snare_positions = _generate_snare_comping(
                bar_start_tick, snare_intensity, swing, prev_snare_positions, vel_mult,
                beats_per_bar=beats_per_bar)
            notes.extend(snare_notes)

            # Ghost notes layer (more active when lead lays out)
            notes.extend(_generate_ghost_notes(
                bar_start_tick, local_intensity, swing,
                vel_mult * react_ghost_mult, beats_per_bar=beats_per_bar))

    return notes


def _replay_kick_pattern(prev_pattern: List[NoteEvent], new_bar_tick: int,
                         vel_mult: float,
                         beats_per_bar: int = 4) -> List[NoteEvent]:
    """Replay a kick pattern at a new bar position with slight timing/velocity variation."""
    if not prev_pattern:
        return []
    tpb = ticks_per_bar(beats_per_bar)
    # Get relative positions from the pattern
    first_tick = prev_pattern[0].start_tick
    # Estimate the bar start of the original pattern
    orig_bar_tick = (first_tick // tpb) * tpb
    result = []
    for note in prev_pattern:
        offset = note.start_tick - orig_bar_tick
        new_tick = new_bar_tick + offset + random.randint(-8, 8)
        new_vel = max(1, min(127, round(note.velocity * vel_mult / max(0.5, vel_mult))
                             + random.randint(-5, 5)))
        result.append(_drum_note(KICK, max(0, new_tick),
                                 note.duration_ticks, new_vel))
    return result


def _generate_brush_stir(bar_start_tick: int, intensity: float = 0.5,
                          vel_mult: float = 1.0,
                          beats_per_bar: int = 4) -> List[NoteEvent]:
    """Generate continuous brush stirring on snare — long sustained notes.

    Uses few, long-duration notes so the brush sample rings as a continuous
    wash rather than discrete staccato taps.  Velocity range expands with
    intensity for dynamic expression.
    """
    notes = []

    # 2-4 notes per bar depending on intensity (fewer = smoother wash)
    num_notes = 2 if intensity < 0.4 else (3 if intensity < 0.7 else 4)

    # Each note lasts long enough to overlap into the next
    note_spacing = float(beats_per_bar) / num_notes  # beats between note onsets
    note_dur = int(note_spacing * TICKS_PER_QUARTER * 1.5)  # 50% overlap

    # Velocity range expands with intensity
    vel_lo = int(12 + intensity * 8)    # 12-20
    vel_hi = int(24 + intensity * 18)   # 24-42

    for i in range(num_notes):
        beat_offset = i * note_spacing
        tick = bar_start_tick + int(beat_offset * TICKS_PER_QUARTER)

        # Sine-wave velocity: one cycle per bar (slow brush circle)
        phase = (i / num_notes) * 2 * math.pi
        sine_mod = 0.5 + 0.5 * math.sin(phase)
        vel = int(vel_lo + sine_mod * (vel_hi - vel_lo))
        vel = min(50, round(vel * vel_mult))
        vel = _humanize_velocity(vel, amount=3)

        notes.append(_drum_note(
            SNARE, _humanize_tick(tick, 20), note_dur, vel,
        ))

    return notes


def _generate_brush_accents(bar_start_tick: int, intensity: float = 0.5,
                             vel_mult: float = 1.0,
                             beats_per_bar: int = 4) -> List[NoteEvent]:
    """Generate side-stick brush accents on beats 2 and 4 (and group boundaries).

    Lighter than stick cross-rim shots. Occasional extra offbeat accent
    at higher intensities.
    """
    notes = []
    # Accent beats match the hi-hat positions for the meter
    if beats_per_bar == 5:
        accent_beats = [1.0, 3.0, 4.0]
    elif beats_per_bar == 7:
        accent_beats = [1.0, 3.0, 5.0]
    else:
        accent_beats = [1.0, 3.0]

    for beat in accent_beats:
        # 12% skip chance
        if random.random() < 0.12:
            continue
        tick = _swing_tick(bar_start_tick, beat, True)
        vel = _humanize_velocity(round(random.randint(45, 65) * vel_mult), amount=4)
        notes.append(_drum_note(
            SIDE_STICK, _humanize_tick(tick, 10), TICKS_PER_8TH, vel,
        ))

    # Extra offbeat accent at higher intensity
    if intensity > 0.4 and random.random() < 0.20:
        extra_pool = [0.5, 1.5, 2.5, 3.5]
        if beats_per_bar >= 5:
            extra_pool.append(4.5)
        if beats_per_bar >= 7:
            extra_pool.extend([5.5, 6.5])
        extra_beat = random.choice(extra_pool)
        tick = _swing_tick(bar_start_tick, extra_beat, True)
        vel = _humanize_velocity(round(random.randint(30, 45) * vel_mult), amount=4)
        notes.append(_drum_note(
            SIDE_STICK, _humanize_tick(tick, 12), TICKS_PER_8TH, vel,
        ))

    return notes


# ---------------------------------------------------------------------------
# Brush-specific kick patterns and fills
# ---------------------------------------------------------------------------

_BRUSH_KICK_PATTERNS = [
    [(0.0, True)],                          # Just beat 1 (feathered)
    [(0.0, True), (2.0, False)],            # Beats 1 & 3
    [(0.0, True), (2.5, False)],            # Beat 1 + anticipation
]

_BRUSH_KICK_PATTERNS_5 = [
    [(0.0, True)],                          # Just beat 1
    [(0.0, True), (3.0, False)],            # Beat 1 + 2-group start
    [(0.0, True), (2.0, False)],            # Beat 1 + beat 3
]

_BRUSH_KICK_PATTERNS_7 = [
    [(0.0, True)],                          # Just beat 1
    [(0.0, True), (4.0, False)],            # Beat 1 + 3-group start
    [(0.0, True), (2.0, False), (4.0, False)],  # Spread
]


def _fill_brush_sweep(bar_start_tick: int, swing: bool,
                      beats_per_bar: int = 4) -> List[NoteEvent]:
    """Brush sweep fill — arc-shaped long notes on snare."""
    notes = []
    # 6 notes across the bar with long durations (wash, not taps)
    num_notes = 6 if beats_per_bar <= 5 else 8
    dur = TICKS_PER_QUARTER  # each note rings for a full beat
    for i in range(num_notes):
        beat_offset = i * (float(beats_per_bar) / num_notes)
        tick = bar_start_tick + int(beat_offset * TICKS_PER_QUARTER)
        # Arc shape: crescendo first half, decrescendo second
        mid = (num_notes - 1) / 2.0
        arc = 1.0 - abs(i - mid) / max(1, mid)
        vel = int(18 + arc * 30)  # 18-48 range
        notes.append(_drum_note(SNARE, _humanize_tick(tick, 12), dur,
                                _humanize_velocity(vel, amount=3)))
    return notes


def _fill_brush_press(bar_start_tick: int, swing: bool,
                      beats_per_bar: int = 4) -> List[NoteEvent]:
    """Brush press fill — gentle wash first half, pressing harder second half."""
    notes = []
    dur = TICKS_PER_QUARTER  # long ring per note
    half = beats_per_bar // 2
    # Gentle wash on first half
    for i in range(half):
        beat = float(i)
        tick = bar_start_tick + int(beat * TICKS_PER_QUARTER)
        vel = int(16 + i * 6)  # 16-22+
        notes.append(_drum_note(SNARE, _humanize_tick(tick, 15), dur,
                                _humanize_velocity(vel, amount=3)))
    # Press harder second half (3 notes, crescendo)
    press_start = float(half)
    press_beats = [press_start, press_start + 0.75, press_start + 1.5]
    for i, beat in enumerate(press_beats):
        if beat >= beats_per_bar:
            break
        tick = bar_start_tick + int(beat * TICKS_PER_QUARTER)
        vel = int(28 + i * 12)  # 28-52
        notes.append(_drum_note(SNARE, _humanize_tick(tick, 8), dur,
                                _humanize_velocity(vel, amount=3)))
    # Side stick accent near the end
    side_beat = float(beats_per_bar - 1)
    tick = _swing_tick(bar_start_tick, side_beat, swing)
    notes.append(_drum_note(SIDE_STICK, _humanize_tick(tick, 5), TICKS_PER_8TH,
                            _humanize_velocity(45, amount=5)))
    return notes


def _fill_brush_sparse(bar_start_tick: int, swing: bool,
                        beats_per_bar: int = 4) -> List[NoteEvent]:
    """Minimal brush fill: 2-3 side-stick accents with space."""
    notes = []
    beat_pool = [0.5, 1.0, 2.0, 2.5, 3.0, 3.5]
    if beats_per_bar >= 5:
        beat_pool.extend([4.0, 4.5])
    if beats_per_bar >= 7:
        beat_pool.extend([5.0, 5.5, 6.0, 6.5])
    num = random.randint(2, 3)
    beats = sorted(random.sample(beat_pool, k=min(num, len(beat_pool))))
    for beat in beats:
        tick = _swing_tick(bar_start_tick, beat, swing)
        notes.append(_drum_note(SIDE_STICK, _humanize_tick(tick, 8), TICKS_PER_8TH,
                                _humanize_velocity(random.randint(45, 65))))
    return notes


_BRUSH_FILL_FUNCTIONS = [_fill_brush_sweep, _fill_brush_press, _fill_brush_sparse]


def generate_brushes_drums(total_beats: int, intensity: float = 0.5,
                            swing: bool = True, fill_every: int = 16,
                            bar_intensities: Optional[List[float]] = None,
                            bar_context: Optional[list] = None,
                            bar_feel: Optional[list] = None,
                            beats_per_bar: int = 4) -> List[NoteEvent]:
    """Generate a brush drum pattern — intimate piano trio style.

    Continuous snare stirring + side-stick accents on 2 & 4 + feathered kick.
    No ride cymbal, no crash cymbals. Responds to melody dynamics and
    harmonic context via reactive parameters.

    Args:
        total_beats: Total number of beats to generate.
        intensity: Drumming intensity 0.0-1.0.
        swing: Accepted for API compatibility (brushes are always swung implicitly).
        fill_every: Insert a brush fill every N bars (0 = no fills). Default 16.
        bar_intensities: Per-bar reactive intensity from melody.
        bar_context: Per-bar BarContext from lead instruments.
        bar_feel: Per-bar BarFeel for timing humanization.

    Returns:
        List of NoteEvent objects (all on channel 9).
    """
    intensity = max(0.0, min(1.0, intensity))
    tpb = ticks_per_bar(beats_per_bar)
    total_bars = total_beats // beats_per_bar
    notes: List[NoteEvent] = []

    # Dynamic arc: ±15% velocity wave
    arc_period = random.choice([8, 12, 16])

    # Multi-bar kick phrasing (like swing but softer)
    kick_pattern: List[NoteEvent] = []
    kick_hold = 0

    for bar_idx in range(total_bars):
        global _current_feel
        _current_feel = bar_feel[bar_idx] if bar_feel and bar_idx < len(bar_feel) else None

        bar_start_tick = bar_idx * tpb

        # Per-bar reactive intensity
        local_intensity = (bar_intensities[bar_idx]
                           if bar_intensities and bar_idx < len(bar_intensities)
                           else intensity)

        # Dynamic arc
        arc_phase = (bar_idx % arc_period) / arc_period
        vel_mult = 1.0 + 0.15 * math.sin(arc_phase * 2 * math.pi)

        # Inter-instrument reactivity
        ctx = bar_context[bar_idx] if bar_context and bar_idx < len(bar_context) else None
        react_vel_mult = 1.0
        react_stir_mult = 1.0
        form_section = ""
        is_key_change = False

        if ctx is not None:
            if ctx.density > 0.7:
                react_vel_mult = 0.85   # Soften when lead is busy
                react_stir_mult = 0.8   # Lighter stir
            elif ctx.has_silence:
                react_stir_mult = 1.2   # More active stir when lead rests
            form_section = ctx.form_section
            is_key_change = getattr(ctx, 'is_key_change', False)

        is_bridge = form_section == "B"

        # Check for brush fill
        is_fill_bar = (
            fill_every > 0
            and bar_idx > 0
            and (bar_idx + 1) % fill_every == 0
            and bar_idx < total_bars - 1
        )

        if is_fill_bar:
            fill_func = random.choice(_BRUSH_FILL_FUNCTIONS)
            notes.extend(fill_func(bar_start_tick, swing, beats_per_bar=beats_per_bar))
        else:
            # Brush stir — intensity-responsive
            stir_vel_mult = vel_mult * react_vel_mult * react_stir_mult
            notes.extend(_generate_brush_stir(bar_start_tick, local_intensity,
                                               stir_vel_mult,
                                               beats_per_bar=beats_per_bar))

            # Brush accents (side-stick on 2 & 4)
            # Bridge: slightly more accents
            accent_intensity = min(1.0, local_intensity + 0.15) if is_bridge else local_intensity
            notes.extend(_generate_brush_accents(bar_start_tick, accent_intensity,
                                                  vel_mult * react_vel_mult,
                                                  beats_per_bar=beats_per_bar))

            # Feathered kick — musical patterns but very soft
            if kick_hold <= 0:
                kick_intensity = local_intensity * 0.6
                if is_bridge:
                    kick_intensity *= 0.7  # Even sparser on bridge
                # Choose pattern pool based on intensity and meter
                if beats_per_bar == 5:
                    bk_patterns = _BRUSH_KICK_PATTERNS_5
                elif beats_per_bar == 7:
                    bk_patterns = _BRUSH_KICK_PATTERNS_7
                else:
                    bk_patterns = _BRUSH_KICK_PATTERNS
                pattern_pool = (bk_patterns[:1] if kick_intensity < 0.3
                                else bk_patterns)
                silence_prob = max(0.15, 0.50 - local_intensity * 0.40)
                if random.random() < silence_prob:
                    kick_pattern = []
                else:
                    pattern = random.choice(pattern_pool)
                    kick_pattern = []
                    for beat_offset, is_accent in pattern:
                        tick = _swing_tick(bar_start_tick, beat_offset, swing)
                        if is_accent:
                            vel = _humanize_velocity(round(
                                random.randint(25, 45) * vel_mult))
                        else:
                            vel = _humanize_velocity(round(
                                random.randint(15, 30) * vel_mult))
                        kick_pattern.append(_drum_note(
                            KICK, _humanize_tick(tick, 10),
                            TICKS_PER_QUARTER, vel))
                kick_hold = random.randint(1, 3)
            else:
                # Replay with variation (feathered)
                if kick_pattern:
                    kick_pattern = _replay_kick_pattern(
                        kick_pattern, bar_start_tick, vel_mult * 0.6,
                        beats_per_bar=beats_per_bar)
            kick_hold -= 1
            notes.extend(kick_pattern or [])

            # Key change: side-stick accent + kick on beat 1
            if is_key_change:
                tick = _swing_tick(bar_start_tick, 0.0, swing)
                notes.append(_drum_note(
                    SIDE_STICK, _humanize_tick(tick, 5), TICKS_PER_8TH,
                    _humanize_velocity(round(random.randint(55, 70) * vel_mult))))
                if random.random() < 0.60:
                    notes.append(_drum_note(
                        KICK, _humanize_tick(tick, 8), TICKS_PER_QUARTER,
                        _humanize_velocity(round(random.randint(35, 50) * vel_mult))))

    return notes


def generate_modal_drums(total_beats: int, intensity: float = 0.5,
                         swing: bool = True, fill_every: int = 8,
                         beats_per_bar: int = 4) -> List[NoteEvent]:
    """Generate a modal/Elvin Jones-influenced drum pattern.

    Sparse ride (hit on ~70% of positions), floor tom coloring on beats 2&4
    instead of hi-hat, more open feel. Used for modal jazz.
    """
    intensity = max(0.0, min(1.0, intensity))
    tpb = ticks_per_bar(beats_per_bar)
    total_bars = total_beats // beats_per_bar
    notes: List[NoteEvent] = []
    need_crash = False

    # Build beat lists for the meter
    if beats_per_bar == 5:
        ride_accent_beats = [0.0, 2.0, 3.0]
        ride_swing_beats = [1.5, 3.5, 4.5]
        floor_tom_beats = [1.0, 3.0, 4.0]
        ghost_pool = [0.5, 1.5, 2.5, 3.5, 4.5]
    elif beats_per_bar == 7:
        ride_accent_beats = [0.0, 2.0, 4.0]
        ride_swing_beats = [1.5, 3.5, 5.5]
        floor_tom_beats = [1.0, 3.0, 5.0]
        ghost_pool = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    else:
        ride_accent_beats = [0.0, 2.0]
        ride_swing_beats = [1.5, 3.5]
        floor_tom_beats = [1.0, 3.0]
        ghost_pool = [0.5, 1.5, 2.5, 3.5]

    for bar_idx in range(total_bars):
        bar_start_tick = bar_idx * tpb

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
            notes.extend(fill_func(bar_start_tick, swing, beats_per_bar=beats_per_bar))
            need_crash = True
        else:
            # Sparse ride: only ~70% of standard positions
            for beat in ride_accent_beats:
                if random.random() < 0.70:
                    tick = _swing_tick(bar_start_tick, beat, swing)
                    vel = _humanize_velocity(random.randint(80, 95))
                    notes.append(_drum_note(RIDE, _humanize_tick(tick, 5), TICKS_PER_8TH, vel))

            for beat in ride_swing_beats:
                if random.random() < 0.70:
                    tick = _swing_tick(bar_start_tick, beat, swing)
                    vel = _humanize_velocity(random.randint(70, 80))
                    notes.append(_drum_note(RIDE, _humanize_tick(tick, 8), TICKS_PER_8TH, vel))

            # Floor tom on off-beats (instead of hi-hat)
            for beat in floor_tom_beats:
                if random.random() < 0.60:
                    tick = _swing_tick(bar_start_tick, beat, swing)
                    vel = _humanize_velocity(random.randint(45, 60))
                    notes.append(_drum_note(FLOOR_TOM, _humanize_tick(tick, 8), TICKS_PER_8TH, vel))

            # Occasional kick — sparser than swing
            for beat_idx in range(beats_per_bar):
                if random.random() < 0.25:
                    tick = _swing_tick(bar_start_tick, float(beat_idx), swing)
                    vel = _humanize_velocity(random.randint(30, 50))
                    notes.append(_drum_note(KICK, _humanize_tick(tick, 8), TICKS_PER_QUARTER, vel))

            # Occasional snare ghost note — very sparse
            if random.random() < 0.3 * intensity:
                beat = random.choice(ghost_pool)
                tick = _swing_tick(bar_start_tick, beat, swing)
                vel = _humanize_velocity(random.randint(35, 50))
                notes.append(_drum_note(SNARE, _humanize_tick(tick, 10), TICKS_PER_8TH, vel))

    return notes


# ---------------------------------------------------------------------------
# Drum solo generator
# ---------------------------------------------------------------------------


def generate_drum_solo(total_beats: int, intensity: float = 0.7,
                       swing: bool = True,
                       beats_per_bar: int = 4) -> List[NoteEvent]:
    """Generate an improvised jazz drum solo.

    Builds phrases from varied rhythmic ideas (ride work, snare
    conversations, tom melodies, fill patterns) with space between
    phrases and a dynamic arc.  Intensity controls how busy/loud.
    """
    intensity = max(0.0, min(1.0, intensity))
    tpb = ticks_per_bar(beats_per_bar)
    total_bars = total_beats // beats_per_bar
    notes: List[NoteEvent] = []

    # Build beat position pools based on meter
    num_16ths = beats_per_bar * 4
    num_triplets = beats_per_bar * 3
    ride_beats = [i * 0.5 for i in range(beats_per_bar * 2)]
    kick_accent_beats = [0.0] + [float(b) + 0.5 for b in range(beats_per_bar - 1) if b >= 2]
    sparse_possible = [i * 0.5 for i in range(beats_per_bar * 2)]

    bar_idx = 0
    while bar_idx < total_bars:
        # Dynamic arc within the solo: quieter start/end, louder middle
        progress = bar_idx / max(1, total_bars - 1)
        arc = 0.4 + 0.6 * math.sin(progress * math.pi)
        local_int = min(1.0, intensity * arc)

        # Choose phrase length (1-4 bars)
        phrase_bars = min(random.choice([1, 1, 2, 2, 3, 4]), total_bars - bar_idx)

        # Choose phrase type
        phrase_type = random.choices(
            ["ride_solo", "snare_work", "tom_melody", "fill_chain",
             "sparse_groove", "trade_lick"],
            weights=[0.20, 0.20, 0.15, 0.15, 0.15, 0.15],
            k=1,
        )[0]

        for pb in range(phrase_bars):
            bi = bar_idx + pb
            bar_tick = bi * tpb
            vel_scale = 0.7 + 0.3 * local_int

            if phrase_type == "ride_solo":
                # Ride cymbal conversation — varied rhythms on ride + kick accents
                for beat in ride_beats:
                    if random.random() < 0.65:
                        tick = _swing_tick(bar_tick, beat, swing)
                        vel = _humanize_velocity(int(random.randint(75, 100) * vel_scale))
                        notes.append(_drum_note(RIDE, _humanize_tick(tick, 5),
                                                TICKS_PER_8TH, vel))
                # Kick accents
                for beat in kick_accent_beats:
                    if random.random() < 0.35 * local_int:
                        tick = _swing_tick(bar_tick, beat, swing)
                        vel = _humanize_velocity(int(random.randint(50, 75) * vel_scale))
                        notes.append(_drum_note(KICK, _humanize_tick(tick, 8),
                                                TICKS_PER_QUARTER, vel))

            elif phrase_type == "snare_work":
                # Snare-centered: buzz rolls, accents, ghost notes
                for beat_16 in range(num_16ths):
                    beat_off = beat_16 * 0.25
                    tick = bar_tick + int(beat_off * TICKS_PER_QUARTER)
                    if beat_16 % 4 == 0:
                        # Accent on downbeats
                        if random.random() < 0.7:
                            vel = _humanize_velocity(int(random.randint(85, 110) * vel_scale))
                            notes.append(_drum_note(SNARE, _humanize_tick(tick, 3),
                                                    TICKS_PER_16TH, vel))
                    elif random.random() < 0.30 * local_int:
                        # Ghost notes
                        vel = _humanize_velocity(int(random.randint(25, 45) * vel_scale))
                        notes.append(_drum_note(SNARE, _humanize_tick(tick, 5),
                                                TICKS_PER_16TH, vel))
                # Ride on off-beats
                if beats_per_bar == 5:
                    ride_off = [1.0, 3.0, 4.0]
                elif beats_per_bar == 7:
                    ride_off = [1.0, 3.0, 5.0]
                else:
                    ride_off = [1.0, 3.0]
                for beat in ride_off:
                    tick = _swing_tick(bar_tick, beat, swing)
                    vel = _humanize_velocity(int(random.randint(60, 80) * vel_scale))
                    notes.append(_drum_note(RIDE, _humanize_tick(tick, 5),
                                            TICKS_PER_8TH, vel))

            elif phrase_type == "tom_melody":
                # Melodic tom patterns — moving around the kit
                toms = [HI_TOM, MID_TOM, LO_TOM, FLOOR_TOM]
                triplet_dur = TICKS_PER_QUARTER // 3
                num_hits = random.randint(6, min(num_triplets, 12))
                positions = sorted(random.sample(
                    [i * (1.0/3.0) for i in range(num_triplets)],
                    k=min(num_hits, num_triplets)))
                for pos in positions:
                    tick = bar_tick + int(pos * TICKS_PER_QUARTER)
                    tom = random.choice(toms)
                    vel = _humanize_velocity(int(random.randint(60, 100) * vel_scale))
                    notes.append(_drum_note(tom, _humanize_tick(tick, 5),
                                            triplet_dur, vel))

            elif phrase_type == "fill_chain":
                # Use existing fill functions
                fill_func = random.choice(_FILL_FUNCTIONS)
                notes.extend(fill_func(bar_tick, swing, beats_per_bar=beats_per_bar))

            elif phrase_type == "sparse_groove":
                # Sparse accents with lots of space — Max Roach style
                num_hits = random.randint(2, 5)
                chosen = sorted(random.sample(sparse_possible,
                                              k=min(num_hits, len(sparse_possible))))
                drums_pool = [SNARE, FLOOR_TOM, HI_TOM, RIDE, KICK]
                for beat in chosen:
                    tick = _swing_tick(bar_tick, beat, swing)
                    drum = random.choice(drums_pool)
                    vel = _humanize_velocity(int(random.randint(75, 110) * vel_scale))
                    notes.append(_drum_note(drum, _humanize_tick(tick, 5),
                                            TICKS_PER_8TH, vel))

            else:  # trade_lick
                # Short rhythmic motif: 2-3 hits then space
                motif_start = random.choice([0.0, 0.5, 1.0])
                for j in range(random.randint(2, 4)):
                    beat = motif_start + j * 0.5
                    if beat >= float(beats_per_bar):
                        break
                    tick = _swing_tick(bar_tick, beat, swing)
                    drum = random.choice([SNARE, SNARE, HI_TOM, FLOOR_TOM])
                    vel = _humanize_velocity(int(random.randint(70, 105) * vel_scale))
                    notes.append(_drum_note(drum, _humanize_tick(tick, 5),
                                            TICKS_PER_8TH, vel))

        bar_idx += phrase_bars

        # Breathing space between phrases (0-2 bars of silence)
        if bar_idx < total_bars:
            silence = random.choice([0, 0, 1, 1, 2])
            bar_idx = min(bar_idx + silence, total_bars)

    return notes


# ---------------------------------------------------------------------------
# Expression: CC4 (foot controller) for hi-hat open/close articulation
# ---------------------------------------------------------------------------


def generate_hihat_expression(
    drum_notes: List[NoteEvent],
    channel: int = DRUM_CHANNEL,
) -> List[CCEvent]:
    """Generate CC4 foot controller events to articulate hi-hat open/closed.

    CC4 = 0-20 before closed hi-hat hits, CC4 = 90-127 before open hi-hat hits.
    This controls hi-hat tightness on synths that respond to CC4.

    Returns:
        List[CCEvent]
    """
    ccs: List[CCEvent] = []
    for note in drum_notes:
        if note.pitch == CLOSED_HH:
            val = random.randint(0, 20)
            # Place CC slightly before the hit
            cc_tick = max(0, note.start_tick - 5)
            ccs.append(CCEvent(cc_number=4, value=val, start_tick=cc_tick, channel=channel))
        elif note.pitch == OPEN_HH:
            val = random.randint(90, 127)
            cc_tick = max(0, note.start_tick - 5)
            ccs.append(CCEvent(cc_number=4, value=val, start_tick=cc_tick, channel=channel))
    return ccs