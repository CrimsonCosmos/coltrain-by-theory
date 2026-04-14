"""Piano comping generator for Coltrain.

Generates idiomatic jazz piano voicings with rhythmic comping patterns,
voice leading, and intensity-dependent pattern selection.
"""

import math
import random
from typing import List, Optional, Tuple

from . import NoteEvent, CCEvent, BarFeel, TICKS_PER_QUARTER, TICKS_PER_BAR, TICKS_PER_8TH, TICKS_PER_16TH, ticks_per_bar

# Module-level rhythmic feel state (set per bar in generator loop)
_current_feel: Optional[BarFeel] = None

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

# Per-voicing-type ceiling: sparse types ok higher, dense types capped at A4
VOICING_CEILING = {
    "shell": 72, "guide_tone": 72, "stride": 72,
    "rootless_a": 69, "rootless_b": 69, "drop2": 69,
    "quartal": 69, "open_evans": 69, "so_what": 69,
}

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
    # Pattern 4 - Sparse offbeat (single hit on the "and" of 2)
    [(1.5, 1.5)],
    # Pattern 5 - Dense
    [(0.0, 0.5), (1.0, 0.5), (2.0, 0.5), (3.0, 0.5)],
    # Pattern 6 - Freddie Green (4-on-the-floor quarters)
    [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0)],
    # Pattern 7 - Anticipation push (hits before barline area)
    [(3.5, 1.5), (1.5, 0.5)],
    # Pattern 8 - Dotted rhythm
    [(0.0, 1.5), (1.5, 0.5), (3.0, 1.0)],
    # Pattern 9 - Upbeat pair (and-of-1, and-of-3)
    [(0.5, 1.0), (2.5, 1.0)],
    # Pattern 10 - Stride (bass on 1&3, chord stab on 2&4)
    [(0.0, 0.75), (1.0, 0.75), (2.0, 0.75), (3.0, 0.75)],
    # Pattern 11 - Backbeat push (and-of-2 into 3, and-of-4)
    [(1.5, 1.0), (3.5, 0.5)],
    # Pattern 12 - Bud Powell (hit on 1, anticipate 3)
    [(0.0, 1.0), (2.5, 0.5)],
    # Pattern 13 - Offbeat stabs (and-of-1, and-of-2, and-of-3)
    [(0.5, 0.5), (1.5, 0.5), (2.5, 0.5)],
    # Pattern 14 - Single anticipation (and-of-4 only — sparse, forward-leaning)
    [(3.5, 1.0)],
]

# Which patterns to use at each intensity range
PATTERNS_BY_INTENSITY = {
    "low": [4, 9, 14, 0, 12],          # mostly offbeat sparse, some 2&4
    "medium": [1, 2, 7, 9, 11, 12, 0], # syncopated mix
    "high": [3, 7, 11, 13, 1, 8],      # dense syncopation
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


def _humanize_tick(tick: int, amount: int = 10, beat_offset: float = -1.0) -> int:
    """Add timing variation with Tatum-style downbeat/upbeat bias.

    Downbeats pulled ahead (crisp), upbeats pushed behind (lazy).
    """
    spread = _current_feel.timing_spread if _current_feel else 1.0
    scaled = max(1, int(amount * spread))
    base_jitter = random.randint(-scaled, scaled)
    if beat_offset >= 0.0:
        frac = beat_offset % 1.0
        if frac < 0.15 or frac > 0.85:
            bias = -random.randint(3, 8)   # downbeat: pull ahead
        elif 0.35 < frac < 0.65:
            bias = random.randint(5, 15)   # upbeat: push behind
        else:
            bias = 0
        return max(0, tick + base_jitter + bias)
    return max(0, tick + base_jitter)


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
                    center_midi: int = 60,
                    include_extensions: bool = True) -> List[int]:
    """Build a chord voicing as a list of MIDI pitches.

    Args:
        root_pc: Root pitch class (0-11).
        quality: Chord quality string.
        voicing_type: One of 'shell', 'rootless_a', 'rootless_b', 'drop2'.
        center_midi: Target center of the voicing for octave placement.
        include_extensions: Whether to include 9th/extensions in rootless voicings.
            When False, rootless voicings substitute the 5th for the 9th,
            producing a cleaner, more consonant sound.

    Returns:
        Sorted list of MIDI pitches within COMP_LOW-COMP_HIGH.
    """
    third = _get_3rd(quality)
    fifth = _get_5th(quality)
    seventh = _get_7th(quality)
    ninth = _get_9th(quality)

    # When extensions are excluded, substitute 5th for 9th in rootless voicings
    ninth_or_sub = ninth if include_extensions else fifth

    if voicing_type == "shell":
        # Root + 3rd + 7th (3 notes)
        pcs = [0, third, seventh]
    elif voicing_type == "rootless_a":
        # 3rd + 5th + 7th + 9th (4 notes, no root)
        pcs = [third, fifth, seventh, ninth_or_sub]
    elif voicing_type == "rootless_b":
        # 7th + 9th + 3rd + 5th (4 notes, no root, different ordering/inversion)
        pcs = [seventh, ninth_or_sub, third, fifth]
    elif voicing_type == "drop2":
        # Close position (root, 3rd, 5th, 7th) but 2nd-from-top dropped an octave
        # Close position bottom-up: root, 3rd, 5th, 7th
        # 2nd from top is 5th, drop it an octave
        pcs = [0, third, fifth, seventh]
    elif voicing_type == "quartal":
        # McCoy Tyner-style stacked perfect 4ths from the chord's 3rd
        # Creates open, modern voicings characteristic of Coltrane quartet
        pcs = [third, third + 5, third + 10, third + 15]
    elif voicing_type == "open_evans":
        # Bill Evans open voicing: root + 7th spread wide, 3rd + 5th up an octave
        # Spacious, open sound characteristic of Evans trio recordings
        pcs = [0, seventh, third + 12, fifth + 12]
    elif voicing_type == "so_what":
        # Bill Evans "So What" voicing: stacked perfect 4ths from the 3rd
        # 3rd, 3rd+5, 3rd+10, 3rd+14 (three 4ths then a major 3rd on top)
        pcs = [third, third + 5, third + 10, third + 14]
    elif voicing_type == "guide_tone":
        # Ultra-sparse: just 3rd + 7th (2 notes only)
        pcs = [third, seventh]
    elif voicing_type == "stride":
        # Monk stride: sparse shell voicing (root + 3rd + 7th)
        # Bass note handled separately by stride pattern logic
        pcs = [0, third, seventh]
    else:
        pcs = [0, third, seventh]

    # Per-voicing ceiling to keep dense voicings out of melody register
    ceiling = VOICING_CEILING.get(voicing_type, COMP_HIGH)

    # Convert pitch classes to MIDI notes near center_midi
    midi_notes = []
    for pc_offset in pcs:
        target_pc = (root_pc + pc_offset) % 12
        # Find the nearest MIDI note with this pitch class to center_midi
        candidates = []
        for midi in range(COMP_LOW, ceiling + 1):
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
    midi_notes = [n for n in midi_notes if COMP_LOW <= n <= ceiling]

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
                     voicing_type: str,
                     include_extensions: bool = True) -> List[int]:
    """Build a voicing that voice-leads smoothly from prev_voicing.

    Tries multiple center points and picks the one with lowest voice-leading cost.

    Args:
        root_pc: Root pitch class.
        quality: Chord quality.
        prev_voicing: Previous voicing's MIDI pitches.
        voicing_type: Voicing type string.
        include_extensions: Pass through to _build_voicing.

    Returns:
        Best-voiced list of MIDI pitches.
    """
    if not prev_voicing:
        v = _build_voicing(root_pc, quality, voicing_type, center_midi=60,
                           include_extensions=include_extensions)
        return _enforce_low_interval_limits(v)

    prev_center = sum(prev_voicing) // len(prev_voicing)

    # Try building voicings centered at different positions
    best_voicing = None
    best_cost = float("inf")

    for center_offset in range(-6, 7, 2):
        candidate_center = prev_center + center_offset
        candidate = _build_voicing(root_pc, quality, voicing_type,
                                   candidate_center,
                                   include_extensions=include_extensions)
        if not candidate:
            continue
        cost = _voice_leading_cost(prev_voicing, candidate)
        if cost < best_cost:
            best_cost = cost
            best_voicing = candidate

    if best_voicing is None:
        v = _build_voicing(root_pc, quality, voicing_type, center_midi=60,
                           include_extensions=include_extensions)
        return _enforce_low_interval_limits(v)

    return _enforce_low_interval_limits(best_voicing)


def _enforce_low_interval_limits(voicing: List[int]) -> List[int]:
    """Enforce minimum intervals between adjacent notes in low registers.

    Below C3 (MIDI 48): adjacent notes must be >= 7 semitones apart (P5).
    Below G3 (MIDI 55): adjacent notes must be >= 5 semitones apart (P4).
    Violations fixed by shifting upper note up 12, or dropping it if > COMP_HIGH.
    """
    if len(voicing) < 2:
        return voicing
    result = sorted(voicing)
    i = 0
    while i < len(result) - 1:
        lower = result[i]
        upper = result[i + 1]
        gap = upper - lower
        min_gap = 0
        if lower < 48:
            min_gap = 7  # P5
        elif lower < 55:
            min_gap = 5  # P4
        if min_gap > 0 and gap < min_gap:
            shifted = upper + 12
            if shifted <= COMP_HIGH:
                result[i + 1] = shifted
                result.sort()
                # Re-check from beginning after rearrangement
                i = 0
                continue
            else:
                result.pop(i + 1)
                continue
        i += 1
    return result


# ---------------------------------------------------------------------------
# Chord lookup by beat
# ---------------------------------------------------------------------------


def _chord_at_beat(chords, beat: float):
    """Return the chord active at the given beat."""
    for chord in chords:
        if chord.start_beat <= beat < chord.end_beat:
            return chord
    return chords[-1] if chords else None


def _stride_bass_note(root_pc: int, beat_in_bar: float) -> int:
    """Return a low bass note for stride left hand.

    Root on beats 1, 5th on beat 3.
    """
    if abs(beat_in_bar - 2.0) < 0.1:  # beat 3
        target_pc = (root_pc + 7) % 12  # perfect 5th
    else:
        target_pc = root_pc
    # Low register: 36-48 (C2-C3)
    for midi in range(36, 49):
        if midi % 12 == target_pc:
            return midi
    return 36


# ---------------------------------------------------------------------------
# Context-aware voicing selection
# ---------------------------------------------------------------------------


def _select_voicing_type(intensity: float, coltrane: bool, context: str) -> str:
    """Select a voicing type based on intensity, style, and context.

    Head context prefers sparse voicings (shell, guide_tone, open_evans).
    Solo context uses the full palette.
    """
    if context == "head" and intensity < 0.7:
        # Head: sparse voicings to stay out of melody's way
        pool = ["shell", "guide_tone", "open_evans"]
        wts = [0.35, 0.35, 0.30]
        return random.choices(pool, weights=wts, k=1)[0]

    if intensity < 0.3:
        pool = ["shell", "guide_tone", "open_evans"]
        wts = [0.45, 0.25, 0.30]
        return random.choices(pool, weights=wts, k=1)[0]
    elif intensity < 0.7:
        pool = ["shell", "guide_tone", "rootless_a", "rootless_b", "open_evans", "so_what", "stride"]
        wts = [0.10, 0.12, 0.20, 0.18, 0.18, 0.12, 0.10]
        if coltrane and intensity > 0.5:
            pool.append("quartal")
            wts.append(0.15)
            total = sum(wts)
            wts = [w / total for w in wts]
        return random.choices(pool, weights=wts, k=1)[0]
    else:
        pool = ["rootless_a", "rootless_b", "drop2", "open_evans", "so_what", "guide_tone", "stride"]
        wts = [0.20, 0.15, 0.18, 0.18, 0.14, 0.05, 0.10]
        if coltrane:
            pool.append("quartal")
            wts.append(0.15)
            total = sum(wts)
            wts = [w / total for w in wts]
        return random.choices(pool, weights=wts, k=1)[0]


# ---------------------------------------------------------------------------
# Context-aware velocity helper
# ---------------------------------------------------------------------------


def _comping_velocity(intensity: float, context: str) -> Tuple[int, int]:
    """Return (vel_lo, vel_hi) for comping based on intensity and context.

    Left hand is always quieter than right hand (melody) to create
    natural foreground/background separation.
    """
    if context == "head":
        if intensity < 0.3:
            return 32, 46
        elif intensity < 0.6:
            return 36, 50
        else:
            return 40, 54
    else:  # solo
        if intensity < 0.3:
            return 36, 50
        elif intensity < 0.6:
            return 42, 56
        else:
            return 46, 62


# ---------------------------------------------------------------------------
# Descending chord cascade (Tatum-style parallel motion)
# ---------------------------------------------------------------------------


def _cascade_voicings(start_voicing: List[int],
                      steps: int, step_beats: float = 0.5) -> List[Tuple[List[int], float]]:
    """Generate a descending chord cascade: each voicing drops ~2-3 semitones.

    Returns list of (voicing, beat_offset) pairs.
    """
    result: List[Tuple[List[int], float]] = []
    current = list(start_voicing)
    for i in range(steps):
        result.append((list(current), i * step_beats))
        # Drop each note 2-3 semitones, keeping voices distinct
        current = [max(COMP_LOW, n - random.choice([2, 3])) for n in current]
        current = sorted(set(current))
    return result


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate_comping(chords, total_beats: int, intensity: float = 0.5,
                      swing: bool = True, coltrane: bool = False,
                      bass_sync_ticks: Optional[List[int]] = None,
                      sync_probability: float = 0.25,
                      context: str = "solo",
                      bar_intensities: Optional[List[float]] = None,
                      bar_feel: Optional[list] = None,
                      beats_per_bar: int = 4) -> List[NoteEvent]:
    """Generate piano comping voicings over a chord progression.

    Args:
        chords: List of ChordEvent objects.
        total_beats: Total number of beats to generate.
        intensity: Comping intensity 0.0-1.0. Affects pattern density,
                   voicing type, and velocity.
        swing: Whether to apply swing feel.
        context: "head" or "solo" — affects density, velocity, and voicing choices.

    Returns:
        List of NoteEvent objects (multiple notes per chord hit).
    """
    if not chords:
        return []

    intensity = max(0.0, min(1.0, intensity))
    tpb = ticks_per_bar(beats_per_bar)
    total_bars = total_beats // beats_per_bar
    notes: List[NoteEvent] = []

    # Unified voicing palette: blends Evans open voicings, Monk stride,
    # and standard rootless/shell voicings at all intensity levels
    voicing_type = _select_voicing_type(intensity, coltrane, context)

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
    silence_streak = 0     # Remaining bars of forced silence (clustered rest)
    phrase_length = random.choice([4, 6, 8])  # Bars per dynamic phrase

    for bar_idx in range(total_bars):
        global _current_feel
        _current_feel = bar_feel[bar_idx] if bar_feel and bar_idx < len(bar_feel) else None

        bar_start_beat = bar_idx * float(beats_per_bar)
        bar_start_tick = bar_idx * tpb

        # Per-bar reactive intensity
        local_intensity = (bar_intensities[bar_idx]
                           if bar_intensities and bar_idx < len(bar_intensities)
                           else intensity)

        # Clustered silence: continue multi-bar rest streaks
        if silence_streak > 0:
            silence_streak -= 1
            continue

        # Lay-out (rest) bars: skip this bar entirely for breathing room
        if context == "head":
            # Head: left hand can rest more since melody carries harmony
            if local_intensity < 0.4:
                layoff_prob = 0.45
            elif local_intensity < 0.7:
                layoff_prob = 0.30
            else:
                layoff_prob = 0.20
        else:
            # Solo: left hand should be more present for harmonic grounding
            if local_intensity < 0.4:
                layoff_prob = 0.35
            elif local_intensity < 0.7:
                layoff_prob = 0.20
            else:
                layoff_prob = 0.12
        if random.random() < layoff_prob:
            # 40% chance to extend rest 1-2 more bars
            if random.random() < 0.40:
                silence_streak = random.randint(1, 2)
            continue

        # Bass-piano sync: place a chord voicing at a bass downbeat tick
        if bass_sync_ticks and random.random() < sync_probability:
            bar_end_tick = bar_start_tick + tpb
            bar_sync_ticks = [t for t in bass_sync_ticks
                              if bar_start_tick <= t < bar_end_tick]
            if bar_sync_ticks:
                sync_tick = random.choice(bar_sync_ticks)
                sync_beat = bar_start_beat + (sync_tick - bar_start_tick) / TICKS_PER_QUARTER
                chord = _chord_at_beat(chords, sync_beat)
                if chord is not None:
                    use_ext = random.random() < (0.75 if chord.quality in ("dom7", "7", "aug7") else 0.40)
                    voicing = _smooth_voicing(
                        chord.root_pc, chord.quality, prev_voicing, voicing_type,
                        include_extensions=use_ext,
                    )
                    if voicing:
                        prev_voicing = voicing
                        # Staccato sync stabs
                        dur_ticks = max(TICKS_PER_16TH,
                                        int(TICKS_PER_QUARTER * random.uniform(0.35, 0.60)))
                        vel_lo, vel_hi = _comping_velocity(local_intensity, context)
                        base_vel = random.randint(vel_lo, vel_hi)
                        spread_amount = random.randint(0, 18) if len(voicing) > 1 else 0
                        for note_idx, midi_pitch in enumerate(sorted(voicing)):
                            note_tick = sync_tick + (note_idx * spread_amount // max(1, len(voicing) - 1))
                            vel = _humanize_velocity(base_vel, amount=4)
                            notes.append(NoteEvent(
                                pitch=midi_pitch,
                                start_tick=note_tick,
                                duration_ticks=dur_ticks,
                                velocity=vel,
                                channel=1,
                            ))
                        continue  # Skip normal pattern for this bar

        # Re-evaluate voicing type every 8 bars for gradual evolution
        if bar_idx > 0 and bar_idx % 8 == 0:
            voicing_type = _select_voicing_type(local_intensity, coltrane, context)

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

        # Descending chord cascade: triggered at phrase boundaries at high intensity
        if (local_intensity > 0.65
                and bar_idx % phrase_length == phrase_length - 1
                and len(prev_voicing) >= 3
                and random.random() < 0.20):
            cascade = _cascade_voicings(prev_voicing, steps=random.randint(4, 8))
            vel_lo, vel_hi = _comping_velocity(local_intensity, context)
            for casc_voicing, casc_offset in cascade:
                casc_tick = bar_start_tick + int(casc_offset * TICKS_PER_QUARTER)
                casc_tick = _humanize_tick(casc_tick, amount=12)
                casc_vel = max(1, min(127, round(
                    random.randint(vel_lo, vel_hi) * vel_phrase_mult * 0.9)))
                spread_amount = random.randint(5, 15)
                for note_idx, midi_pitch in enumerate(sorted(casc_voicing)):
                    note_tick = casc_tick + (note_idx * spread_amount // max(1, len(casc_voicing) - 1))
                    notes.append(NoteEvent(
                        pitch=midi_pitch,
                        start_tick=max(0, note_tick),
                        duration_ticks=int(0.4 * TICKS_PER_QUARTER),
                        velocity=_humanize_velocity(casc_vel - note_idx * 2, amount=3),
                        channel=1,
                    ))
                prev_voicing = casc_voicing
            continue  # Skip normal pattern for this bar

        for beat_offset, duration_beats in pattern:
            abs_beat = bar_start_beat + beat_offset

            chord = _chord_at_beat(chords, abs_beat)
            if chord is None:
                continue

            # Vary extension inclusion: dominant-quality chords get extensions
            # more often, creating consonance/dissonance movement
            if chord.quality in ("dom7", "7", "aug7"):
                ext_prob = 0.75  # dominant chords: usually tense
            elif chord.quality in ("min7", "min7b5"):
                ext_prob = 0.50  # minor chords: moderate
            else:
                ext_prob = 0.30  # major/stable chords: usually clean
            use_ext = random.random() < ext_prob

            # Build voicing with voice leading from previous
            voicing = _smooth_voicing(
                chord.root_pc, chord.quality, prev_voicing, voicing_type,
                include_extensions=use_ext,
            )
            if not voicing:
                continue

            # Octave doubling: reinforce top note an octave lower at high intensity
            # (Tatum-style fullness — 38% of chords in reference)
            if (local_intensity > 0.55
                    and len(voicing) >= 3
                    and random.random() < 0.35 * local_intensity):
                top_note = max(voicing)
                doubled = top_note - 12
                if doubled >= COMP_LOW and doubled not in voicing:
                    voicing = sorted(voicing + [doubled])

            prev_voicing = voicing

            # Calculate tick position with swing
            tick = _swing_tick(bar_start_tick, beat_offset, swing)
            tick = _humanize_tick(tick, amount=20 if swing else 8, beat_offset=beat_offset)

            # Duration: staccato/legato variation based on beat position
            dur_ticks = int(duration_beats * TICKS_PER_QUARTER)
            beat_in_bar = beat_offset
            if abs(beat_in_bar - 1.0) < 0.1 or abs(beat_in_bar - 3.0) < 0.1:
                # Beats 2&4: staccato stabs (cut 40-60%)
                dur_ticks = int(dur_ticks * random.uniform(0.35, 0.55))
            elif abs(beat_in_bar) < 0.1 or abs(beat_in_bar - 2.0) < 0.1:
                # Beats 1&3: more legato (cut 10-25%)
                dur_ticks = int(dur_ticks * random.uniform(0.70, 0.85))
            else:
                # Syncopated: random mix (cut 25-50%)
                dur_ticks = int(dur_ticks * random.uniform(0.45, 0.70))
            dur_ticks = max(TICKS_PER_16TH, dur_ticks)

            # Stride mode: beats 1&3 = single bass note, beats 2&4 = chord stab
            if voicing_type == "stride":
                beat_in_bar = beat_offset
                is_bass_beat = (abs(beat_in_bar) < 0.1 or abs(beat_in_bar - 2.0) < 0.1)
                if is_bass_beat:
                    # Single bass note in low register
                    bass_pitch = _stride_bass_note(chord.root_pc, beat_in_bar)
                    base_vel = max(1, min(127, round(
                        random.randint(70, 90) * vel_phrase_mult)))
                    notes.append(NoteEvent(
                        pitch=bass_pitch,
                        start_tick=tick,
                        duration_ticks=dur_ticks,
                        velocity=_humanize_velocity(base_vel, amount=4),
                        channel=1,
                    ))
                    continue  # skip voicing emission
                # Beats 2&4: percussive chord stab (fall through to normal voicing emission)

            # Velocity: intensity-scaled ranges with phrase arc
            vel_lo, vel_hi = _comping_velocity(local_intensity, context)

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
                    channel=1,
                ))

    return notes


# ---------------------------------------------------------------------------
# Head melody harmonization (block chords)
# ---------------------------------------------------------------------------


def harmonize_head_melody(melody_notes: List[NoteEvent],
                          chords: list,
                          total_beats: int,
                          swing: bool = True,
                          beats_per_bar: int = 4) -> List[NoteEvent]:
    """Add block chord harmonization below head melody notes.

    On strong beats with sustained notes, adds 2 chord tones below the
    melody note to create block chord voicings with melody on top.
    Makes the right hand sound like a pianist playing the tune, not a horn.

    Args:
        melody_notes: Head melody NoteEvent list.
        chords: ChordEvent list for the section.
        total_beats: Section length in beats.
        swing: Whether swing feel is active.

    Returns:
        Extended list including original melody notes plus harmony notes.
    """
    if not melody_notes or not chords:
        return melody_notes

    result = list(melody_notes)

    for note in melody_notes:
        tpb_local = ticks_per_bar(beats_per_bar)
        beat_in_bar = (note.start_tick % tpb_local) / TICKS_PER_QUARTER

        # Harmonize on strong beats (1 and 3) with longer notes
        is_strong_beat = (beat_in_bar < 0.3 or abs(beat_in_bar - 2.0) < 0.3)
        is_long_enough = note.duration_ticks >= TICKS_PER_QUARTER

        if not (is_strong_beat and is_long_enough):
            # 15% chance to also harmonize off-beat notes
            if random.random() > 0.15:
                continue

        # Find active chord
        note_beat = note.start_tick / TICKS_PER_QUARTER
        chord = _chord_at_beat(chords, note_beat)
        if chord is None:
            continue

        # Get chord tone intervals
        intervals = _get_intervals(chord.quality)
        melody_pc = note.pitch % 12
        root_pc = chord.root_pc

        # Build voicing: chord tones below melody, within right-hand range
        harmony_floor = max(55, note.pitch - 12)

        harmony_pitches = []
        for interval in intervals:
            tone_pc = (root_pc + interval) % 12
            if tone_pc == melody_pc:
                continue
            # Find nearest instance below melody note
            candidate = note.pitch - ((melody_pc - tone_pc) % 12)
            if candidate == note.pitch:
                candidate -= 12
            if candidate < harmony_floor:
                continue
            harmony_pitches.append(candidate)

        # Take top 2 (closest to melody) for clean block voicing
        harmony_pitches.sort(reverse=True)
        harmony_pitches = harmony_pitches[:2]

        if not harmony_pitches:
            continue

        # Emit harmony notes with slightly lower velocity
        vel_reduction = random.randint(8, 15)
        for hp in harmony_pitches:
            result.append(NoteEvent(
                pitch=hp,
                start_tick=note.start_tick + random.randint(0, 8),
                duration_ticks=note.duration_ticks,
                velocity=max(1, note.velocity - vel_reduction),
                channel=0,
            ))

    result.sort(key=lambda n: n.start_tick)
    return result


# ---------------------------------------------------------------------------
# Sustain pedal generator
# ---------------------------------------------------------------------------


def generate_sustain_pedal(piano_notes: List[NoteEvent],
                            channel: int = 0,
                            beats_per_bar: int = 4) -> List[CCEvent]:
    """Generate sustain pedal (CC64) events from piano note timings.

    Groups piano notes by approximate start tick (within 20-tick window = same voicing).
    For each voicing: pedal down 10 ticks before, pedal up 15 ticks before next voicing.
    Last voicing holds pedal for 1 bar.

    Args:
        piano_notes: Piano NoteEvent list (sorted by start_tick).
        channel: MIDI channel for CC events.

    Returns:
        List of CCEvent objects for sustain pedal on/off.
    """
    if not piano_notes:
        return []

    # Group notes into "voicings" by proximity of start_tick
    voicing_ticks: List[int] = []
    window = 20  # ticks
    for note in sorted(piano_notes, key=lambda n: n.start_tick):
        if not voicing_ticks or note.start_tick - voicing_ticks[-1] > window:
            voicing_ticks.append(note.start_tick)

    if not voicing_ticks:
        return []

    cc_events: List[CCEvent] = []
    for i, tick in enumerate(voicing_ticks):
        # Pedal down: 10 ticks before the voicing
        pedal_down_tick = max(0, tick - 10)

        # Pedal up: 15 ticks before the next voicing, or hold for 1 bar if last
        if i + 1 < len(voicing_ticks):
            pedal_up_tick = max(pedal_down_tick + 1, voicing_ticks[i + 1] - 15)
        else:
            pedal_up_tick = tick + ticks_per_bar(beats_per_bar)

        # Pedal up (release previous) just before pedal down
        if i > 0:
            # Already handled by previous iteration's pedal_up
            pass

        cc_events.append(CCEvent(
            cc_number=64, value=127, start_tick=pedal_down_tick, channel=channel,
        ))
        cc_events.append(CCEvent(
            cc_number=64, value=0, start_tick=pedal_up_tick, channel=channel,
        ))

    cc_events.sort(key=lambda e: e.start_tick)
    return cc_events
