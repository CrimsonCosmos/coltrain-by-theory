"""Coltrane reharmonization engine for Coltrain.

Expands V->I (and ii->V->I) cadences through multi-tonic substitutions
at three density levels, inspired by John Coltrane's harmonic innovations.

Density levels:
  - "off"    : No changes
  - "light"  : Tritone substitution (bII7 -> I)
  - "medium" : Short Coltrane cycle (bVII7 -> bIII7 -> bVI7 -> I)
  - "heavy"  : Full 6-chord descending major-third cycle
"""

from typing import List

from coltrain.theory.chord import ChordEvent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dominant qualities that can function as V chords
_DOMINANT_QUALITIES = {"7", "dom7"}

# Tonic qualities that can function as I chords in a V->I resolution
_TONIC_QUALITIES = {"maj7", "maj", "6"}

# Minimum duration (in beats) for a chord to be eligible for reharmonization.
# Chords shorter than this are too brief to subdivide meaningfully.
_MIN_DURATION_FOR_REHARM = 1.0

# Valid density settings
_VALID_DENSITIES = {"off", "light", "medium", "heavy"}


# ---------------------------------------------------------------------------
# Helper: detect V->I resolution
# ---------------------------------------------------------------------------


def _is_v_to_i(v_chord: ChordEvent, i_chord: ChordEvent) -> bool:
    """Return True if v_chord -> i_chord forms a V7 -> Imaj resolution.

    Conditions:
      1. v_chord has a dominant quality ("7" or "dom7").
      2. i_chord has a tonic quality ("maj7", "maj", or "6").
      3. The root of v_chord is a perfect 5th above i_chord's root
         (i.e., v_root == (i_root + 7) % 12).
    """
    if v_chord.quality not in _DOMINANT_QUALITIES:
        return False
    if i_chord.quality not in _TONIC_QUALITIES:
        return False
    if (v_chord.root_pc - i_chord.root_pc) % 12 != 7:
        return False
    return True


# ---------------------------------------------------------------------------
# Helper: assign key_center_pc for substitution chords
# ---------------------------------------------------------------------------


def _key_center_for_sub(root_pc: int, quality: str) -> int:
    """Determine the key_center_pc for a substitution chord.

    - Dominant chords function as V of something: key = (root - 7) % 12.
    - Major chords are temporary tonics: key = root.
    """
    if quality in _DOMINANT_QUALITIES:
        return (root_pc - 7) % 12
    return root_pc


# ---------------------------------------------------------------------------
# Substitution builders
# ---------------------------------------------------------------------------


def _tritone_sub(
    v_chord: ChordEvent, i_chord: ChordEvent
) -> List[ChordEvent]:
    """Light density: tritone substitution.

    Replaces V7 with bII7 -> I.
    The V chord's duration is split in half: first half for bII7, second half
    returns to I (prepended to the original I chord's remaining duration).

    Returns two chords: [bII7, I_adjusted].
    """
    v_dur = v_chord.duration_beats
    half = v_dur / 2.0

    # bII7: root is a tritone above V's root (= V_root + 6)
    sub_root = (v_chord.root_pc + 6) % 12
    sub = ChordEvent(
        root_pc=sub_root,
        quality="7",
        start_beat=v_chord.start_beat,
        duration_beats=half,
        key_center_pc=_key_center_for_sub(sub_root, "7"),
        function="bII",
        form_section=v_chord.form_section,
    )

    # I chord starts half a V-duration earlier, gaining that time
    i_adjusted = ChordEvent(
        root_pc=i_chord.root_pc,
        quality=i_chord.quality,
        start_beat=v_chord.start_beat + half,
        duration_beats=half + i_chord.duration_beats,
        key_center_pc=i_chord.root_pc,
        function=i_chord.function or "I",
        form_section=i_chord.form_section,
    )

    return [sub, i_adjusted]


def _short_coltrane_cycle(
    v_chord: ChordEvent, i_chord: ChordEvent
) -> List[ChordEvent]:
    """Medium density: short Coltrane cycle.

    Replaces V7 with three dominant chords descending by major thirds,
    resolving to I. The V chord's duration is split evenly among the
    three substitution chords; I keeps its original duration.

    Cycle (intervals from I root): bVII7(+10), bIII7(+3), bVI7(+8) -> I.
    """
    i_root = i_chord.root_pc
    v_dur = v_chord.duration_beats
    sub_dur = v_dur / 3.0

    intervals = [10, 3, 8]
    functions = ["bVII", "bIII", "bVI"]
    result: List[ChordEvent] = []

    for idx, (interval, fn) in enumerate(zip(intervals, functions)):
        root = (i_root + interval) % 12
        result.append(ChordEvent(
            root_pc=root,
            quality="7",
            start_beat=v_chord.start_beat + idx * sub_dur,
            duration_beats=sub_dur,
            key_center_pc=_key_center_for_sub(root, "7"),
            function=fn,
            form_section=v_chord.form_section,
        ))

    # I chord keeps its original start and duration
    result.append(ChordEvent(
        root_pc=i_chord.root_pc,
        quality=i_chord.quality,
        start_beat=i_chord.start_beat,
        duration_beats=i_chord.duration_beats,
        key_center_pc=i_chord.root_pc,
        function=i_chord.function or "I",
        form_section=i_chord.form_section,
    ))

    return result


def _full_coltrane_cycle(
    v_chord: ChordEvent, i_chord: ChordEvent
) -> List[ChordEvent]:
    """Heavy density: full 6-chord descending major-third cycle.

    Consumes the total duration of V + I and distributes it across 6 chords.
    The first 5 chords get equal slices; chord 6 (the resolution to I) gets
    the remainder, ensuring the total duration is preserved exactly.

    Cycle (intervals from I root):
      1. bVII7   (+10, dom7)
      2. IVmaj7  (+5,  maj7)
      3. bVI7    (+8,  dom7)
      4. bIIImaj7(+3,  maj7)
      5. bV7     (+6,  dom7)
      6. Imaj7   (+0,  original I quality)
    """
    i_root = i_chord.root_pc
    total_dur = v_chord.duration_beats + i_chord.duration_beats
    slice_dur = total_dur / 6.0

    cycle_specs = [
        (10, "7",    "bVII"),
        (5,  "maj7", "IV"),
        (8,  "7",    "bVI"),
        (3,  "maj7", "bIII"),
        (6,  "7",    "bV"),
    ]

    result: List[ChordEvent] = []
    current_beat = v_chord.start_beat

    for interval, quality, fn in cycle_specs:
        root = (i_root + interval) % 12
        result.append(ChordEvent(
            root_pc=root,
            quality=quality,
            start_beat=current_beat,
            duration_beats=slice_dur,
            key_center_pc=_key_center_for_sub(root, quality),
            function=fn,
            form_section=v_chord.form_section,
        ))
        current_beat += slice_dur

    # Final chord: resolution to I, gets whatever duration remains
    remainder = total_dur - 5 * slice_dur
    result.append(ChordEvent(
        root_pc=i_chord.root_pc,
        quality=i_chord.quality,
        start_beat=current_beat,
        duration_beats=remainder,
        key_center_pc=i_chord.root_pc,
        function=i_chord.function or "I",
        form_section=i_chord.form_section,
    ))

    return result


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_SUBSTITUTION_FN = {
    "light": _tritone_sub,
    "medium": _short_coltrane_cycle,
    "heavy": _full_coltrane_cycle,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def reharmonize(
    chords: List[ChordEvent], density: str = "off"
) -> List[ChordEvent]:
    """Apply Coltrane-style reharmonization to a chord progression.

    Scans for V7 -> Imaj resolutions and expands them with multi-tonic
    substitutions at the requested density level.

    Args:
        chords: Input chord progression (list of ChordEvent). Must be sorted
                by start_beat in ascending order.
        density: One of "off", "light", "medium", "heavy".

    Returns:
        A new list of ChordEvent with the same total duration but expanded
        harmony at cadence points. Non-cadential chords are preserved
        unchanged.

    Raises:
        ValueError: If density is not a recognized level.
    """
    if density not in _VALID_DENSITIES:
        raise ValueError(
            f"Unknown density '{density}'. "
            f"Must be one of: {', '.join(sorted(_VALID_DENSITIES))}"
        )

    if density == "off" or len(chords) < 2:
        return list(chords)

    sub_fn = _SUBSTITUTION_FN[density]
    result: List[ChordEvent] = []
    i = 0

    while i < len(chords):
        # Check if this chord and the next form a V -> I resolution
        if i + 1 < len(chords) and _is_v_to_i(chords[i], chords[i + 1]):
            v_chord = chords[i]
            i_chord = chords[i + 1]

            # Skip reharmonization if either chord is too short to subdivide
            if (v_chord.duration_beats < _MIN_DURATION_FOR_REHARM
                    or i_chord.duration_beats < _MIN_DURATION_FOR_REHARM):
                result.append(v_chord)
                i += 1
                continue

            # Apply the substitution and skip both V and I chords
            expanded = sub_fn(v_chord, i_chord)
            result.extend(expanded)
            i += 2
        else:
            # Not a cadence -- pass through unchanged
            result.append(chords[i])
            i += 1

    return result
