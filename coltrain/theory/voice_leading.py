"""Voice leading optimization for Coltrain.

Builds chord voicings and selects optimal voice leading between chords
by minimizing voice-leading cost (stepwise motion, avoiding parallels).
"""

from typing import List, Optional, Tuple
import itertools

from .chord import CHORD_TONES

# ---------------------------------------------------------------------------
# Voicing type definitions
# Each maps to a function that takes (root_pc, quality) and returns intervals
# from root that define the voicing. These are abstract intervals; actual MIDI
# notes are placed within a target range.
# ---------------------------------------------------------------------------

VOICING_TYPES = ("shell", "rootless_a", "rootless_b", "drop2")

# Intervals for each voicing type, expressed as semitone offsets from root.
# These are computed from chord tones by position.

def _get_voicing_intervals(quality: str, voicing_type: str) -> Optional[Tuple[int, ...]]:
    """Return the semitone intervals for a voicing type applied to a chord quality.

    Returns None if the voicing type is incompatible with the chord quality.
    """
    tones = CHORD_TONES.get(quality)
    if tones is None:
        return None

    if voicing_type == "shell":
        # Root, 3rd, 7th (or highest tone for triads)
        if len(tones) >= 4:
            return (tones[0], tones[1], tones[3])
        elif len(tones) == 3:
            return (tones[0], tones[1], tones[2])
        else:
            return tuple(tones)

    elif voicing_type == "rootless_a":
        # 3rd, 5th, 7th, 9th -- needs at least 4-note chord
        if len(tones) < 4:
            return None
        # 3rd, 5th, 7th, 9th (9th = 14 semitones, wrapped to single octave = 2)
        ninth = 2  # major 9th default
        return (tones[1], tones[2], tones[3], (tones[0] + 14) % 12)

    elif voicing_type == "rootless_b":
        # 7th, 9th, 3rd, 5th -- needs at least 4-note chord
        if len(tones) < 4:
            return None
        ninth = (tones[0] + 14) % 12
        return (tones[3], ninth, tones[1], tones[2])

    elif voicing_type == "drop2":
        # Close-position 7th chord with 2nd voice from top dropped an octave
        # Close position = root, 3, 5, 7 ascending
        if len(tones) < 4:
            return None
        # In close position top-down: 7, 5, 3, root
        # Drop the 2nd from top (5th) down an octave
        # Result: 5(low), root, 3, 7
        return (tones[2], tones[0], tones[1], tones[3])

    return None


def build_voicing(
    root_pc: int,
    quality: str,
    voicing_type: str,
    target_range: Tuple[int, int] = (48, 72),
) -> List[int]:
    """Build a chord voicing within the target MIDI range.

    Args:
        root_pc: Root pitch class (0-11).
        quality: Chord quality string.
        voicing_type: One of VOICING_TYPES.
        target_range: (low, high) MIDI note bounds.

    Returns:
        Sorted list of MIDI note numbers forming the voicing.

    Raises:
        ValueError: If quality or voicing_type is unknown/incompatible.
    """
    intervals = _get_voicing_intervals(quality, voicing_type)
    if intervals is None:
        raise ValueError(
            f"Voicing type '{voicing_type}' is incompatible with quality '{quality}'"
        )

    low, high = target_range
    mid = (low + high) // 2

    # Place each interval as close to the center of the range as possible,
    # while keeping all notes within range and in ascending order.
    notes = []
    for interval in intervals:
        pc = (root_pc + interval) % 12
        # Find the instance of this pitch class closest to mid
        # Start from the lowest possible octave
        base = low + ((pc - low % 12) % 12)
        if base < low:
            base += 12

        # Pick the octave closest to mid
        best = base
        candidate = base
        while candidate <= high:
            if abs(candidate - mid) < abs(best - mid):
                best = candidate
            candidate += 12

        if best > high:
            # Try going down
            best = base
            while best > high:
                best -= 12
            if best < low:
                best = base  # fallback: may be out of range

        notes.append(best)

    # Ensure ascending order by adjusting octaves
    result = _ensure_ascending(notes, low, high)
    return result


def _ensure_ascending(notes: List[int], low: int, high: int) -> List[int]:
    """Adjust notes so they are in ascending order within the range."""
    if not notes:
        return notes

    result = [notes[0]]
    for i in range(1, len(notes)):
        note = notes[i]
        prev = result[-1]
        # Move note up until it's above previous note
        while note <= prev:
            note += 12
        # If it's gone above the range, try bringing the previous down
        if note > high:
            # Try a lower octave that's still above prev
            candidate = note - 12
            if candidate > prev and candidate >= low:
                note = candidate
        result.append(note)

    return sorted(result)


def _all_inversions(notes: List[int], low: int, high: int) -> List[List[int]]:
    """Generate all inversions of a voicing that fit within [low, high]."""
    if not notes:
        return [notes]

    inversions = []
    n = len(notes)

    for rotation in range(n):
        # Rotate: move bottom note up an octave
        rotated = list(notes)
        for _ in range(rotation):
            moved = rotated.pop(0) + 12
            rotated.append(moved)
        rotated.sort()

        # Transpose to fit within range
        # Shift down as much as possible while staying >= low
        while rotated[0] > low + 12:
            rotated = [x - 12 for x in rotated]
        while rotated[0] < low:
            rotated = [x + 12 for x in rotated]

        if all(low <= x <= high for x in rotated):
            inversions.append(rotated)

        # Also try one octave lower
        lower = [x - 12 for x in rotated]
        if all(low <= x <= high for x in lower):
            inversions.append(lower)

        # And one octave higher
        higher = [x + 12 for x in rotated]
        if all(low <= x <= high for x in higher):
            inversions.append(higher)

    return inversions if inversions else [notes]


def voice_leading_cost(voicing_a: List[int], voicing_b: List[int]) -> float:
    """Compute the voice-leading cost between two voicings.

    Lower cost means smoother voice leading.

    Cost components:
    - Sum of absolute semitone distances between corresponding voices.
    - Stepwise motion (0-2 semitones) costs 0; larger intervals cost more.
    - Parallel fifths penalty: +15 per instance.
    - Parallel octaves penalty: +20 per instance.

    If voicings have different numbers of voices, the cost is computed on the
    overlapping voices (by index from bottom), with extra voices adding penalty.
    """
    if not voicing_a or not voicing_b:
        return 0.0

    a = sorted(voicing_a)
    b = sorted(voicing_b)

    min_voices = min(len(a), len(b))
    max_voices = max(len(a), len(b))

    cost = 0.0

    # Penalty for mismatched voice count
    cost += (max_voices - min_voices) * 10.0

    # Interval costs for corresponding voices
    motions = []
    for i in range(min_voices):
        dist = abs(b[i] - a[i])
        if dist <= 2:
            # Stepwise motion -- free
            pass
        else:
            # Larger intervals cost proportionally
            cost += dist * 1.5
        motions.append(b[i] - a[i])

    # Check for parallel fifths and octaves
    for i in range(min_voices):
        for j in range(i + 1, min_voices):
            interval_a = (a[j] - a[i]) % 12
            interval_b = (b[j] - b[i]) % 12

            # Both voices move in the same direction (parallel motion)
            if motions[i] != 0 and motions[j] != 0:
                same_direction = (motions[i] > 0) == (motions[j] > 0)
                if same_direction:
                    if interval_a == 7 and interval_b == 7:
                        cost += 15.0  # Parallel fifths
                    if interval_a == 0 and interval_b == 0:
                        cost += 20.0  # Parallel octaves/unisons

    return cost


def best_voicing(
    root_pc: int,
    quality: str,
    prev_voicing: Optional[List[int]] = None,
    voicing_types: Optional[List[str]] = None,
    target_range: Tuple[int, int] = (48, 72),
) -> List[int]:
    """Find the best voicing for a chord given the previous voicing.

    Tries all specified voicing types and all inversions, picking the one
    with the lowest voice-leading cost from prev_voicing.

    Args:
        root_pc: Root pitch class (0-11).
        quality: Chord quality string.
        prev_voicing: Previous voicing (list of MIDI notes), or None.
        voicing_types: List of voicing type names to try. Defaults to all.
        target_range: (low, high) MIDI note bounds.

    Returns:
        Sorted list of MIDI note numbers for the best voicing.
    """
    if voicing_types is None:
        voicing_types = list(VOICING_TYPES)

    low, high = target_range
    candidates = []

    for vtype in voicing_types:
        try:
            base = build_voicing(root_pc, quality, vtype, target_range)
        except ValueError:
            continue

        # Generate all inversions
        inversions = _all_inversions(base, low, high)
        candidates.extend(inversions)

    if not candidates:
        # Fallback: just place chord tones in range
        tones = CHORD_TONES.get(quality, (0, 4, 7))
        fallback = []
        for t in tones:
            pc = (root_pc + t) % 12
            note = low + ((pc - low % 12) % 12)
            if note < low:
                note += 12
            if note <= high:
                fallback.append(note)
        candidates = [sorted(fallback)] if fallback else [[root_pc + 48]]

    if prev_voicing is None:
        # No previous voicing: pick the candidate closest to center of range
        mid = (low + high) / 2.0
        best = min(candidates, key=lambda v: abs(sum(v) / max(len(v), 1) - mid))
        return best

    # Pick candidate with lowest voice-leading cost
    best_cost = float("inf")
    best_v = candidates[0]
    for v in candidates:
        c = voice_leading_cost(prev_voicing, v)
        if c < best_cost:
            best_cost = c
            best_v = v

    return best_v
