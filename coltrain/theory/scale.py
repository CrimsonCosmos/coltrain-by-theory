"""Scale and mode system for Coltrain.

Provides scale definitions, chord-scale mapping, and note classification
relative to a given chord context.
"""

from typing import List

from .chord import CHORD_TONES

# ---------------------------------------------------------------------------
# Scale interval sets (semitones from root)
# ---------------------------------------------------------------------------

SCALES = {
    # Diatonic modes
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),

    # Melodic minor modes
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),
    "locrian_nat2": (0, 2, 3, 5, 6, 8, 10),
    "lydian_augmented": (0, 2, 4, 6, 8, 9, 11),
    "lydian_dominant": (0, 2, 4, 6, 7, 9, 10),
    "mixolydian_b6": (0, 2, 4, 5, 7, 8, 10),
    "altered": (0, 1, 3, 4, 6, 8, 10),

    # Symmetric scales
    "whole_tone": (0, 2, 4, 6, 8, 10),
    "diminished_hw": (0, 1, 3, 4, 6, 7, 9, 10),
    "diminished_wh": (0, 2, 3, 5, 6, 8, 9, 11),

    # Bebop scales (8 notes -- chromatic passing tone for strong-beat chord tones)
    "bebop_dominant": (0, 2, 4, 5, 7, 9, 10, 11),
    "bebop_major": (0, 2, 4, 5, 7, 8, 9, 11),
    "bebop_minor": (0, 2, 3, 5, 7, 8, 9, 10),

    # Pentatonic / blues
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "blues": (0, 3, 5, 6, 7, 10),
}

# ---------------------------------------------------------------------------
# Chord quality -> preferred scale names (first entry is default)
# ---------------------------------------------------------------------------

CHORD_SCALE_MAP = {
    "maj7": ["ionian", "lydian"],
    "maj": ["ionian", "lydian"],
    "6": ["ionian", "lydian", "major_pentatonic"],
    "dom7": ["mixolydian", "bebop_dominant", "lydian_dominant"],
    "7": ["mixolydian", "bebop_dominant", "lydian_dominant"],
    "min7": ["dorian", "bebop_minor", "aeolian"],
    "min": ["dorian", "aeolian", "minor_pentatonic"],
    "min6": ["dorian", "melodic_minor"],
    "min7b5": ["locrian_nat2", "locrian"],
    "dim7": ["diminished_hw"],
    "dim": ["diminished_hw"],
    "aug7": ["whole_tone", "lydian_augmented"],
    "aug": ["whole_tone", "lydian_augmented"],
    "sus4": ["mixolydian", "dorian"],
    "sus2": ["mixolydian", "dorian"],
    "minmaj7": ["melodic_minor"],
    "7alt": ["altered"],
}

# ---------------------------------------------------------------------------
# Avoid notes by chord quality (intervals that clash with chord tones)
# These are scale degrees that create a minor 9th with a chord tone.
# ---------------------------------------------------------------------------

_AVOID_NOTES = {
    "maj7": frozenset([5]),         # 4th (b9 above 3rd)
    "maj": frozenset([5]),
    "6": frozenset([5]),
    "dom7": frozenset(),            # No strict avoids on dominants
    "7": frozenset(),
    "min7": frozenset([8]),         # b6 (b9 above 5th) in aeolian; dorian avoids none
    "min": frozenset([8]),
    "min6": frozenset(),
    "min7b5": frozenset(),
    "dim7": frozenset(),
    "dim": frozenset(),
    "aug7": frozenset(),
    "aug": frozenset(),
    "sus4": frozenset([4]),         # M3 clashes with sus4
    "sus2": frozenset([3]),         # m3 clashes with sus2
    "minmaj7": frozenset([8]),
    "7alt": frozenset(),
}


def get_scale_notes_midi(root_pc: int, scale_name: str, low: int, high: int) -> List[int]:
    """Return all MIDI notes belonging to a scale within [low, high] inclusive.

    Args:
        root_pc: Root pitch class (0-11).
        scale_name: Key into SCALES dict.
        low: Lowest MIDI note.
        high: Highest MIDI note.

    Returns:
        Sorted list of MIDI note numbers.
    """
    if scale_name not in SCALES:
        raise ValueError(f"Unknown scale: {scale_name}")

    intervals = SCALES[scale_name]
    interval_set = set(intervals)
    result = []
    for midi_note in range(low, high + 1):
        interval = (midi_note % 12 - root_pc) % 12
        if interval in interval_set:
            result.append(midi_note)
    return result


def classify_note(midi_note: int, root_pc: int, quality: str) -> str:
    """Classify a MIDI note relative to a chord as chord_tone, scale_tone, or avoid_note.

    Uses the first (default) scale from CHORD_SCALE_MAP for the given quality.

    Args:
        midi_note: MIDI note number.
        root_pc: Root pitch class (0-11).
        quality: Chord quality string.

    Returns:
        One of: 'chord_tone', 'scale_tone', 'avoid_note'.
    """
    interval = (midi_note % 12 - root_pc) % 12

    # Check chord tone first
    if quality in CHORD_TONES:
        if interval in CHORD_TONES[quality]:
            return "chord_tone"

    # Check avoid notes
    avoid = _AVOID_NOTES.get(quality, frozenset())
    if interval in avoid:
        return "avoid_note"

    # Check if it's in the preferred scale
    scale_names = CHORD_SCALE_MAP.get(quality, [])
    if scale_names:
        primary_scale = SCALES.get(scale_names[0])
        if primary_scale is not None:
            if interval in set(primary_scale):
                return "scale_tone"

    # Not in chord or primary scale -> avoid
    return "avoid_note"
