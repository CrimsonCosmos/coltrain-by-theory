"""Chord representation, parsing, and tone classification for Coltrain."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .pitch import SEMITONE_NAMES, NOTE_TO_PC, pc_to_name

# ---------------------------------------------------------------------------
# Chord tone intervals by quality
# ---------------------------------------------------------------------------

CHORD_TONES = {
    # Seventh chords
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "dom7": (0, 4, 7, 10),
    "7": (0, 4, 7, 10),
    "min7b5": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9),
    "aug7": (0, 4, 8, 10),
    "minmaj7": (0, 3, 7, 11),
    # Triads
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    # Sus and sixth chords
    "sus4": (0, 5, 7, 10),
    "sus2": (0, 2, 7, 10),
    "min6": (0, 3, 7, 9),
    "6": (0, 4, 7, 9),
}

# ---------------------------------------------------------------------------
# Available tensions (extensions) by quality, as semitone intervals from root
# ---------------------------------------------------------------------------

TENSIONS = {
    "maj7": [14, 18, 21],       # 9, #11, 13
    "dom7": [14, 21],           # 9, 13
    "min7": [14, 17, 21],       # 9, 11, 13
    "min7b5": [14, 17, 20],     # 9, 11, b13
    "dim7": [14, 17, 20],       # 9, 11, b13
    "aug7": [14, 18],           # 9, #11
    "minmaj7": [14, 17, 21],    # 9, 11, 13
    "sus4": [14, 21],           # 9, 13
    "sus2": [17, 21],           # 11, 13
    "min6": [14, 17],           # 9, 11
    "6": [14, 18],              # 9, #11
    "7": [14, 21],              # 9, 13 (alias for dom7)
    "maj": [14],                # 9
    "min": [14],                # 9
    "dim": [],
    "aug": [],
}

# ---------------------------------------------------------------------------
# ChordEvent dataclass
# ---------------------------------------------------------------------------


@dataclass
class ChordEvent:
    """A chord occurring at a specific point in a progression."""
    root_pc: int                    # Pitch class 0-11
    quality: str                    # Key into CHORD_TONES
    start_beat: float               # Absolute beat position in the arrangement
    duration_beats: float           # How long this chord lasts
    key_center_pc: int = 0          # For Coltrane multi-tonic analysis
    function: str = ""              # Roman numeral function: 'I', 'ii', 'V', etc.

    @property
    def end_beat(self) -> float:
        return self.start_beat + self.duration_beats

    @property
    def label(self) -> str:
        return chord_label(self.root_pc, self.quality)

    def __repr__(self) -> str:
        return (
            f"ChordEvent({self.label}, beat={self.start_beat}, "
            f"dur={self.duration_beats}, fn={self.function})"
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def get_chord_tones_midi(root_pc: int, quality: str, low: int, high: int) -> List[int]:
    """Return all MIDI notes that are chord tones within [low, high] inclusive.

    Args:
        root_pc: Root pitch class (0-11).
        quality: Chord quality string (key into CHORD_TONES).
        low: Lowest MIDI note to include.
        high: Highest MIDI note to include.

    Returns:
        Sorted list of MIDI note numbers.
    """
    if quality not in CHORD_TONES:
        raise ValueError(f"Unknown chord quality: {quality}")

    intervals = CHORD_TONES[quality]
    result = []
    for midi_note in range(low, high + 1):
        pc = midi_note % 12
        interval = (pc - root_pc) % 12
        if interval in intervals:
            result.append(midi_note)
    return result


# Quality aliases used by parse_chord_string
_QUALITY_ALIASES = [
    # Longer patterns first to avoid partial matches
    ("maj7", "maj7"),
    ("Maj7", "maj7"),
    ("M7", "maj7"),
    ("ma7", "maj7"),
    ("delta", "maj7"),
    ("min7b5", "min7b5"),
    ("m7b5", "min7b5"),
    ("-7b5", "min7b5"),
    ("min/maj7", "minmaj7"),
    ("m/M7", "minmaj7"),
    ("minmaj7", "minmaj7"),
    ("mM7", "minmaj7"),
    ("min7", "min7"),
    ("m7", "min7"),
    ("-7", "min7"),
    ("mi7", "min7"),
    ("dim7", "dim7"),
    ("o7", "dim7"),
    ("aug7", "aug7"),
    ("+7", "aug7"),
    ("7sus4", "sus4"),
    ("7sus2", "sus2"),
    ("sus4", "sus4"),
    ("sus2", "sus2"),
    ("min6", "min6"),
    ("m6", "min6"),
    ("-6", "min6"),
    ("dom7", "dom7"),
    ("7", "7"),
    ("maj", "maj"),
    ("Maj", "maj"),
    ("M", "maj"),
    ("min", "min"),
    ("m", "min"),
    ("-", "min"),
    ("mi", "min"),
    ("dim", "dim"),
    ("o", "dim"),
    ("aug", "aug"),
    ("+", "aug"),
    ("6", "6"),
]


def parse_chord_string(s: str) -> Tuple[int, str]:
    """Parse a chord string like 'Cmaj7', 'Bbmin7', 'F#7' into (root_pc, quality).

    Returns:
        Tuple of (root_pc, quality_string).

    Raises:
        ValueError: If the chord string cannot be parsed.
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty chord string")

    # Extract root note (1 or 2 characters)
    root_name = None
    rest = None

    if len(s) >= 2 and s[1] in ("#", "b"):
        root_name = s[:2]
        rest = s[2:]
    else:
        root_name = s[0]
        rest = s[1:]

    if root_name not in NOTE_TO_PC:
        raise ValueError(f"Unknown root note: {root_name}")

    root_pc = NOTE_TO_PC[root_name]

    # Match quality
    if rest == "":
        return (root_pc, "maj")

    for alias, quality in _QUALITY_ALIASES:
        if rest == alias:
            return (root_pc, quality)

    # Try startswith for compound suffixes (e.g., "7#9" -> treat as "7")
    for alias, quality in _QUALITY_ALIASES:
        if rest.startswith(alias):
            return (root_pc, quality)

    raise ValueError(f"Unknown chord quality: {rest} in '{s}'")


def chord_label(root_pc: int, quality: str) -> str:
    """Build a human-readable chord label like 'Cmaj7', 'Bbm7'.

    Uses shorthand: min7 -> m7, dom7 -> 7, min7b5 -> m7b5, etc.
    """
    root = pc_to_name(root_pc)

    shorthand = {
        "min7": "m7",
        "min": "m",
        "dom7": "7",
        "min7b5": "m7b5",
        "dim7": "dim7",
        "aug7": "+7",
        "minmaj7": "mM7",
        "min6": "m6",
    }

    q = shorthand.get(quality, quality)
    return f"{root}{q}"
