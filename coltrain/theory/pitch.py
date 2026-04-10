"""Core pitch and note representation for Coltrain."""

SEMITONE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

NOTE_TO_PC = {
    "C": 0, "B#": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "E#": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}

# Interval names indexed by semitone distance (0-11)
_INTERVAL_NAMES = {
    0: "P1",
    1: "m2",
    2: "M2",
    3: "m3",
    4: "M3",
    5: "P4",
    6: "TT",
    7: "P5",
    8: "m6",
    9: "M6",
    10: "m7",
    11: "M7",
}


def pc_to_name(pc: int) -> str:
    """Convert pitch class (0-11) to note name string."""
    return SEMITONE_NAMES[pc % 12]


def note_to_midi(name: str, octave: int) -> int:
    """Convert note name and octave to MIDI number. C4 = 60."""
    if name not in NOTE_TO_PC:
        raise ValueError(f"Unknown note name: {name}")
    return (octave + 1) * 12 + NOTE_TO_PC[name]


def midi_to_note(midi: int) -> tuple:
    """Convert MIDI number to (name, octave) tuple."""
    octave = (midi // 12) - 1
    pc = midi % 12
    return (SEMITONE_NAMES[pc], octave)


def midi_to_pc(midi: int) -> int:
    """Convert MIDI number to pitch class (0-11)."""
    return midi % 12


def interval_name(semitones: int) -> str:
    """Return the interval name for a given number of semitones.

    Examples: 0 -> 'P1', 3 -> 'm3', 7 -> 'P5', 11 -> 'M7'.
    Values outside 0-11 are wrapped to a single octave.
    """
    return _INTERVAL_NAMES[semitones % 12]
