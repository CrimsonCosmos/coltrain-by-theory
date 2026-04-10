"""Chord progressions, form templates, and harmonic structure for Coltrain.

Provides standard jazz forms (blues, rhythm changes, Giant Steps, etc.)
and utilities for building chord sequences from templates.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .chord import ChordEvent
from .pitch import NOTE_TO_PC

# ---------------------------------------------------------------------------
# KEY_MAP: note name -> pitch class
# ---------------------------------------------------------------------------

KEY_MAP = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}


# ---------------------------------------------------------------------------
# FormTemplate
# ---------------------------------------------------------------------------

@dataclass
class FormTemplate:
    """Describes a jazz form (blues, AABA, etc.) as a chord chart template.

    Attributes:
        name: Human-readable name.
        bars: Total number of bars in one chorus.
        sections: List of (label, start_bar, end_bar) for structural sections.
            start_bar is inclusive, end_bar is exclusive.
        changes: List of (bar, beat, root_offset, quality) tuples.
            bar is 0-indexed, beat is 1-indexed within the bar (1-4),
            root_offset is semitones above the key root.
        beats_per_bar: Time signature numerator (default 4).
        key_center_offsets: Optional dict mapping (bar, beat) to key center
            offset in semitones (for Coltrane multi-tonic). If None, key
            center equals the form's root key.
    """
    name: str
    bars: int
    sections: List[Tuple[str, int, int]]
    changes: List[Tuple[int, int, int, str]]
    beats_per_bar: int = 4
    key_center_offsets: Optional[Dict[Tuple[int, int], int]] = None


# ---------------------------------------------------------------------------
# Form template definitions
# ---------------------------------------------------------------------------

# 12-bar blues with bebop passing chords
_BLUES_12 = FormTemplate(
    name="blues12",
    bars=12,
    sections=[
        ("A", 0, 4),
        ("B", 4, 8),
        ("C", 8, 12),
    ],
    changes=[
        # bar, beat, root_offset, quality
        (0, 1, 0, "7"),       # I7
        (1, 1, 0, "7"),       # I7
        (2, 1, 0, "7"),       # I7
        (3, 1, 0, "7"),       # I7
        (4, 1, 5, "7"),       # IV7
        (5, 1, 5, "7"),       # IV7
        (6, 1, 0, "7"),       # I7
        (7, 1, 0, "7"),       # I7
        (8, 1, 7, "7"),       # V7
        (9, 1, 5, "7"),       # IV7
        (10, 1, 0, "7"),      # I7
        (11, 1, 7, "7"),      # V7 (turnaround)
    ],
)

# Charlie Parker / bebop blues changes
_BLUES_BIRD = FormTemplate(
    name="blues_bird",
    bars=12,
    sections=[
        ("A", 0, 4),
        ("B", 4, 8),
        ("C", 8, 12),
    ],
    changes=[
        (0, 1, 0, "7"),       # I7
        (1, 1, 5, "7"),       # IV7
        (2, 1, 0, "7"),       # I7
        (2, 3, 1, "7"),       # bII7 (passing)
        (3, 1, 0, "7"),       # I7
        (3, 3, 10, "min7"),   # viim7 (ii of IV)
        (4, 1, 5, "7"),       # IV7
        (5, 1, 1, "dim7"),    # #IVdim7
        (6, 1, 0, "7"),       # I7
        (6, 3, 9, "min7"),    # vim7
        (7, 1, 2, "min7"),    # iim7
        (7, 3, 7, "7"),       # V7
        (8, 1, 4, "min7"),    # iiim7
        (8, 3, 9, "7"),       # VI7
        (9, 1, 2, "min7"),    # iim7
        (9, 3, 7, "7"),       # V7
        (10, 1, 0, "7"),      # I7
        (10, 3, 9, "7"),      # VI7
        (11, 1, 2, "min7"),   # iim7
        (11, 3, 7, "7"),      # V7
    ],
)

# Rhythm Changes (32-bar AABA)
_RHYTHM_CHANGES = FormTemplate(
    name="rhythm_changes",
    bars=32,
    sections=[
        ("A1", 0, 8),
        ("A2", 8, 16),
        ("B", 16, 24),
        ("A3", 24, 32),
    ],
    changes=[
        # A1 section: I-vi-ii-V pattern
        (0, 1, 0, "maj7"),    # Imaj7
        (0, 3, 9, "min7"),    # vim7
        (1, 1, 2, "min7"),    # iim7
        (1, 3, 7, "7"),       # V7
        (2, 1, 0, "maj7"),    # Imaj7
        (2, 3, 9, "min7"),    # vim7
        (3, 1, 2, "min7"),    # iim7
        (3, 3, 7, "7"),       # V7
        (4, 1, 0, "maj7"),    # Imaj7
        (4, 3, 4, "7"),       # III7 (secondary dom)
        (5, 1, 9, "min7"),    # vim7
        (5, 3, 2, "7"),       # II7 (secondary dom)
        (6, 1, 2, "min7"),    # iim7
        (6, 3, 7, "7"),       # V7
        (7, 1, 0, "maj7"),    # Imaj7
        (7, 3, 7, "7"),       # V7 (turnaround)

        # A2 section (same as A1)
        (8, 1, 0, "maj7"),
        (8, 3, 9, "min7"),
        (9, 1, 2, "min7"),
        (9, 3, 7, "7"),
        (10, 1, 0, "maj7"),
        (10, 3, 9, "min7"),
        (11, 1, 2, "min7"),
        (11, 3, 7, "7"),
        (12, 1, 0, "maj7"),
        (12, 3, 4, "7"),
        (13, 1, 9, "min7"),
        (13, 3, 2, "7"),
        (14, 1, 2, "min7"),
        (14, 3, 7, "7"),
        (15, 1, 0, "maj7"),
        (15, 3, 7, "7"),

        # B section: cycle of dominants (III7 - VI7 - II7 - V7)
        (16, 1, 4, "7"),      # III7
        (17, 1, 4, "7"),      # III7
        (18, 1, 9, "7"),      # VI7
        (19, 1, 9, "7"),      # VI7
        (20, 1, 2, "7"),      # II7
        (21, 1, 2, "7"),      # II7
        (22, 1, 7, "7"),      # V7
        (23, 1, 7, "7"),      # V7

        # A3 section (same as A1)
        (24, 1, 0, "maj7"),
        (24, 3, 9, "min7"),
        (25, 1, 2, "min7"),
        (25, 3, 7, "7"),
        (26, 1, 0, "maj7"),
        (26, 3, 9, "min7"),
        (27, 1, 2, "min7"),
        (27, 3, 7, "7"),
        (28, 1, 0, "maj7"),
        (28, 3, 4, "7"),
        (29, 1, 9, "min7"),
        (29, 3, 2, "7"),
        (30, 1, 2, "min7"),
        (30, 3, 7, "7"),
        (31, 1, 0, "maj7"),
        (31, 3, 7, "7"),
    ],
)

# Generic 32-bar AABA standard
_AABA_32 = FormTemplate(
    name="aaba32",
    bars=32,
    sections=[
        ("A1", 0, 8),
        ("A2", 8, 16),
        ("B", 16, 24),
        ("A3", 24, 32),
    ],
    changes=[
        # A1
        (0, 1, 0, "maj7"),     # Imaj7
        (1, 1, 0, "maj7"),
        (2, 1, 2, "min7"),     # iim7
        (2, 3, 7, "7"),        # V7
        (3, 1, 0, "maj7"),
        (3, 3, 7, "7"),
        (4, 1, 0, "maj7"),
        (5, 1, 9, "min7"),     # vim7
        (5, 3, 2, "7"),        # II7
        (6, 1, 2, "min7"),     # iim7
        (6, 3, 7, "7"),        # V7
        (7, 1, 0, "maj7"),
        (7, 3, 7, "7"),

        # A2 (same)
        (8, 1, 0, "maj7"),
        (9, 1, 0, "maj7"),
        (10, 1, 2, "min7"),
        (10, 3, 7, "7"),
        (11, 1, 0, "maj7"),
        (11, 3, 7, "7"),
        (12, 1, 0, "maj7"),
        (13, 1, 9, "min7"),
        (13, 3, 2, "7"),
        (14, 1, 2, "min7"),
        (14, 3, 7, "7"),
        (15, 1, 0, "maj7"),
        (15, 3, 7, "7"),

        # B (bridge -- typically different key area)
        (16, 1, 4, "7"),       # III7 (V of vi)
        (17, 1, 4, "7"),
        (18, 1, 9, "7"),       # VI7 (V of ii)
        (19, 1, 9, "7"),
        (20, 1, 2, "7"),       # II7 (V of V)
        (21, 1, 2, "7"),
        (22, 1, 2, "min7"),    # iim7
        (22, 3, 7, "7"),       # V7
        (23, 1, 2, "min7"),
        (23, 3, 7, "7"),

        # A3 (same as A1)
        (24, 1, 0, "maj7"),
        (25, 1, 0, "maj7"),
        (26, 1, 2, "min7"),
        (26, 3, 7, "7"),
        (27, 1, 0, "maj7"),
        (27, 3, 7, "7"),
        (28, 1, 0, "maj7"),
        (29, 1, 9, "min7"),
        (29, 3, 2, "7"),
        (30, 1, 2, "min7"),
        (30, 3, 7, "7"),
        (31, 1, 0, "maj7"),
        (31, 3, 0, "maj7"),
    ],
)

# Giant Steps -- Coltrane's 16-bar form with 3 key centers at major 3rds
# Key centers: root (B=0), +4 semitones (G for B), +8 semitones (Eb for B)
# Harmonic rhythm: 2 beats per chord
_GIANT_STEPS = FormTemplate(
    name="giantsteps",
    bars=16,
    sections=[
        ("A", 0, 8),
        ("B", 8, 16),
    ],
    changes=[
        # A section
        # Bar 0: Bmaj7 (2 beats) | D7 (2 beats)
        (0, 1, 0, "maj7"),    # I of key center 1
        (0, 3, 2, "7"),       # V7 of key center 2
        # Bar 1: Gmaj7 (2 beats) | Bb7 (2 beats)
        (1, 1, 7, "maj7"),    # I of key center 2 (root+7 if root=B -> G)
        (1, 3, 10, "7"),      # V7 of key center 3
        # Bar 2: Ebmaj7 (4 beats)
        (2, 1, 3, "maj7"),    # I of key center 3 (root+3 -> Eb for B)
        # Bar 3: Am7 (2 beats) | D7 (2 beats)
        (3, 1, 9, "min7"),    # iim7 of key center 2
        (3, 3, 2, "7"),       # V7 of key center 2
        # Bar 4: Gmaj7 (2 beats) | Bb7 (2 beats)
        (4, 1, 7, "maj7"),    # I of key center 2
        (4, 3, 10, "7"),      # V7 of key center 3
        # Bar 5: Ebmaj7 (4 beats)
        (5, 1, 3, "maj7"),    # I of key center 3
        # Bar 6: F#m7 (2 beats) | B7 (2 beats)
        (6, 1, 6, "min7"),    # iim7 of key center 1
        (6, 3, 0, "7"),       # V7 of key center 1 (resolving back)
        # Bar 7: Bmaj7 (2 beats) | D7 (2 beats) -- leading to B section (or repeat)
        # Actually in Giant Steps, bar 7 is: Emaj7 (2 beats) | (no, let me be precise)

        # Let me use the actual Giant Steps changes precisely:
        # The real changes (in B):
        # |Bmaj7   D7  |Gmaj7   Bb7 |Ebmaj7       |Am7    D7  |
        # |Gmaj7   Bb7 |Ebmaj7      |F#m7   B7    |Emaj7      |
        # |Am7     D7  |Gmaj7   Bb7 |Ebmaj7       |F#m7   B7  |
        # |Emaj7       |C#m7   F#7  |Bmaj7        |Fm7    Bb7 |

        # Correction -- let me redo from bar 7 onward:
        (7, 1, 4, "maj7"),    # Emaj7 (key center 1 IV? No -- E is the third key center offset by P4 from B)

        # B section
        (8, 1, 9, "min7"),    # Am7 -- ii of G
        (8, 3, 2, "7"),       # D7 -- V of G
        (9, 1, 7, "maj7"),    # Gmaj7
        (9, 3, 10, "7"),      # Bb7 -- V of Eb
        (10, 1, 3, "maj7"),   # Ebmaj7
        (11, 1, 6, "min7"),   # F#m7 -- ii of E (or B depending on analysis)
        (11, 3, 0, "7"),      # B7 -- V of E
        (12, 1, 4, "maj7"),   # Emaj7
        (13, 1, 1, "min7"),   # C#m7 -- ii of B
        (13, 3, 6, "7"),      # F#7 -- V of B
        (14, 1, 0, "maj7"),   # Bmaj7
        (15, 1, 5, "min7"),   # Fm7 -- ii of Eb
        (15, 3, 10, "7"),     # Bb7 -- V of Eb (turnaround)
    ],
    key_center_offsets={
        # Key center 1 = root (0), Key center 2 = root+8 (G for B), Key center 3 = root+4 (Eb for B)
        # Wait -- for Giant Steps in B: key centers are B, G, Eb
        # B=0 offset, G=7 offset from B? No: key centers at major 3rds apart.
        # B(0), Eb(3), G(7) -- dividing the octave in major thirds
        # Each chord belongs to a key center:
        (0, 1): 0,    # Bmaj7 -> B key center
        (0, 3): 7,    # D7 -> V of G key center
        (1, 1): 7,    # Gmaj7 -> G key center
        (1, 3): 3,    # Bb7 -> V of Eb key center
        (2, 1): 3,    # Ebmaj7 -> Eb key center
        (3, 1): 7,    # Am7 -> ii of G
        (3, 3): 7,    # D7 -> V of G
        (4, 1): 7,    # Gmaj7 -> G
        (4, 3): 3,    # Bb7 -> V of Eb
        (5, 1): 3,    # Ebmaj7 -> Eb
        (6, 1): 0,    # F#m7 -> ii of B? Actually ii of E. Let's use E=4
        (6, 3): 0,    # B7 -> could be V of E (key_center=4)
        (7, 1): 4,    # Emaj7 -> E key center (really this is a 4th center)
        (8, 1): 7,    # Am7 -> ii of G
        (8, 3): 7,    # D7 -> V of G
        (9, 1): 7,    # Gmaj7 -> G
        (9, 3): 3,    # Bb7 -> V of Eb
        (10, 1): 3,   # Ebmaj7 -> Eb
        (11, 1): 0,   # F#m7 -> ii of B
        (11, 3): 0,   # B7 -> V of E
        (12, 1): 4,   # Emaj7 -> E
        (13, 1): 0,   # C#m7 -> ii of B
        (13, 3): 0,   # F#7 -> V of B
        (14, 1): 0,   # Bmaj7 -> B
        (15, 1): 3,   # Fm7 -> ii of Eb
        (15, 3): 3,   # Bb7 -> V of Eb
    },
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FORM_TEMPLATES: Dict[str, FormTemplate] = {
    "blues12": _BLUES_12,
    "blues_bird": _BLUES_BIRD,
    "rhythm_changes": _RHYTHM_CHANGES,
    "aaba32": _AABA_32,
    "giantsteps": _GIANT_STEPS,
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def transpose_changes(
    changes: List[Tuple[int, int, int, str]],
    key_pc: int,
    beats_per_bar: int = 4,
    key_center_offsets: Optional[Dict[Tuple[int, int], int]] = None,
) -> List[ChordEvent]:
    """Transpose template changes to an absolute key and return ChordEvents.

    Args:
        changes: List of (bar, beat, root_offset, quality) from a FormTemplate.
        key_pc: Key root pitch class (0-11).
        beats_per_bar: Beats per bar (default 4).
        key_center_offsets: Optional dict of (bar, beat) -> key center offset.

    Returns:
        List of ChordEvent with absolute pitch classes and beat positions.
    """
    events = []

    for i, (bar, beat, root_offset, quality) in enumerate(changes):
        root_pc = (key_pc + root_offset) % 12
        # beat is 1-indexed, convert to 0-indexed absolute beat
        start_beat = bar * beats_per_bar + (beat - 1)

        # Determine duration: extends until the next chord
        if i + 1 < len(changes):
            next_bar, next_beat = changes[i + 1][0], changes[i + 1][1]
            next_start = next_bar * beats_per_bar + (next_beat - 1)
            duration = next_start - start_beat
        else:
            # Last chord: assume it fills to the end of its bar
            bar_end = (bar + 1) * beats_per_bar
            duration = bar_end - start_beat

        # Key center
        if key_center_offsets and (bar, beat) in key_center_offsets:
            kc_offset = key_center_offsets[(bar, beat)]
            kc_pc = (key_pc + kc_offset) % 12
        else:
            kc_pc = key_pc

        # Infer function (rough heuristic based on interval from key center)
        interval_from_kc = (root_pc - kc_pc) % 12
        fn = _interval_to_function(interval_from_kc, quality)

        events.append(ChordEvent(
            root_pc=root_pc,
            quality=quality,
            start_beat=float(start_beat),
            duration_beats=float(duration),
            key_center_pc=kc_pc,
            function=fn,
        ))

    return events


def _interval_to_function(interval: int, quality: str) -> str:
    """Heuristic mapping of interval from key center + quality to Roman numeral."""
    # Major key function map
    fn_map = {
        (0, "maj7"): "I",
        (0, "maj"): "I",
        (0, "7"): "I7",
        (0, "dom7"): "I7",
        (2, "min7"): "ii",
        (2, "min"): "ii",
        (2, "7"): "II7",
        (2, "dom7"): "II7",
        (3, "min7"): "biii",
        (3, "maj7"): "bIII",
        (4, "min7"): "iii",
        (4, "7"): "III7",
        (4, "dom7"): "III7",
        (4, "maj7"): "III",
        (5, "maj7"): "IV",
        (5, "maj"): "IV",
        (5, "7"): "IV7",
        (5, "dom7"): "IV7",
        (5, "min7"): "iv",
        (6, "min7"): "#iv",
        (6, "dim7"): "#ivo",
        (6, "7"): "#IV7",
        (7, "7"): "V",
        (7, "dom7"): "V",
        (7, "maj7"): "V",
        (7, "maj"): "V",
        (7, "min7"): "v",
        (8, "maj7"): "bVI",
        (9, "min7"): "vi",
        (9, "7"): "VI7",
        (9, "dom7"): "VI7",
        (9, "min7b5"): "vi",
        (10, "7"): "bVII7",
        (10, "dom7"): "bVII7",
        (10, "min7"): "bvii",
        (11, "min7b5"): "vii",
        (11, "dim7"): "viio",
        (1, "min7"): "bii",
        (1, "dim7"): "#Io",
        (1, "7"): "bII7",
    }

    return fn_map.get((interval, quality), "?")


def build_chord_sequence(
    form_name: str,
    key_pc: int,
    num_choruses: int = 1,
) -> List[ChordEvent]:
    """Build a complete chord sequence from a form template.

    Args:
        form_name: Key into FORM_TEMPLATES.
        key_pc: Key root pitch class (0-11).
        num_choruses: Number of times to repeat the form.

    Returns:
        List of ChordEvent sorted by start_beat.

    Raises:
        ValueError: If form_name is unknown.
    """
    if form_name not in FORM_TEMPLATES:
        raise ValueError(
            f"Unknown form: {form_name}. "
            f"Available: {list(FORM_TEMPLATES.keys())}"
        )

    template = FORM_TEMPLATES[form_name]
    total_beats_per_chorus = template.bars * template.beats_per_bar
    all_events = []

    for chorus in range(num_choruses):
        beat_offset = chorus * total_beats_per_chorus
        events = transpose_changes(
            template.changes,
            key_pc,
            template.beats_per_bar,
            template.key_center_offsets,
        )
        # Offset beat positions for this chorus
        for ev in events:
            ev.start_beat += beat_offset
        all_events.extend(events)

    # Sort by start beat
    all_events.sort(key=lambda e: e.start_beat)
    return all_events


def generate_key_centers(root_pc: int, divisions: int) -> List[int]:
    """Divide the octave equally into `divisions` key centers.

    This is the geometric harmony concept behind Giant Steps and other
    Coltrane compositions. Dividing by 3 gives major thirds (the
    Coltrane multi-tonic system), by 4 gives minor thirds, by 6 gives
    whole tones, etc.

    Args:
        root_pc: Starting pitch class (0-11).
        divisions: Number of equal divisions of the octave.

    Returns:
        List of pitch classes, starting from root_pc.

    Raises:
        ValueError: If divisions doesn't evenly divide 12.
    """
    if 12 % divisions != 0:
        raise ValueError(
            f"divisions={divisions} does not evenly divide 12. "
            f"Valid values: 1, 2, 3, 4, 6, 12."
        )

    step = 12 // divisions
    return [(root_pc + i * step) % 12 for i in range(divisions)]
