"""Generation package for Coltrain rule-based jazz generation."""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

TICKS_PER_QUARTER = 480
TICKS_PER_8TH = 240
TICKS_PER_16TH = 120
TICKS_PER_BAR = 1920  # 4 beats * 480


# ---------------------------------------------------------------------------
# Core data type
# ---------------------------------------------------------------------------

@dataclass
class NoteEvent:
    """A single note in the generated output."""
    pitch: int          # MIDI pitch (0-127)
    start_tick: int     # Absolute tick position
    duration_ticks: int # Note length in ticks
    velocity: int       # 0-127
    channel: int = 0    # MIDI channel (9 = drums)


@dataclass
class CCEvent:
    """A MIDI Control Change event."""
    cc_number: int      # CC number (e.g. 64 = sustain pedal)
    value: int          # 0-127
    start_tick: int     # Absolute tick position
    channel: int = 0    # MIDI channel


@dataclass
class PitchBendEvent:
    """A MIDI Pitch Bend event."""
    value: int          # -8192 to +8191 (0 = center)
    start_tick: int     # Absolute tick position
    channel: int = 0    # MIDI channel


@dataclass
class BarContext:
    """Per-bar statistics from lead instruments for rhythm section reactivity."""
    density: float = 0.0       # 0.0-1.0, normalized notes-per-bar
    avg_velocity: float = 70.0 # Average velocity this bar
    avg_register: float = 65.0 # Average MIDI pitch
    has_silence: bool = False   # True if < 1 note in bar
    chord_count: int = 1       # Number of chords in this bar (1=stable, 2+=moving)
    form_section: str = ""     # "A", "B", "C" section label
    is_key_change: bool = False  # Key center changes in this bar


@dataclass
class BarFeel:
    """Per-bar rhythmic feel for context-sensitive timing."""
    offset_bias: float = 0.0   # -1.0 (lay back) to +1.0 (push ahead)
    timing_spread: float = 1.0 # 0.5 (tight) to 1.5 (loose/rubato)
    swing_depth: float = 1.0   # 0.8 (less swing) to 1.2 (deeper swing)
