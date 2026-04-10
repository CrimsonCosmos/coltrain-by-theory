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
