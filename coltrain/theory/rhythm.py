"""Timing utilities for Coltrain.

Provides MIDI tick conversions, swing, and humanization functions.
"""

import random

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKS_PER_QUARTER = 480
TICKS_PER_8TH = 240
TICKS_PER_16TH = 120
TICKS_PER_BAR = 1920  # 4/4 time: 4 * 480


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def beat_to_ticks(beat: float) -> int:
    """Convert a beat position (float) to MIDI ticks.

    Beat 0.0 = tick 0, beat 1.0 = tick 480, beat 0.5 = tick 240, etc.
    """
    return round(beat * TICKS_PER_QUARTER)


def ticks_to_beat(ticks: int) -> float:
    """Convert MIDI ticks to a beat position (float)."""
    return ticks / TICKS_PER_QUARTER


# ---------------------------------------------------------------------------
# Swing
# ---------------------------------------------------------------------------


def apply_swing(tick: int, swing_ratio: float = 0.667) -> int:
    """Apply swing feel by shifting offbeat 8th notes toward a triplet position.

    In straight 8ths, each pair of 8ths is evenly spaced (50/50).
    With swing_ratio=0.667, the first 8th gets 2/3 of the beat and
    the second gets 1/3, creating a triplet shuffle feel.

    Args:
        tick: The original tick position.
        swing_ratio: Ratio of the beat given to the downbeat 8th note.
            0.5 = straight, 0.667 = triplet swing, 0.75 = heavy swing.

    Returns:
        The adjusted tick position.
    """
    # Which beat pair does this tick fall in?
    pair_length = TICKS_PER_QUARTER  # Two 8th notes = one quarter note
    pair_index = tick // pair_length
    position_in_pair = tick % pair_length

    # The midpoint of the pair (where the offbeat 8th sits in straight time)
    straight_mid = TICKS_PER_8TH  # 240

    if position_in_pair <= straight_mid:
        # On the downbeat side: scale from [0, 240] -> [0, swing_point]
        swing_point = round(pair_length * swing_ratio)
        if straight_mid > 0:
            new_pos = round(position_in_pair * swing_point / straight_mid)
        else:
            new_pos = 0
    else:
        # On the offbeat side: scale from [240, 480] -> [swing_point, 480]
        swing_point = round(pair_length * swing_ratio)
        remaining_straight = pair_length - straight_mid
        remaining_swung = pair_length - swing_point
        offset = position_in_pair - straight_mid
        if remaining_straight > 0:
            new_pos = swing_point + round(offset * remaining_swung / remaining_straight)
        else:
            new_pos = swing_point

    return pair_index * pair_length + new_pos


# ---------------------------------------------------------------------------
# Humanization
# ---------------------------------------------------------------------------


def humanize_timing(tick: int, amount: int = 10) -> int:
    """Add random timing jitter to a tick position.

    Args:
        tick: Original tick position.
        amount: Maximum jitter in ticks (applied as +/- amount).

    Returns:
        Jittered tick position, never negative.
    """
    jitter = random.randint(-amount, amount)
    return max(0, tick + jitter)


def humanize_velocity(velocity: int, amount: int = 8) -> int:
    """Add random velocity jitter to a MIDI velocity value.

    Args:
        velocity: Original velocity (1-127).
        amount: Maximum jitter (applied as +/- amount).

    Returns:
        Jittered velocity, clamped to 1-127.
    """
    jitter = random.randint(-amount, amount)
    return max(1, min(127, velocity + jitter))
