"""Arrangement engine for Coltrain.

Orchestrates the full jazz arrangement: intro, head in, solos, trading fours,
head out, and coda. Builds chord progressions from form templates and
coordinates all instrument generators into a unified multi-track output.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from coltrain.generation import NoteEvent, CCEvent, PitchBendEvent, BarContext, BarFeel, TICKS_PER_QUARTER, TICKS_PER_BAR, ticks_per_bar
from coltrain.theory.chord import ChordEvent, CHORD_TONES
from coltrain.theory.pitch import NOTE_TO_PC

from coltrain.generation.melody import generate_head_melody, generate_solo, generate_trading_fours, generate_melody_expression
from coltrain.generation.bass import generate_walking_bass, generate_two_feel_bass, generate_modal_bass, generate_bass_solo, generate_bass_expression
from coltrain.generation.drums import generate_drums, generate_modal_drums, generate_brushes_drums, generate_drum_solo, generate_hihat_expression
from coltrain.generation.piano import generate_comping, generate_sustain_pedal, harmonize_head_melody
from coltrain.generation.humanize import humanize_track
from coltrain.generation.reharmonize import reharmonize

# ---------------------------------------------------------------------------
# ArrangementSection data type
# ---------------------------------------------------------------------------


@dataclass
class ArrangementSection:
    """A section of the jazz arrangement."""
    name: str           # 'intro', 'head_in', 'solo', 'trading', 'head_out', 'coda',
                        # 'drum_solo', 'bass_solo'
    start_beat: int
    end_beat: int
    intensity: float    # 0.0-1.0
    is_melody: bool = False
    is_solo: bool = False
    is_trading: bool = False
    is_drum_solo: bool = False
    is_bass_solo: bool = False
    beats_per_bar: int = 4

    @property
    def total_beats(self) -> int:
        return self.end_beat - self.start_beat

    @property
    def total_bars(self) -> int:
        return self.total_beats // self.beats_per_bar

    def __repr__(self) -> str:
        flags = []
        if self.is_melody:
            flags.append("melody")
        if self.is_solo:
            flags.append("solo")
        if self.is_trading:
            flags.append("trading")
        if self.is_drum_solo:
            flags.append("drum solo")
        if self.is_bass_solo:
            flags.append("bass solo")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        bpb = self.beats_per_bar
        return (
            f"  {self.name:12s}  bars {self.start_beat // bpb + 1:3d}-{self.end_beat // bpb:3d}"
            f"  ({self.total_bars} bars)  intensity={self.intensity:.1f}{flag_str}"
        )


# ---------------------------------------------------------------------------
# Form templates: chord progressions in relative notation
# Each entry: (root_interval_from_key, quality, duration_beats)
# Root intervals are semitones above the key center.
# ---------------------------------------------------------------------------

FORM_TEMPLATES = {
    "blues12": {
        "bars": 12,
        "sections": [("A", 0, 16), ("B", 16, 32), ("C", 32, 48)],
        "changes": [
            # Bar 1-4: I7
            (0, "7", 4), (0, "7", 4), (0, "7", 4), (0, "7", 4),
            # Bar 5-6: IV7
            (5, "7", 4), (5, "7", 4),
            # Bar 7-8: I7
            (0, "7", 4), (0, "7", 4),
            # Bar 9: V7
            (7, "7", 4),
            # Bar 10: IV7
            (5, "7", 4),
            # Bar 11-12: I7, V7 (turnaround)
            (0, "7", 4), (7, "7", 4),
        ],
    },
    "blues_bird": {
        "bars": 12,
        "sections": [("A", 0, 16), ("B", 16, 32), ("C", 32, 48)],
        "changes": [
            # Bird blues (Charlie Parker style)
            # Bar 1: Imaj7
            (0, "maj7", 4),
            # Bar 2: iv-7 | bVII7
            (5, "min7", 2), (10, "7", 2),
            # Bar 3: Imaj7
            (0, "maj7", 4),
            # Bar 4: #ivo7 | iv-7
            (6, "dim7", 2), (5, "min7", 2),
            # Bar 5: IVmaj7
            (5, "maj7", 4),
            # Bar 6: iv-7 | bVII7
            (5, "min7", 2), (10, "7", 2),
            # Bar 7: Imaj7
            (0, "maj7", 4),
            # Bar 8: iii-7 | VI7
            (4, "min7", 2), (9, "7", 2),
            # Bar 9: ii-7
            (2, "min7", 4),
            # Bar 10: V7
            (7, "7", 4),
            # Bar 11: Imaj7 | vi-7
            (0, "maj7", 2), (9, "min7", 2),
            # Bar 12: ii-7 | V7
            (2, "min7", 2), (7, "7", 2),
        ],
    },
    "rhythm_changes": {
        "bars": 32,
        "sections": [("A", 0, 32), ("A", 32, 64), ("B", 64, 96), ("A", 96, 128)],
        "changes": [
            # A section (bars 1-8)
            (0, "maj7", 2), (9, "min7", 1), (2, "min7", 1),  # Bb | G-7 C-7
            (5, "min7", 1), (10, "7", 1), (0, "maj7", 2),     # F-7 Bb7 | Ebmaj7
            (4, "min7", 2), (9, "7", 2),                       # D-7 | G7
            (2, "min7", 2), (7, "7", 2),                       # C-7 | F7
            # A section repeat (bars 9-16)
            (0, "maj7", 2), (9, "min7", 1), (2, "min7", 1),
            (5, "min7", 1), (10, "7", 1), (0, "maj7", 2),
            (4, "min7", 2), (9, "7", 2),
            (2, "min7", 2), (7, "7", 2),
            # B section (bridge, bars 17-24) -- cycle of dominants
            (4, "7", 4), (4, "7", 4),                          # D7 | D7
            (9, "7", 4), (9, "7", 4),                          # G7 | G7
            (2, "7", 4), (2, "7", 4),                          # C7 | C7
            (7, "7", 4), (7, "7", 4),                          # F7 | F7
            # A section (bars 25-32)
            (0, "maj7", 2), (9, "min7", 1), (2, "min7", 1),
            (5, "min7", 1), (10, "7", 1), (0, "maj7", 2),
            (4, "min7", 2), (9, "7", 2),
            (2, "min7", 2), (7, "7", 2),
        ],
    },
    "aaba32": {
        "bars": 32,
        "sections": [("A", 0, 32), ("A", 32, 64), ("B", 64, 96), ("A", 96, 128)],
        "changes": [
            # A section (bars 1-8) -- ii-V-I in major
            (2, "min7", 4), (7, "7", 4),                       # ii-7 | V7
            (0, "maj7", 4), (0, "maj7", 4),                   # Imaj7 | Imaj7
            (2, "min7", 4), (7, "7", 4),                       # ii-7 | V7
            (0, "maj7", 4), (0, "maj7", 4),                   # Imaj7 | Imaj7
            # A repeat (bars 9-16)
            (2, "min7", 4), (7, "7", 4),
            (0, "maj7", 4), (0, "maj7", 4),
            (2, "min7", 4), (7, "7", 4),
            (0, "maj7", 4), (0, "maj7", 4),
            # B section (bridge, bars 17-24) -- to IV
            (5, "min7", 4), (10, "7", 4),                      # iv-7 (of IV) | bVII7
            (3, "maj7", 4), (3, "maj7", 4),                   # bIIImaj7 | bIIImaj7
            (2, "min7", 4), (7, "7", 4),                       # ii-7 | V7
            (0, "maj7", 4), (7, "7", 4),                       # Imaj7 | V7
            # A section (bars 25-32)
            (2, "min7", 4), (7, "7", 4),
            (0, "maj7", 4), (0, "maj7", 4),
            (2, "min7", 4), (7, "7", 4),
            (0, "maj7", 4), (0, "maj7", 4),
        ],
    },
    "giantsteps": {
        "bars": 16,
        "sections": [("A", 0, 32), ("B", 32, 64)],
        "changes": [
            # Giant Steps (John Coltrane) -- 16 bars
            # Key centers rotate at major thirds: B, G, Eb
            # Bar 1: Bmaj7 | D7
            (0, "maj7", 2), (3, "7", 2),
            # Bar 2: Gmaj7
            (8, "maj7", 4),
            # Bar 3: Bb7 | Ebmaj7
            (10, "7", 2), (3, "maj7", 2),
            # Bar 4: Am7 | D7
            (9, "min7", 2), (3, "7", 2),
            # Bar 5: Gmaj7 | Bb7
            (8, "maj7", 2), (10, "7", 2),
            # Bar 6: Ebmaj7
            (3, "maj7", 4),
            # Bar 7: F#7 | Bmaj7
            (6, "7", 2), (0, "maj7", 2),
            # Bar 8: Fm7 | Bb7
            (5, "min7", 2), (10, "7", 2),
            # Bar 9: Ebmaj7 | Am7
            (3, "maj7", 2), (9, "min7", 2),
            # Bar 10: D7 | Gmaj7
            (3, "7", 2), (8, "maj7", 2),
            # Bar 11: C#m7 | F#7
            (1, "min7", 2), (6, "7", 2),
            # Bar 12: Bmaj7 | Fm7
            (0, "maj7", 2), (5, "min7", 2),
            # Bar 13: Bb7 | Ebmaj7
            (10, "7", 2), (3, "maj7", 2),
            # Bar 14: Am7 | D7
            (9, "min7", 2), (3, "7", 2),
            # Bar 15: Gmaj7 | C#m7
            (8, "maj7", 2), (1, "min7", 2),
            # Bar 16: F#7 | Bmaj7
            (6, "7", 2), (0, "maj7", 2),
        ],
    },
    "coltrain": {
        "bars": 32,
        "sections": [("A", 0, 32), ("A", 32, 64), ("B", 64, 96), ("A", 96, 128)],
        "changes": [
            # === A section (bars 1-8): bebop-blues grounding ===
            # Bar 1: Imaj7 — home, breathe
            (0, "maj7", 4),
            # Bar 2: vi-7 | II7 — secondary dominant chain
            (9, "min7", 2), (2, "7", 2),
            # Bar 3: ii-7 | V7 — classic ii-V
            (2, "min7", 2), (7, "7", 2),
            # Bar 4: Imaj7 | #iv°7 — resolve + chromatic passing
            (0, "maj7", 2), (6, "dim7", 2),
            # Bar 5: iv-7 | bVII7 — blues minor iv + tritone sub
            (5, "min7", 2), (10, "7", 2),
            # Bar 6: iii-7 | VI7 — secondary dominant
            (4, "min7", 2), (9, "7", 2),
            # Bar 7: ii-7 | V7 — resolving ii-V
            (2, "min7", 2), (7, "7", 2),
            # Bar 8: Imaj7 | V7 — tonic + turnaround
            (0, "maj7", 2), (7, "7", 2),

            # === A section repeat (bars 9-16) ===
            (0, "maj7", 4),
            (9, "min7", 2), (2, "7", 2),
            (2, "min7", 2), (7, "7", 2),
            (0, "maj7", 2), (6, "dim7", 2),
            (5, "min7", 2), (10, "7", 2),
            (4, "min7", 2), (9, "7", 2),
            (2, "min7", 2), (7, "7", 2),
            (0, "maj7", 2), (7, "7", 2),

            # === B section / bridge (bars 17-24): Coltrane adventure ===
            # Bar 17: ii-V to bVI — major-third modulation
            (10, "min7", 2), (3, "7", 2),
            # Bar 18: bVImaj7 — first distant key center
            (8, "maj7", 4),
            # Bar 19: ii-V to III — second major-third modulation
            (6, "min7", 2), (11, "7", 2),
            # Bar 20: IIImaj7 — second distant key center
            (4, "maj7", 4),
            # Bar 21: VI7 | II7 — cycle of dominants back
            (9, "7", 2), (2, "7", 2),
            # Bar 22: V7 — dominant pedal, tension builds
            (7, "7", 4),
            # Bar 23: iii-7 | VI7 — secondary dominant chain
            (4, "min7", 2), (9, "7", 2),
            # Bar 24: ii-7 | V7 — final ii-V home
            (2, "min7", 2), (7, "7", 2),

            # === A section final (bars 25-32) ===
            (0, "maj7", 4),
            (9, "min7", 2), (2, "7", 2),
            (2, "min7", 2), (7, "7", 2),
            (0, "maj7", 2), (6, "dim7", 2),
            (5, "min7", 2), (10, "7", 2),
            (4, "min7", 2), (9, "7", 2),
            (2, "min7", 2), (7, "7", 2),
            (0, "maj7", 2), (7, "7", 2),
        ],
    },
    # ---- Odd-meter modal forms ----
    "modal_5": {
        "bars": 20,
        "beats_per_bar": 5,
        "sections": [("A", 0, 50), ("B", 50, 100)],
        "changes": [
            # A section (bars 1-10): Ebm vamp — Take Five feel
            (3, "min7", 5), (3, "min7", 5), (3, "min7", 5), (3, "min7", 5),
            (3, "min7", 5), (3, "min7", 5), (3, "min7", 5), (3, "min7", 5),
            (3, "min7", 5), (3, "min7", 5),
            # B section (bars 11-20): IV-V motion, brief contrast
            (8, "min7", 5), (10, "7", 5),    # iv-7 | bVII7
            (3, "min7", 5), (3, "min7", 5),  # Ebm7 vamp
            (8, "min7", 5), (10, "7", 5),    # iv-7 | bVII7
            (3, "min7", 5), (3, "min7", 5),  # Ebm7 resolve
            (3, "min7", 5), (3, "min7", 5),  # Ebm7 vamp
        ],
    },
    "modal_7": {
        "bars": 16,
        "beats_per_bar": 7,
        "sections": [("A", 0, 56), ("B", 56, 112)],
        "changes": [
            # A section (bars 1-8): Dorian vamp
            (2, "min7", 7), (2, "min7", 7), (2, "min7", 7), (2, "min7", 7),
            (0, "maj7", 7), (0, "maj7", 7),  # brief tonic major
            (2, "min7", 7), (2, "min7", 7),
            # B section (bars 9-16): modal interchange
            (5, "min7", 7), (5, "min7", 7),  # iv-7
            (7, "7", 7), (7, "7", 7),        # V7
            (2, "min7", 7), (2, "min7", 7),  # back to ii Dorian
            (2, "min7", 7), (2, "min7", 7),
        ],
    },
}

# Key center assignments for Giant Steps (Coltrane multi-tonic system)
# Each chord maps to one of three key centers separated by major thirds.
# For key of B (pc=11): key centers are B(11), G(7), Eb(3)
GIANT_STEPS_KEY_CENTERS = {
    # Intervals from key -> key center interval from key
    0: 0,    # Imaj7 -> key center I (B)
    3: 3,    # Ebmaj7 -> key center bIII (Eb)  / also D7 target
    8: 8,    # Gmaj7 -> key center bVI (G)
    6: 0,    # F#7 -> resolves to I (B)
    10: 3,   # Bb7 -> resolves to bIII (Eb)
    9: 8,    # Am7 -> ii of G key center
    1: 0,    # C#m7 -> ii of B key center
    5: 3,    # Fm7 -> ii of Eb key center
}


# ---------------------------------------------------------------------------
# Chord progression builder
# ---------------------------------------------------------------------------


def _get_form_section(beat_in_chorus: float, sections: list) -> str:
    """Return the form section label for a beat position within one chorus."""
    for label, start_beat, end_beat in sections:
        if start_beat <= beat_in_chorus < end_beat:
            return label
    return ""


def build_chord_progression(
    form_name: str,
    key_pc: int,
    num_choruses: int = 1,
    start_beat: float = 0.0,
    coltrane: bool = False,
) -> List[ChordEvent]:
    """Build a chord progression from a form template.

    Args:
        form_name: Form template name (key into FORM_TEMPLATES).
        key_pc: Key pitch class (0-11). For Giant Steps, this is the
                starting key center (e.g., 11 for B).
        num_choruses: Number of times to repeat the form.
        start_beat: Beat offset for the start of the progression.
        coltrane: If True, assign key centers for multi-tonic analysis.

    Returns:
        List of ChordEvent objects spanning the full progression.

    Raises:
        ValueError: If form_name is not recognized.
    """
    if form_name not in FORM_TEMPLATES:
        raise ValueError(f"Unknown form: {form_name}. Available: {list(FORM_TEMPLATES.keys())}")

    template = FORM_TEMPLATES[form_name]
    changes = template["changes"]
    sections = template.get("sections", [])

    chords: List[ChordEvent] = []
    beat = start_beat

    for chorus_idx in range(num_choruses):
        chorus_start = beat
        for root_interval, quality, duration in changes:
            root_pc = (key_pc + root_interval) % 12

            # Assign key center
            if coltrane and form_name == "giantsteps":
                kc_interval = GIANT_STEPS_KEY_CENTERS.get(root_interval, 0)
                key_center_pc = (key_pc + kc_interval) % 12
            elif coltrane:
                # For non-Giant-Steps forms with Coltrane mode:
                # infer key center from chord function
                key_center_pc = _infer_key_center(root_interval, quality, key_pc)
            else:
                key_center_pc = key_pc

            chords.append(ChordEvent(
                root_pc=root_pc,
                quality=quality,
                start_beat=beat,
                duration_beats=float(duration),
                key_center_pc=key_center_pc,
                form_section=_get_form_section(beat - chorus_start, sections),
            ))
            beat += duration

    return chords


def _infer_key_center(root_interval: int, quality: str, key_pc: int) -> int:
    """Infer the key center for a chord based on its function.

    Heuristic:
    - maj/maj7: chord is tonic of its key -> key_center = root
    - min7: chord is ii of key a 5th above -> key_center = root + 5 (mod 12)
      (actually root - 2 mod 12 gives the I, since ii is 2 above I)
    - dom7/7: chord is V of key a 5th below -> key_center = root - 7 (mod 12)
      (V is 7 above I, so I = root - 7)
    - min7b5: chord is vii of key a semitone above -> key_center = root + 1
    - dim7: enharmonic, use key of the tune
    """
    root_abs = (key_pc + root_interval) % 12

    if quality in ("maj7", "maj", "6"):
        return root_abs
    elif quality in ("min7", "min", "min6"):
        # ii -> I is root - 2
        return (root_abs - 2) % 12
    elif quality in ("dom7", "7"):
        # V -> I is root - 7
        return (root_abs - 7) % 12
    elif quality in ("min7b5",):
        # vii -> I is root + 1
        return (root_abs + 1) % 12
    else:
        return key_pc


# ---------------------------------------------------------------------------
# Arrangement builder
# ---------------------------------------------------------------------------


def build_arrangement(
    form_name: str,
    num_choruses: int,
    bars_per_chorus: int,
    drum_solo: bool = False,
    bass_solo: bool = False,
    beats_per_bar: int = 4,
) -> List[ArrangementSection]:
    """Build the arrangement structure (section layout) for a jazz tune.

    Structure varies by number of choruses:
    - 1 chorus:  intro(4) + head(1 chorus) + coda(4)
    - 2 choruses: intro(4) + head_in(1) + solo(1) + coda(4)
    - 3+: intro(4) + head_in(1) + solos + [bass_solo] + trading + [drum_solo]
           + head_out(1) + coda(4)

    Args:
        form_name: Form template name.
        num_choruses: Number of solo choruses (total arrangement is longer).
        bars_per_chorus: Number of bars in one chorus of the form.
        drum_solo: Insert a 1-chorus drum solo before head_out.
        bass_solo: Insert a 1-chorus bass solo after piano solos.

    Returns:
        List of ArrangementSection objects.
    """
    sections: List[ArrangementSection] = []
    current_beat = 0
    beats_per_chorus = bars_per_chorus * beats_per_bar
    intro_bars = 4
    coda_bars = 4

    # Intro: 4 bars
    intro_end = current_beat + intro_bars * beats_per_bar
    sections.append(ArrangementSection(
        name="intro",
        start_beat=current_beat,
        end_beat=intro_end,
        intensity=0.3,
        is_melody=False,
    ))
    current_beat = intro_end

    if num_choruses == 1:
        # Simple: head only
        head_end = current_beat + beats_per_chorus
        sections.append(ArrangementSection(
            name="head_in",
            start_beat=current_beat,
            end_beat=head_end,
            intensity=0.5,
            is_melody=True,
        ))
        current_beat = head_end
    elif num_choruses == 2:
        # Head in + 1 solo chorus
        head_end = current_beat + beats_per_chorus
        sections.append(ArrangementSection(
            name="head_in",
            start_beat=current_beat,
            end_beat=head_end,
            intensity=0.5,
            is_melody=True,
        ))
        current_beat = head_end

        solo_end = current_beat + beats_per_chorus
        sections.append(ArrangementSection(
            name="solo_1",
            start_beat=current_beat,
            end_beat=solo_end,
            intensity=0.6,
            is_solo=True,
        ))
        current_beat = solo_end
    else:
        # Full arrangement: head_in + solos + trading + head_out
        # Head in
        head_end = current_beat + beats_per_chorus
        sections.append(ArrangementSection(
            name="head_in",
            start_beat=current_beat,
            end_beat=head_end,
            intensity=0.5,
            is_melody=True,
        ))
        current_beat = head_end

        # Solo choruses: ramp intensity across solos
        num_solo_choruses = max(1, num_choruses - 2)  # Reserve 1 for trading, 1 for head_out
        for solo_idx in range(num_solo_choruses):
            # Intensity ramps from 0.5 to 0.8 across solos
            progress = solo_idx / max(1, num_solo_choruses - 1) if num_solo_choruses > 1 else 0.5
            solo_intensity = 0.5 + progress * 0.3

            solo_end = current_beat + beats_per_chorus
            sections.append(ArrangementSection(
                name=f"solo_{solo_idx + 1}",
                start_beat=current_beat,
                end_beat=solo_end,
                intensity=solo_intensity,
                is_solo=True,
            ))
            current_beat = solo_end

        # Bass solo: 1 chorus (after piano solos, before trading)
        if bass_solo:
            bs_end = current_beat + beats_per_chorus
            sections.append(ArrangementSection(
                name="bass_solo",
                start_beat=current_beat,
                end_beat=bs_end,
                intensity=0.6,
                is_bass_solo=True,
            ))
            current_beat = bs_end

        # Trading fours: 1 chorus
        trading_end = current_beat + beats_per_chorus
        sections.append(ArrangementSection(
            name="trading",
            start_beat=current_beat,
            end_beat=trading_end,
            intensity=0.6,
            is_trading=True,
        ))
        current_beat = trading_end

        # Drum solo: 1 chorus (after trading, before head out)
        if drum_solo:
            ds_end = current_beat + beats_per_chorus
            sections.append(ArrangementSection(
                name="drum_solo",
                start_beat=current_beat,
                end_beat=ds_end,
                intensity=0.7,
                is_drum_solo=True,
            ))
            current_beat = ds_end

        # Head out
        head_out_end = current_beat + beats_per_chorus
        sections.append(ArrangementSection(
            name="head_out",
            start_beat=current_beat,
            end_beat=head_out_end,
            intensity=0.5,
            is_melody=True,
        ))
        current_beat = head_out_end

    # Coda: 4 bars
    coda_end = current_beat + coda_bars * beats_per_bar
    sections.append(ArrangementSection(
        name="coda",
        start_beat=current_beat,
        end_beat=coda_end,
        intensity=0.3,
        is_melody=True,
    ))

    # Stamp beats_per_bar on all sections
    for s in sections:
        s.beats_per_bar = beats_per_bar

    return sections


# ---------------------------------------------------------------------------
# Note offset helpers
# ---------------------------------------------------------------------------


def _offset_notes(notes: List[NoteEvent], tick_offset: int) -> List[NoteEvent]:
    """Add a tick offset to all notes' start_tick values."""
    for n in notes:
        n.start_tick += tick_offset
    return notes


def _merge_tracks(
    dest: Dict[str, List[NoteEvent]],
    new_notes: Dict[str, List[NoteEvent]],
) -> None:
    """Merge new_notes into dest by extending each track's list."""
    for track_name, note_list in new_notes.items():
        if track_name not in dest:
            dest[track_name] = []
        dest[track_name].extend(note_list)


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------


def _generate_intro(
    section: ArrangementSection,
    chords: List[ChordEvent],
    swing: bool,
    bass_style: str = "walking",
    drum_style: str = "swing",
    beats_per_bar: int = 4,
) -> Dict[str, List[NoteEvent]]:
    """Generate intro section: sparse piano voicings, two-feel bass, light drums."""
    section_chords = _chords_for_section(chords, section)
    beats = section.total_beats
    tick_offset = section.start_beat * TICKS_PER_QUARTER

    # Generate bass first for sync extraction
    if bass_style == "modal":
        bass_notes = generate_modal_bass(section_chords, beats, swing=swing,
                                         beats_per_bar=beats_per_bar)
    else:
        bass_notes = generate_two_feel_bass(section_chords, beats, swing=swing,
                                            beats_per_bar=beats_per_bar)
    bass_sync = _extract_downbeat_ticks(bass_notes, beats_per_bar)

    if drum_style == "modal":
        drum_notes = generate_modal_drums(beats, intensity=0.2, swing=swing, fill_every=0,
                                          beats_per_bar=beats_per_bar)
    elif drum_style == "brushes":
        drum_notes = generate_brushes_drums(beats, intensity=0.2, swing=swing, fill_every=0,
                                            beats_per_bar=beats_per_bar)
    else:
        drum_notes = generate_drums(beats, intensity=0.2, swing=swing, fill_every=0,
                                    beats_per_bar=beats_per_bar)
    piano_notes = generate_comping(section_chords, beats, intensity=0.2, swing=swing,
                                   bass_sync_ticks=bass_sync, context="head",
                                   beats_per_bar=beats_per_bar)

    _offset_notes(bass_notes, int(tick_offset))
    _offset_notes(drum_notes, int(tick_offset))
    _offset_notes(piano_notes, int(tick_offset))

    return {
        "melody": [],
        "piano": piano_notes,
        "bass": bass_notes,
        "drums": drum_notes,
    }


def _generate_head(
    section: ArrangementSection,
    chords: List[ChordEvent],
    swing: bool,
    coltrane: bool = False,
    bass_style: str = "walking",
    drum_style: str = "swing",
    beats_per_bar: int = 4,
) -> Dict[str, List[NoteEvent]]:
    """Generate head section: melody + full rhythm section."""
    section_chords = _chords_for_section(chords, section)
    beats = section.total_beats
    tick_offset = section.start_beat * TICKS_PER_QUARTER

    melody_notes = generate_head_melody(section_chords, float(beats), swing=swing,
                                        beats_per_bar=beats_per_bar)

    # Harmonize head melody: add block chords below melody on strong beats
    melody_notes = harmonize_head_melody(melody_notes, section_chords, beats, swing=swing,
                                         beats_per_bar=beats_per_bar)

    # Compute per-bar reactive energy from head melody
    bar_energies = _compute_reactive_energy(melody_notes, beats, section.intensity,
                                            beats_per_bar)
    bar_feel = _compute_rhythmic_feel(bar_energies, melody_notes, beats, beats_per_bar)

    # Generate piano first, then extract context for rhythm section
    piano_notes = generate_comping(section_chords, beats, intensity=section.intensity,
                                   swing=swing, coltrane=coltrane,
                                   bass_sync_ticks=[], context="head",
                                   bar_intensities=bar_energies,
                                   bar_feel=bar_feel,
                                   beats_per_bar=beats_per_bar)

    lead_context = _extract_bar_context(melody_notes + piano_notes, beats,
                                        chords=section_chords,
                                        beats_per_bar=beats_per_bar)

    if bass_style == "modal":
        bass_notes = generate_modal_bass(section_chords, beats, swing=swing,
                                         beats_per_bar=beats_per_bar)
    else:
        bass_notes = generate_walking_bass(section_chords, beats, swing=swing,
                                           intensity=section.intensity,
                                           bar_intensities=bar_energies,
                                           bar_context=lead_context,
                                           bar_feel=bar_feel,
                                           beats_per_bar=beats_per_bar)

    if drum_style == "modal":
        drum_notes = generate_modal_drums(beats, intensity=section.intensity, swing=swing,
                                          fill_every=8, beats_per_bar=beats_per_bar)
    elif drum_style == "brushes":
        drum_notes = generate_brushes_drums(beats, intensity=section.intensity, swing=swing,
                                            fill_every=16,
                                            bar_intensities=bar_energies,
                                            bar_context=lead_context,
                                            bar_feel=bar_feel,
                                            beats_per_bar=beats_per_bar)
    else:
        drum_notes = generate_drums(beats, intensity=section.intensity, swing=swing, fill_every=8,
                                    bar_intensities=bar_energies, bar_context=lead_context,
                                    bar_feel=bar_feel, beats_per_bar=beats_per_bar)

    _offset_notes(melody_notes, int(tick_offset))
    _offset_notes(bass_notes, int(tick_offset))
    _offset_notes(drum_notes, int(tick_offset))
    _offset_notes(piano_notes, int(tick_offset))

    return {
        "melody": melody_notes,
        "piano": piano_notes,
        "bass": bass_notes,
        "drums": drum_notes,
    }


def _generate_solo_section(
    section: ArrangementSection,
    chords: List[ChordEvent],
    tension_curve: str,
    swing: bool,
    coltrane: bool,
    seed: Optional[int],
    density_scale: float = 1.0,
    bass_style: str = "walking",
    drum_style: str = "swing",
    beats_per_bar: int = 4,
) -> Dict[str, List[NoteEvent]]:
    """Generate solo section: improvised melody + rhythm section at increasing intensity."""
    section_chords = _chords_for_section(chords, section)
    beats = section.total_beats
    tick_offset = section.start_beat * TICKS_PER_QUARTER

    solo_seed = None
    if seed is not None:
        solo_seed = seed + hash(section.name) % 10000

    # Scale intensity by multi-chorus arc density_scale
    scaled_intensity = min(1.0, section.intensity * density_scale)

    melody_notes = generate_solo(
        section_chords, float(beats),
        tension_curve=tension_curve,
        swing=swing,
        coltrane=coltrane,
        seed=solo_seed,
        intensity=scaled_intensity,
        beats_per_bar=beats_per_bar,
    )

    # Compute per-bar reactive energy from what the melody played
    bar_energies = _compute_reactive_energy(melody_notes, beats, scaled_intensity,
                                            beats_per_bar)
    bar_feel = _compute_rhythmic_feel(bar_energies, melody_notes, beats, beats_per_bar)

    # Generate piano comping first (needs bass sync, but we generate bass after
    # to give it context). Use empty sync for now — piano doesn't need bass context
    # as much as bass/drums need melody+piano context.
    piano_notes = generate_comping(
        section_chords, beats, intensity=scaled_intensity, swing=swing,
        coltrane=coltrane, bass_sync_ticks=[], context="solo",
        bar_intensities=bar_energies,
        bar_feel=bar_feel,
        beats_per_bar=beats_per_bar,
    )

    # Extract context from melody + piano for rhythm section reactivity
    lead_context = _extract_bar_context(melody_notes + piano_notes, beats,
                                        chords=section_chords,
                                        beats_per_bar=beats_per_bar)

    # Generate bass with reactive context
    if bass_style == "modal":
        bass_notes = generate_modal_bass(section_chords, beats, swing=swing,
                                         beats_per_bar=beats_per_bar)
    else:
        bass_notes = generate_walking_bass(section_chords, beats, swing=swing,
                                           intensity=scaled_intensity,
                                           bar_intensities=bar_energies,
                                           bar_context=lead_context,
                                           bar_feel=bar_feel,
                                           beats_per_bar=beats_per_bar)

    if drum_style == "modal":
        drum_notes = generate_modal_drums(
            beats, intensity=scaled_intensity, swing=swing,
            fill_every=4 if scaled_intensity > 0.6 else 8,
            beats_per_bar=beats_per_bar,
        )
    elif drum_style == "brushes":
        drum_notes = generate_brushes_drums(
            beats, intensity=scaled_intensity, swing=swing,
            fill_every=8 if scaled_intensity > 0.6 else 16,
            bar_intensities=bar_energies,
            bar_context=lead_context,
            bar_feel=bar_feel,
            beats_per_bar=beats_per_bar,
        )
    else:
        drum_notes = generate_drums(
            beats, intensity=scaled_intensity, swing=swing,
            fill_every=4 if scaled_intensity > 0.6 else 8,
            bar_intensities=bar_energies,
            bar_context=lead_context,
            bar_feel=bar_feel,
            beats_per_bar=beats_per_bar,
        )

    _offset_notes(melody_notes, int(tick_offset))
    _offset_notes(bass_notes, int(tick_offset))
    _offset_notes(drum_notes, int(tick_offset))
    _offset_notes(piano_notes, int(tick_offset))

    return {
        "melody": melody_notes,
        "piano": piano_notes,
        "bass": bass_notes,
        "drums": drum_notes,
    }


def _generate_trading_section(
    section: ArrangementSection,
    chords: List[ChordEvent],
    swing: bool,
    bass_style: str = "walking",
    drum_style: str = "swing",
    beats_per_bar: int = 4,
) -> Dict[str, List[NoteEvent]]:
    """Generate trading fours: melody plays 4 bars, drums fill 4 bars, repeat.

    Bass and piano continue throughout. Drums play fills during their 4-bar
    response and normal pattern during the melody's 4-bar phrases.
    """
    section_chords = _chords_for_section(chords, section)
    beats = section.total_beats
    tick_offset = section.start_beat * TICKS_PER_QUARTER

    melody_notes = generate_trading_fours(
        section_chords, float(beats), intensity=section.intensity,
        beats_per_bar=beats_per_bar,
    )

    bar_energies = _compute_reactive_energy(melody_notes, beats, section.intensity,
                                            beats_per_bar)
    bar_feel = _compute_rhythmic_feel(bar_energies, melody_notes, beats, beats_per_bar)

    piano_notes = generate_comping(
        section_chords, beats, intensity=section.intensity, swing=swing,
        bass_sync_ticks=[], context="solo",
        bar_intensities=bar_energies,
        bar_feel=bar_feel,
        beats_per_bar=beats_per_bar,
    )

    lead_context = _extract_bar_context(melody_notes + piano_notes, beats,
                                        chords=section_chords,
                                        beats_per_bar=beats_per_bar)

    if bass_style == "modal":
        bass_notes = generate_modal_bass(section_chords, beats, swing=swing,
                                         beats_per_bar=beats_per_bar)
    else:
        bass_notes = generate_walking_bass(section_chords, beats, swing=swing,
                                           intensity=section.intensity,
                                           bar_intensities=bar_energies,
                                           bar_context=lead_context,
                                           bar_feel=bar_feel,
                                           beats_per_bar=beats_per_bar)

    # Drums: fills during the response bars (bars 5-8 of each 8-bar phrase)
    # and normal pattern during melody bars (bars 1-4)
    drum_notes = _generate_trading_drums(beats, section.intensity, swing, drum_style,
                                         beats_per_bar)

    _offset_notes(melody_notes, int(tick_offset))
    _offset_notes(bass_notes, int(tick_offset))
    _offset_notes(drum_notes, int(tick_offset))
    _offset_notes(piano_notes, int(tick_offset))

    return {
        "melody": melody_notes,
        "piano": piano_notes,
        "bass": bass_notes,
        "drums": drum_notes,
    }


def _generate_trading_drums(
    total_beats: int,
    intensity: float,
    swing: bool,
    drum_style: str = "swing",
    beats_per_bar: int = 4,
) -> List[NoteEvent]:
    """Generate drums for trading fours: comping during melody, fills during response.

    Bars 1-4 (of each 8-bar phrase): normal swing pattern (supporting melody).
    Bars 5-8: drum solo (high-intensity fills with ride pattern).
    """
    tpb = ticks_per_bar(beats_per_bar)
    total_bars = total_beats // beats_per_bar
    notes: List[NoteEvent] = []
    bar_idx = 0

    def _gen(beats, intens, fill_every=0):
        if drum_style == "brushes":
            return generate_brushes_drums(beats, intensity=intens, swing=swing,
                                          fill_every=fill_every,
                                          beats_per_bar=beats_per_bar)
        elif drum_style == "modal":
            return generate_modal_drums(beats, intensity=intens, swing=swing,
                                        fill_every=fill_every,
                                        beats_per_bar=beats_per_bar)
        else:
            return generate_drums(beats, intensity=intens, swing=swing,
                                  fill_every=fill_every,
                                  beats_per_bar=beats_per_bar)

    while bar_idx < total_bars:
        # 4-bar phrase: melody plays
        for i in range(min(4, total_bars - bar_idx)):
            bar_beats = beats_per_bar
            bar_notes = _gen(bar_beats, intensity * 0.7)
            _offset_notes(bar_notes, (bar_idx + i) * tpb)
            notes.extend(bar_notes)
        bar_idx += 4

        if bar_idx >= total_bars:
            break

        # 4-bar phrase: drums respond (solo fills)
        for i in range(min(4, total_bars - bar_idx)):
            bar_beats = beats_per_bar
            bar_notes = _gen(bar_beats, min(1.0, intensity + 0.3), fill_every=2)
            _offset_notes(bar_notes, (bar_idx + i) * tpb)
            notes.extend(bar_notes)
        bar_idx += 4

    return notes


def _generate_drum_solo_section(
    section: ArrangementSection,
    chords: List[ChordEvent],
    swing: bool,
    bass_style: str = "walking",
    beats_per_bar: int = 4,
) -> Dict[str, List[NoteEvent]]:
    """Generate drum solo section: drums feature, walking bass continues, piano drops out."""
    section_chords = _chords_for_section(chords, section)
    beats = section.total_beats
    tick_offset = section.start_beat * TICKS_PER_QUARTER

    # Drums: full solo
    drum_notes = generate_drum_solo(beats, intensity=section.intensity, swing=swing,
                                    beats_per_bar=beats_per_bar)

    # Bass: keep walking to hold the form
    if bass_style == "modal":
        bass_notes = generate_modal_bass(section_chords, beats, swing=swing,
                                         beats_per_bar=beats_per_bar)
    else:
        bass_notes = generate_walking_bass(section_chords, beats, swing=swing,
                                           intensity=0.5,
                                           beats_per_bar=beats_per_bar)

    _offset_notes(drum_notes, int(tick_offset))
    _offset_notes(bass_notes, int(tick_offset))

    return {
        "melody": [],
        "piano": [],
        "bass": bass_notes,
        "drums": drum_notes,
    }


def _generate_bass_solo_section(
    section: ArrangementSection,
    chords: List[ChordEvent],
    swing: bool,
    drum_style: str = "swing",
    beats_per_bar: int = 4,
) -> Dict[str, List[NoteEvent]]:
    """Generate bass solo section: bass features in upper register, light drums, sparse piano."""
    section_chords = _chords_for_section(chords, section)
    beats = section.total_beats
    tick_offset = section.start_beat * TICKS_PER_QUARTER

    # Bass: melodic solo in upper register
    bass_notes = generate_bass_solo(section_chords, beats, swing=swing,
                                    intensity=section.intensity,
                                    beats_per_bar=beats_per_bar)

    # Drums: very light time-keeping
    if drum_style == "brushes":
        drum_notes = generate_brushes_drums(beats, intensity=0.3, swing=swing, fill_every=0,
                                            beats_per_bar=beats_per_bar)
    else:
        drum_notes = generate_drums(beats, intensity=0.3, swing=swing, fill_every=0,
                                    beats_per_bar=beats_per_bar)

    # Piano: very sparse comping (low intensity for space)
    bass_sync = _extract_downbeat_ticks(bass_notes, beats_per_bar)
    piano_notes = generate_comping(
        section_chords, beats, intensity=0.25, swing=swing,
        bass_sync_ticks=bass_sync, context="head",
        beats_per_bar=beats_per_bar,
    )

    _offset_notes(bass_notes, int(tick_offset))
    _offset_notes(drum_notes, int(tick_offset))
    _offset_notes(piano_notes, int(tick_offset))

    return {
        "melody": [],
        "piano": piano_notes,
        "bass": bass_notes,
        "drums": drum_notes,
    }


def _generate_coda(
    section: ArrangementSection,
    chords: List[ChordEvent],
    swing: bool,
    drum_style: str = "swing",
    beats_per_bar: int = 4,
) -> Dict[str, List[NoteEvent]]:
    """Generate coda with jazz ending: wind-down, ritardando, and fermata.

    Structure (4 bars):
      Bars 1-(N-2): Wind-down with ritardando (lengthening durations, fading)
      Bar N-1:      Approach — bass walks to root, sparse piano shell, soft drums
      Bar N:        Fermata — all instruments land on beat 1, rich final chord held long
    """
    section_chords = _chords_for_section(chords, section)
    beats = section.total_beats

    melody_notes: List[NoteEvent] = []
    bass_notes: List[NoteEvent] = []
    drum_notes: List[NoteEvent] = []
    piano_notes: List[NoteEvent] = []

    coda_bars = beats // beats_per_bar
    if coda_bars <= 0 or not section_chords:
        return {"melody": [], "piano": [], "bass": [], "drums": []}

    final_chord = section_chords[-1]
    root_pc = final_chord.root_pc
    quality = final_chord.quality
    intervals = CHORD_TONES.get(quality, (0, 4, 7, 10))
    third_interval = intervals[1] if len(intervals) > 1 else 4
    fifth_interval = intervals[2] if len(intervals) > 2 else 7
    seventh_interval = intervals[3] if len(intervals) > 3 else intervals[-1]

    # Bass root in low register
    bass_root = 36 + root_pc
    if bass_root < 33:
        bass_root += 12

    # ---- Wind-down bars (all except last 2) ----
    wind_down_bars = max(0, coda_bars - 2)
    for bar_idx in range(wind_down_bars):
        intensity = max(0.10, 0.35 - bar_idx * 0.10)
        rit_factor = 1.0 + bar_idx * 0.15  # ritardando: notes lengthen
        vel_drop = bar_idx * 8

        seg_start = section.start_beat + bar_idx * beats_per_bar
        seg_chords = _chords_for_beat_range(chords, seg_start, seg_start + beats_per_bar)
        if not seg_chords:
            seg_chords = [final_chord]
        seg_tick = int(seg_start * TICKS_PER_QUARTER)

        # Two-feel bass with ritardando
        b = generate_two_feel_bass(seg_chords, beats_per_bar, swing=swing,
                                   beats_per_bar=beats_per_bar)
        _offset_notes(b, seg_tick)
        for n in b:
            n.duration_ticks = int(n.duration_ticks * rit_factor)
            n.velocity = max(20, n.velocity - vel_drop)
        bass_notes.extend(b)

        # Sparse piano comping
        p = generate_comping(seg_chords, beats_per_bar, intensity=intensity, swing=swing,
                             context="head", beats_per_bar=beats_per_bar)
        _offset_notes(p, seg_tick)
        for n in p:
            n.duration_ticks = int(n.duration_ticks * rit_factor)
            n.velocity = max(20, n.velocity - vel_drop // 2)
        piano_notes.extend(p)

        # Light drums
        if drum_style == "brushes":
            d = generate_brushes_drums(beats_per_bar, intensity=intensity, swing=swing,
                                       fill_every=0, beats_per_bar=beats_per_bar)
        elif drum_style == "modal":
            d = generate_modal_drums(beats_per_bar, intensity=intensity, swing=swing,
                                     fill_every=0, beats_per_bar=beats_per_bar)
        else:
            d = generate_drums(beats_per_bar, intensity=intensity, swing=swing, fill_every=0,
                               beats_per_bar=beats_per_bar)
        _offset_notes(d, seg_tick)
        for n in d:
            n.velocity = max(15, n.velocity - vel_drop)
        drum_notes.extend(d)

    # ---- Approach bar (penultimate) ----
    if coda_bars >= 2:
        approach_bar_idx = coda_bars - 2
        approach_start = section.start_beat + approach_bar_idx * beats_per_bar
        approach_tick = int(approach_start * TICKS_PER_QUARTER)

        # Bass: 5th on beat 1 (long half note) → chromatic approach from below
        fifth_midi = bass_root + ((fifth_interval) % 12)
        if fifth_midi > bass_root + 12:
            fifth_midi -= 12
        bass_notes.append(NoteEvent(
            pitch=fifth_midi,
            start_tick=approach_tick,
            duration_ticks=int(2.5 * TICKS_PER_QUARTER),
            velocity=46,
            channel=2,
        ))
        bass_notes.append(NoteEvent(
            pitch=bass_root - 1,  # chromatic approach from one semitone below
            start_tick=approach_tick + int(2.5 * TICKS_PER_QUARTER),
            duration_ticks=int(1.5 * TICKS_PER_QUARTER),
            velocity=40,
            channel=2,
        ))

        # Piano: sparse shell voicing (root + 3rd + 7th), held through bar
        shell_voicing = []
        for interval in [0, third_interval, seventh_interval]:
            pc = (root_pc + interval) % 12
            midi_note = 48 + pc
            if midi_note > 64:
                midi_note -= 12
            shell_voicing.append(midi_note)
        for pitch in sorted(set(shell_voicing)):
            piano_notes.append(NoteEvent(
                pitch=pitch,
                start_tick=approach_tick,
                duration_ticks=int(3.5 * TICKS_PER_QUARTER),
                velocity=34,
                channel=1,
            ))

        # Drums: barely there — single soft ride tap
        drum_notes.append(NoteEvent(
            pitch=51, start_tick=approach_tick,
            duration_ticks=TICKS_PER_QUARTER,
            velocity=22 if drum_style == "brushes" else 26,
            channel=9,
        ))

    # ---- Final bar: fermata (the "button") ----
    final_bar_idx = coda_bars - 1
    final_start = section.start_beat + final_bar_idx * beats_per_bar
    final_tick = int(final_start * TICKS_PER_QUARTER)
    fermata_ticks = 6 * TICKS_PER_QUARTER  # hold 1.5 bars past the downbeat

    # Right hand (melody): resolve to the 3rd for warmth
    melody_pc = (root_pc + third_interval) % 12
    melody_pitch = 72 + melody_pc
    if melody_pitch > 84:
        melody_pitch -= 12
    melody_notes.append(NoteEvent(
        pitch=melody_pitch,
        start_tick=final_tick,
        duration_ticks=fermata_ticks,
        velocity=60,
        channel=0,
    ))

    # Left hand (piano): rich open voicing — root low, 7th mid, 3rd+5th spread
    low_root = 48 + root_pc
    if low_root > 55:
        low_root -= 12
    final_voicing = [low_root]
    # 7th above root
    sev_midi = low_root + seventh_interval
    if sev_midi > low_root + 12:
        sev_midi -= 12
    final_voicing.append(sev_midi)
    # 3rd spread an octave above root for openness
    thi_midi = low_root + third_interval + 12
    if thi_midi > 72:
        thi_midi -= 12
    final_voicing.append(thi_midi)
    # 5th for fullness
    fif_midi = low_root + fifth_interval
    if fif_midi < thi_midi:
        fif_midi += 12
    if 48 <= fif_midi <= 72:
        final_voicing.append(fif_midi)

    for pitch in sorted(set(final_voicing)):
        piano_notes.append(NoteEvent(
            pitch=pitch,
            start_tick=final_tick,
            duration_ticks=fermata_ticks,
            velocity=44,
            channel=1,
        ))

    # Bass: root, held with fermata
    bass_notes.append(NoteEvent(
        pitch=bass_root,
        start_tick=final_tick,
        duration_ticks=fermata_ticks,
        velocity=50,
        channel=2,
    ))

    # Drums: single cymbal + kick on beat 1
    if drum_style == "brushes":
        drum_notes.append(NoteEvent(
            pitch=51, start_tick=final_tick,
            duration_ticks=2 * TICKS_PER_QUARTER, velocity=32, channel=9))
    else:
        drum_notes.append(NoteEvent(
            pitch=49, start_tick=final_tick,
            duration_ticks=2 * TICKS_PER_QUARTER, velocity=50, channel=9))
    drum_notes.append(NoteEvent(
        pitch=36, start_tick=final_tick,
        duration_ticks=TICKS_PER_QUARTER, velocity=36, channel=9))

    return {
        "melody": melody_notes,
        "piano": piano_notes,
        "bass": bass_notes,
        "drums": drum_notes,
    }


# ---------------------------------------------------------------------------
# Chord slicing helpers
# ---------------------------------------------------------------------------


def _chords_for_section(
    chords: List[ChordEvent],
    section: ArrangementSection,
) -> List[ChordEvent]:
    """Extract and re-offset chords that overlap with a given section.

    Returns chords with start_beat relative to the section start (starting at 0).
    """
    return _chords_for_beat_range(chords, section.start_beat, section.end_beat)


def _chords_for_beat_range(
    chords: List[ChordEvent],
    start_beat: float,
    end_beat: float,
) -> List[ChordEvent]:
    """Extract chords overlapping [start_beat, end_beat) and re-offset to start at 0.

    If a chord starts before the range but extends into it, it is clipped.
    """
    result: List[ChordEvent] = []

    for c in chords:
        c_start = c.start_beat
        c_end = c.start_beat + c.duration_beats

        # Skip chords entirely outside the range
        if c_end <= start_beat or c_start >= end_beat:
            continue

        # Clip to range
        clipped_start = max(c_start, start_beat)
        clipped_end = min(c_end, end_beat)
        clipped_duration = clipped_end - clipped_start

        if clipped_duration <= 0:
            continue

        # Re-offset to start at 0
        result.append(ChordEvent(
            root_pc=c.root_pc,
            quality=c.quality,
            start_beat=clipped_start - start_beat,
            duration_beats=clipped_duration,
            key_center_pc=c.key_center_pc,
            function=c.function,
            form_section=c.form_section,
        ))

    return result


# ---------------------------------------------------------------------------
# Bass-piano sync helper
# ---------------------------------------------------------------------------


def _compute_reactive_energy(melody_notes: List[NoteEvent],
                             total_beats: int,
                             base_intensity: float,
                             beats_per_bar: int = 4) -> List[float]:
    """Derive per-bar energy values from what the melody just played.

    Creates emergent dynamics: dense bars cause a dip, silence causes a build.
    Returns one float per bar, clamped to [base*0.5, min(1.0, base*1.5)].
    """
    tpb = ticks_per_bar(beats_per_bar)
    total_bars = total_beats // beats_per_bar
    if total_bars <= 0:
        return []

    # Compute per-bar note counts and avg velocity
    bar_counts = [0] * total_bars
    bar_vel_sum = [0.0] * total_bars
    bar_vel_max = [0] * total_bars
    for n in melody_notes:
        bar_idx = n.start_tick // tpb
        if 0 <= bar_idx < total_bars:
            bar_counts[bar_idx] += 1
            bar_vel_sum[bar_idx] += n.velocity
            bar_vel_max[bar_idx] = max(bar_vel_max[bar_idx], n.velocity)

    floor = max(0.15, base_intensity * 0.5)
    ceil = min(1.0, base_intensity * 1.5)
    energy = base_intensity
    result = []

    for i in range(total_bars):
        if i > 0:
            prev_count = bar_counts[i - 1]
            if prev_count > 5:
                energy -= 0.08   # breathe after dense bar
            elif prev_count < 1:
                energy += 0.05   # tension accumulates in silence
            if bar_vel_max[i - 1] > 100:
                energy -= 0.03   # recede after loud peak

        # Momentum: slow return toward baseline
        energy = 0.85 * energy + 0.15 * base_intensity
        energy = max(floor, min(ceil, energy))
        result.append(energy)

    return result


def _extract_bar_context(notes: List[NoteEvent],
                         total_beats: int,
                         chords: list = None,
                         beats_per_bar: int = 4) -> List[BarContext]:
    """Extract per-bar statistics from generated notes for rhythm section reactivity.

    If *chords* (list of ChordEvent) is provided, also computes harmonic fields:
    chord_count, form_section, and is_key_change per bar.
    """
    tpb = ticks_per_bar(beats_per_bar)
    total_bars = total_beats // beats_per_bar
    if total_bars <= 0:
        return []

    bar_notes: List[List[NoteEvent]] = [[] for _ in range(total_bars)]
    for n in notes:
        idx = n.start_tick // tpb
        if 0 <= idx < total_bars:
            bar_notes[idx].append(n)

    # Pre-compute per-bar chord info if chords provided
    bar_chord_counts = [0] * total_bars
    bar_form_sections = [""] * total_bars
    bar_key_change = [False] * total_bars
    if chords:
        for bar_idx in range(total_bars):
            bar_start = bar_idx * float(beats_per_bar)
            bar_end = bar_start + float(beats_per_bar)
            keys_in_bar = set()
            first_section = ""
            count = 0
            for c in chords:
                # Chord overlaps this bar if it starts before bar_end and ends after bar_start
                if c.start_beat < bar_end and c.end_beat > bar_start:
                    count += 1
                    if not first_section and c.form_section:
                        first_section = c.form_section
                    keys_in_bar.add(c.key_center_pc)
            bar_chord_counts[bar_idx] = max(count, 1)
            bar_form_sections[bar_idx] = first_section
            bar_key_change[bar_idx] = len(keys_in_bar) > 1

    # Find max note count for normalization
    max_count = max((len(bn) for bn in bar_notes), default=1) or 1

    result = []
    for i, bn in enumerate(bar_notes):
        count = len(bn)
        density = count / max_count
        avg_vel = sum(n.velocity for n in bn) / count if count else 70.0
        avg_reg = sum(n.pitch for n in bn) / count if count else 65.0
        result.append(BarContext(
            density=density,
            avg_velocity=avg_vel,
            avg_register=avg_reg,
            has_silence=(count < 1),
            chord_count=bar_chord_counts[i],
            form_section=bar_form_sections[i],
            is_key_change=bar_key_change[i],
        ))
    return result


def _compute_rhythmic_feel(bar_energies: List[float],
                           melody_notes: List[NoteEvent],
                           total_beats: int,
                           beats_per_bar: int = 4) -> List[BarFeel]:
    """Derive per-bar timing feel from musical context.

    Building phrases push ahead; resolving phrases lay back.
    High energy = tight timing; low energy = loose/rubato.
    """
    tpb = ticks_per_bar(beats_per_bar)
    total_bars = total_beats // beats_per_bar
    if total_bars <= 0:
        return []

    # Compute per-bar note counts for density trend detection
    bar_counts = [0] * total_bars
    for n in melody_notes:
        idx = n.start_tick // tpb
        if 0 <= idx < total_bars:
            bar_counts[idx] += 1

    result = []
    for i in range(total_bars):
        energy = bar_energies[i] if i < len(bar_energies) else 0.5

        # Offset bias: density trend determines push/pull
        if i > 0:
            delta = bar_counts[i] - bar_counts[i - 1]
            if delta > 2:
                offset_bias = min(0.6, 0.3 + delta * 0.05)   # building → push
            elif delta < -2:
                offset_bias = max(-0.5, -0.3 + delta * 0.05)  # resolving → lay back
            else:
                offset_bias = 0.0
        else:
            offset_bias = 0.0

        # Timing spread: high energy = tight, low = loose
        timing_spread = max(0.5, min(1.5, 1.4 - energy * 0.6))

        # Swing depth: higher energy = deeper swing
        swing_depth = max(0.8, min(1.2, 0.9 + energy * 0.2))

        result.append(BarFeel(
            offset_bias=offset_bias,
            timing_spread=timing_spread,
            swing_depth=swing_depth,
        ))
    return result


def _extract_downbeat_ticks(bass_notes: List[NoteEvent],
                            beats_per_bar: int = 4) -> List[int]:
    """Return tick positions of bass notes on beats 1 or mid-bar.

    Used to synchronize piano comping with walking bass for the
    Bill Evans-style rhythmic lock.
    """
    tpb = ticks_per_bar(beats_per_bar)
    mid_beat = beats_per_bar // 2
    sync_ticks = []
    for note in bass_notes:
        pos_in_bar = note.start_tick % tpb
        # Beat 1: near tick 0 of bar; Mid-bar: near mid_beat*TICKS_PER_QUARTER
        if pos_in_bar < 30 or abs(pos_in_bar - mid_beat * TICKS_PER_QUARTER) < 30:
            sync_ticks.append(note.start_tick)
    return sorted(sync_ticks)


# ---------------------------------------------------------------------------
# Stereo pan movement
# ---------------------------------------------------------------------------

# Default pan positions (must match writer.py DEFAULT_PAN)
_DEFAULT_PAN = {"melody": 45, "piano": 82, "bass": 58, "drums": 70}

# Alternate pan targets for dramatic shifts
_PAN_SHIFTS = {
    # section_name → {track: shifted_pan}  (only tracks that move)
    "solo":       {"melody": 85, "piano": 38},   # Solo: melody slides right, comping left
    "trading":    {"melody": 64, "piano": 64},    # Trading: both converge to center
    "drum_solo":  {"bass": 40, "drums": 64},      # Drum solo: drums to center, bass left
    "bass_solo":  {"bass": 64, "melody": 35},     # Bass solo: bass to center, melody far left
}


def _generate_pan_shifts(
    arrangement: List[ArrangementSection],
    cc_events: Dict[str, list],
) -> None:
    """Insert CC10 pan events at section boundaries for stereo movement.

    At the start of sections with alternate pan targets, smoothly shift
    the relevant tracks. At head/coda sections, restore defaults.
    Modifies cc_events in place.
    """
    for section in arrangement:
        tick = int(section.start_beat * TICKS_PER_QUARTER)

        # Determine which pan map to use for this section
        if section.is_solo:
            pan_map = _PAN_SHIFTS["solo"]
        elif section.is_trading:
            pan_map = _PAN_SHIFTS["trading"]
        elif section.is_drum_solo:
            pan_map = _PAN_SHIFTS["drum_solo"]
        elif section.is_bass_solo:
            pan_map = _PAN_SHIFTS["bass_solo"]
        else:
            # Head, intro, coda: restore defaults
            pan_map = _DEFAULT_PAN

        for track_name, pan_val in pan_map.items():
            ch_map = {"melody": 0, "piano": 1, "bass": 2, "drums": 9}
            ch = ch_map.get(track_name, 0)
            cc_events.setdefault(track_name, []).append(
                CCEvent(cc_number=10, value=pan_val, start_tick=tick, channel=ch)
            )


# ---------------------------------------------------------------------------
# Master orchestrator
# ---------------------------------------------------------------------------


def generate_arrangement(
    arrangement: List[ArrangementSection],
    chords: List[ChordEvent],
    form_bars: int,
    key_pc: int,
    tension_curve: str = "arc",
    coltrane: bool = False,
    swing: bool = True,
    seed: Optional[int] = None,
    humanize: bool = True,
    tempo: int = 140,
    bass_style: str = "walking",
    drum_style: str = "swing",
    reharmonize_density: str = "off",
    beats_per_bar: int = 4,
) -> Dict[str, List[NoteEvent]]:
    """Generate the full arrangement by orchestrating all instrument generators.

    Args:
        arrangement: List of ArrangementSection objects from build_arrangement.
        chords: Full chord progression (absolute beat positions).
        form_bars: Number of bars in one chorus of the form.
        key_pc: Key pitch class (0-11).
        tension_curve: Tension curve type for solos.
        coltrane: Enable Coltrane features.
        swing: Apply swing feel.
        seed: Random seed for reproducibility.
        humanize: Apply post-processing humanization.
        tempo: BPM (used for humanization timing).
        bass_style: 'walking' or 'modal'.
        drum_style: 'swing' or 'modal'.
        reharmonize_density: 'off', 'light', 'medium', 'heavy'.

    Returns:
        Dict with keys 'melody', 'piano', 'bass', 'drums', each containing
        a list of NoteEvent objects with absolute tick positions.
    """
    if seed is not None:
        random.seed(seed)

    tracks: Dict[str, List[NoteEvent]] = {
        "melody": [],
        "piano": [],
        "bass": [],
        "drums": [],
    }

    # Count solo sections for multi-chorus arc
    solo_sections = [s for s in arrangement if s.is_solo]
    num_solos = len(solo_sections)

    for section in arrangement:
        # Reharmonize solo section chords if enabled
        section_chords = chords
        if section.is_solo and reharmonize_density != "off":
            section_chord_slice = _chords_for_section(chords, section)
            reharmed = reharmonize(section_chord_slice, reharmonize_density)
            # Re-offset back to absolute beat positions
            tick_offset_beats = section.start_beat
            for c in reharmed:
                c.start_beat += tick_offset_beats
            # Build a modified chord list: replace chords in this section's range
            section_chords = [c for c in chords
                              if c.start_beat < section.start_beat
                              or c.start_beat >= section.end_beat]
            section_chords.extend(reharmed)
            section_chords.sort(key=lambda c: c.start_beat)

        # Multi-chorus arc: scale solo intensity by position
        if section.is_solo and num_solos >= 3:
            solo_idx = solo_sections.index(section)
            if solo_idx == 0:
                density_scale = 0.7  # Exploratory
            elif solo_idx == num_solos - 1:
                density_scale = 1.3  # Climactic
            else:
                density_scale = 1.0  # Normal
        else:
            density_scale = 1.0

        if section.name == "intro":
            section_notes = _generate_intro(section, section_chords, swing,
                                            bass_style, drum_style, beats_per_bar)
        elif section.name in ("head_in", "head_out"):
            section_notes = _generate_head(section, section_chords, swing,
                                           coltrane, bass_style, drum_style, beats_per_bar)
        elif section.is_solo:
            section_notes = _generate_solo_section(
                section, section_chords, tension_curve, swing, coltrane, seed,
                density_scale, bass_style, drum_style, beats_per_bar,
            )
        elif section.is_drum_solo:
            section_notes = _generate_drum_solo_section(section, section_chords, swing,
                                                        bass_style, beats_per_bar)
        elif section.is_bass_solo:
            section_notes = _generate_bass_solo_section(section, section_chords, swing,
                                                        drum_style, beats_per_bar)
        elif section.is_trading:
            section_notes = _generate_trading_section(section, section_chords, swing,
                                                      bass_style, drum_style, beats_per_bar)
        elif section.name == "coda":
            section_notes = _generate_coda(section, section_chords, swing, drum_style,
                                           beats_per_bar)
        else:
            section_notes = _generate_head(section, section_chords, swing,
                                           coltrane, bass_style, drum_style, beats_per_bar)

        # Clip notes to section boundaries — prevents bleed into next section
        sec_end_tick = int(section.end_beat * TICKS_PER_QUARTER)
        for track_name in section_notes:
            section_notes[track_name] = [
                n for n in section_notes[track_name] if n.start_tick < sec_end_tick
            ]

        _merge_tracks(tracks, section_notes)

    # Two-handed piano: melody = right hand (ch 0), piano = left hand (ch 1)
    cc_events: Dict[str, list] = {}

    # Sustain pedal for left hand (comping voicings) on channel 1
    cc_events["piano"] = generate_sustain_pedal(tracks["piano"], channel=1,
                                                beats_per_bar=beats_per_bar)

    # Set channels explicitly
    for note in tracks["melody"]:
        note.channel = 0
    for note in tracks["piano"]:
        note.channel = 1

    # Generate expression data (pitch bends + additional CCs)
    bass_bends, bass_expr_ccs = generate_bass_expression(tracks["bass"], chords, channel=2,
                                                         beats_per_bar=beats_per_bar)
    melody_bends, melody_expr_ccs = generate_melody_expression(tracks["melody"], chords, channel=0,
                                                               beats_per_bar=beats_per_bar)
    hihat_ccs = generate_hihat_expression(tracks["drums"], channel=9)

    pitch_bend_events: Dict[str, list] = {}
    if bass_bends:
        pitch_bend_events["bass"] = bass_bends
    if melody_bends:
        pitch_bend_events["melody"] = melody_bends

    # Merge expression CCs into cc_events
    if bass_expr_ccs:
        cc_events.setdefault("bass", []).extend(bass_expr_ccs)
    if melody_expr_ccs:
        cc_events.setdefault("melody", []).extend(melody_expr_ccs)
    if hihat_ccs:
        cc_events.setdefault("drums", []).extend(hihat_ccs)

    # Stereo pan shifts at section boundaries for dramatic movement
    _generate_pan_shifts(arrangement, cc_events)

    # Post-processing: humanization — per-section intensity for musical feel
    if humanize:
        for track_name in tracks:
            instrument = track_name  # "melody", "piano", "bass", "drums"
            humanized_notes: List[NoteEvent] = []

            for section in arrangement:
                sec_start = int(section.start_beat * TICKS_PER_QUARTER)
                sec_end = int(section.end_beat * TICKS_PER_QUARTER)
                section_notes = [
                    n for n in tracks[track_name]
                    if sec_start <= n.start_tick < sec_end
                ]
                if section_notes:
                    humanized_notes.extend(
                        humanize_track(section_notes, instrument,
                                       tempo=tempo, intensity=section.intensity)
                    )

            # Safety: any notes outside defined sections
            max_tick = max(
                (int(s.end_beat * TICKS_PER_QUARTER) for s in arrangement),
                default=0,
            )
            orphans = [n for n in tracks[track_name] if n.start_tick >= max_tick]
            if orphans:
                humanized_notes.extend(
                    humanize_track(orphans, instrument, tempo=tempo, intensity=0.5)
                )

            tracks[track_name] = humanized_notes

    # Sort all tracks by start_tick for clean output
    for track_name in tracks:
        tracks[track_name].sort(key=lambda n: n.start_tick)

    return tracks, cc_events, pitch_bend_events
