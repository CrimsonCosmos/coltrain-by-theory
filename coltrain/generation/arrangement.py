"""Arrangement engine for Coltrain.

Orchestrates the full jazz arrangement: intro, head in, solos, trading fours,
head out, and coda. Builds chord progressions from form templates and
coordinates all instrument generators into a unified multi-track output.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from coltrain.generation import NoteEvent, TICKS_PER_QUARTER, TICKS_PER_BAR
from coltrain.theory.chord import ChordEvent
from coltrain.theory.pitch import NOTE_TO_PC

from coltrain.generation.melody import generate_head_melody, generate_solo, generate_trading_fours
from coltrain.generation.bass import generate_walking_bass, generate_two_feel_bass, generate_modal_bass
from coltrain.generation.drums import generate_drums, generate_modal_drums
from coltrain.generation.piano import generate_comping
from coltrain.generation.humanize import humanize_track
from coltrain.generation.reharmonize import reharmonize

# ---------------------------------------------------------------------------
# ArrangementSection data type
# ---------------------------------------------------------------------------


@dataclass
class ArrangementSection:
    """A section of the jazz arrangement."""
    name: str           # 'intro', 'head_in', 'solo', 'trading', 'head_out', 'coda'
    start_beat: int
    end_beat: int
    intensity: float    # 0.0-1.0
    is_melody: bool = False
    is_solo: bool = False
    is_trading: bool = False

    @property
    def total_beats(self) -> int:
        return self.end_beat - self.start_beat

    @property
    def total_bars(self) -> int:
        return self.total_beats // 4

    def __repr__(self) -> str:
        flags = []
        if self.is_melody:
            flags.append("melody")
        if self.is_solo:
            flags.append("solo")
        if self.is_trading:
            flags.append("trading")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        return (
            f"  {self.name:12s}  bars {self.start_beat // 4 + 1:3d}-{self.end_beat // 4:3d}"
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

    chords: List[ChordEvent] = []
    beat = start_beat

    for chorus_idx in range(num_choruses):
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
) -> List[ArrangementSection]:
    """Build the arrangement structure (section layout) for a jazz tune.

    Structure varies by number of choruses:
    - 1 chorus:  intro(4) + head(1 chorus) + coda(4)
    - 2 choruses: intro(4) + head_in(1) + solo(1) + coda(4)
    - 3+: intro(4) + head_in(1) + solos + trading(1) + head_out(1) + coda(4)

    Args:
        form_name: Form template name.
        num_choruses: Number of solo choruses (total arrangement is longer).
        bars_per_chorus: Number of bars in one chorus of the form.

    Returns:
        List of ArrangementSection objects.
    """
    sections: List[ArrangementSection] = []
    current_beat = 0
    beats_per_chorus = bars_per_chorus * 4
    intro_bars = 4
    coda_bars = 4

    # Intro: 4 bars
    intro_end = current_beat + intro_bars * 4
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
    coda_end = current_beat + coda_bars * 4
    sections.append(ArrangementSection(
        name="coda",
        start_beat=current_beat,
        end_beat=coda_end,
        intensity=0.3,
        is_melody=True,
    ))

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
) -> Dict[str, List[NoteEvent]]:
    """Generate intro section: sparse piano voicings, two-feel bass, light drums."""
    section_chords = _chords_for_section(chords, section)
    beats = section.total_beats
    tick_offset = section.start_beat * TICKS_PER_QUARTER

    if bass_style == "modal":
        bass_notes = generate_modal_bass(section_chords, beats, swing=swing)
    else:
        bass_notes = generate_two_feel_bass(section_chords, beats, swing=swing)
    if drum_style == "modal":
        drum_notes = generate_modal_drums(beats, intensity=0.2, swing=swing, fill_every=0)
    else:
        drum_notes = generate_drums(beats, intensity=0.2, swing=swing, fill_every=0)
    piano_notes = generate_comping(section_chords, beats, intensity=0.2, swing=swing)

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
) -> Dict[str, List[NoteEvent]]:
    """Generate head section: melody + full rhythm section."""
    section_chords = _chords_for_section(chords, section)
    beats = section.total_beats
    tick_offset = section.start_beat * TICKS_PER_QUARTER

    melody_notes = generate_head_melody(section_chords, float(beats), swing=swing)
    if bass_style == "modal":
        bass_notes = generate_modal_bass(section_chords, beats, swing=swing)
    else:
        bass_notes = generate_walking_bass(section_chords, beats, swing=swing,
                                           intensity=section.intensity)
    if drum_style == "modal":
        drum_notes = generate_modal_drums(beats, intensity=section.intensity, swing=swing, fill_every=8)
    else:
        drum_notes = generate_drums(beats, intensity=section.intensity, swing=swing, fill_every=8)
    piano_notes = generate_comping(section_chords, beats, intensity=section.intensity,
                                   swing=swing, coltrane=coltrane)

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
    )
    if bass_style == "modal":
        bass_notes = generate_modal_bass(section_chords, beats, swing=swing)
    else:
        bass_notes = generate_walking_bass(section_chords, beats, swing=swing,
                                           intensity=scaled_intensity)
    if drum_style == "modal":
        drum_notes = generate_modal_drums(
            beats, intensity=scaled_intensity, swing=swing,
            fill_every=4 if scaled_intensity > 0.6 else 8,
        )
    else:
        drum_notes = generate_drums(
            beats, intensity=scaled_intensity, swing=swing,
            fill_every=4 if scaled_intensity > 0.6 else 8,
        )
    piano_notes = generate_comping(
        section_chords, beats, intensity=scaled_intensity, swing=swing,
        coltrane=coltrane,
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
    )

    if bass_style == "modal":
        bass_notes = generate_modal_bass(section_chords, beats, swing=swing)
    else:
        bass_notes = generate_walking_bass(section_chords, beats, swing=swing,
                                           intensity=section.intensity)
    piano_notes = generate_comping(
        section_chords, beats, intensity=section.intensity, swing=swing,
    )

    # Drums: fills during the response bars (bars 5-8 of each 8-bar phrase)
    # and normal pattern during melody bars (bars 1-4)
    drum_notes = _generate_trading_drums(beats, section.intensity, swing)

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
) -> List[NoteEvent]:
    """Generate drums for trading fours: comping during melody, fills during response.

    Bars 1-4 (of each 8-bar phrase): normal swing pattern (supporting melody).
    Bars 5-8: drum solo (high-intensity fills with ride pattern).
    """
    total_bars = total_beats // 4
    notes: List[NoteEvent] = []
    bar_idx = 0

    while bar_idx < total_bars:
        # 4-bar phrase: melody plays
        for i in range(min(4, total_bars - bar_idx)):
            bar_beats = 4
            # Normal comping pattern behind melody
            bar_notes = generate_drums(
                bar_beats, intensity=intensity * 0.7, swing=swing, fill_every=0,
            )
            _offset_notes(bar_notes, (bar_idx + i) * TICKS_PER_BAR)
            notes.extend(bar_notes)
        bar_idx += 4

        if bar_idx >= total_bars:
            break

        # 4-bar phrase: drums respond (solo fills)
        for i in range(min(4, total_bars - bar_idx)):
            bar_beats = 4
            # High-intensity drums for the response
            bar_notes = generate_drums(
                bar_beats, intensity=min(1.0, intensity + 0.3), swing=swing,
                fill_every=2,  # More frequent fills
            )
            _offset_notes(bar_notes, (bar_idx + i) * TICKS_PER_BAR)
            notes.extend(bar_notes)
        bar_idx += 4

    return notes


def _generate_coda(
    section: ArrangementSection,
    chords: List[ChordEvent],
    swing: bool,
) -> Dict[str, List[NoteEvent]]:
    """Generate coda: repeats last 4 bars of changes with decreasing intensity.

    Three repetitions of the final 4 bars (or available chords), plus a
    final sustained chord. The melody sustains the last note.
    """
    section_chords = _chords_for_section(chords, section)
    beats = section.total_beats
    tick_offset = section.start_beat * TICKS_PER_QUARTER

    # Use the last few chords from the section (or all if short)
    # Generate with decreasing intensity
    melody_notes: List[NoteEvent] = []
    bass_notes: List[NoteEvent] = []
    drum_notes: List[NoteEvent] = []
    piano_notes: List[NoteEvent] = []

    coda_bars = beats // 4
    if coda_bars <= 0:
        return {"melody": [], "piano": [], "bass": [], "drums": []}

    # Split into segments with decreasing intensity
    bars_done = 0
    intensities = [0.4, 0.25, 0.15, 0.1]

    for seg_idx in range(min(coda_bars, len(intensities))):
        seg_bars = 1
        seg_beats = seg_bars * 4
        seg_intensity = intensities[seg_idx]

        seg_start_beat = section.start_beat + bars_done * 4
        seg_chords = _chords_for_beat_range(chords, seg_start_beat, seg_start_beat + seg_beats)
        if not seg_chords:
            seg_chords = section_chords[-1:] if section_chords else []
        if not seg_chords:
            bars_done += seg_bars
            continue

        seg_tick_offset = int((section.start_beat + bars_done * 4) * TICKS_PER_QUARTER)

        # Two-feel bass for ritardando feel
        b = generate_two_feel_bass(seg_chords, seg_beats, swing=swing)
        _offset_notes(b, seg_tick_offset)
        bass_notes.extend(b)

        # Sparse drums, no fills
        d = generate_drums(seg_beats, intensity=seg_intensity, swing=swing, fill_every=0)
        _offset_notes(d, int(seg_tick_offset))
        drum_notes.extend(d)

        # Sparse piano
        p = generate_comping(seg_chords, seg_beats, intensity=seg_intensity, swing=swing)
        _offset_notes(p, int(seg_tick_offset))
        piano_notes.extend(p)

        bars_done += seg_bars

    # Final sustained melody note: root of the last chord, held for 2 bars
    if section_chords:
        final_chord = section_chords[-1]
        final_root = final_chord.root_pc
        # Place the sustained note in the melody range
        final_pitch = 60 + final_root  # Middle-ish register
        if final_pitch > 84:
            final_pitch -= 12
        final_tick = int(tick_offset + max(0, (beats - 8)) * TICKS_PER_QUARTER)
        melody_notes.append(NoteEvent(
            pitch=final_pitch,
            start_tick=final_tick,
            duration_ticks=8 * TICKS_PER_QUARTER,  # 2 bars sustained
            velocity=75,
            channel=0,
        ))

    # Fermata: double the last bass note's duration and sustain last piano chord
    if bass_notes:
        bass_notes[-1].duration_ticks *= 2
    if piano_notes:
        # Sustain the last few piano notes (the final chord voicing)
        last_tick = piano_notes[-1].start_tick
        for n in reversed(piano_notes):
            if n.start_tick < last_tick - 30:
                break
            n.duration_ticks = 4 * TICKS_PER_QUARTER  # Hold for a full bar

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
        ))

    return result


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
    lead_instrument: str = "trumpet",
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
                                            bass_style, drum_style)
        elif section.name in ("head_in", "head_out"):
            section_notes = _generate_head(section, section_chords, swing,
                                           coltrane, bass_style, drum_style)
        elif section.is_solo:
            section_notes = _generate_solo_section(
                section, section_chords, tension_curve, swing, coltrane, seed,
                density_scale, bass_style, drum_style,
            )
        elif section.is_trading:
            section_notes = _generate_trading_section(section, section_chords, swing,
                                                      bass_style, drum_style)
        elif section.name == "coda":
            section_notes = _generate_coda(section, section_chords, swing)
        else:
            section_notes = _generate_head(section, section_chords, swing,
                                           coltrane, bass_style, drum_style)

        _merge_tracks(tracks, section_notes)

    # Piano trio mode: when lead is piano, drop the comping track entirely.
    # A real piano trio (Bill Evans, etc.) has one pianist doing both melody
    # and comping — not two keyboards.
    if lead_instrument == "piano":
        tracks["piano"] = []

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

    return tracks
