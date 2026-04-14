"""Melody and solo generator for Coltrain rule-based jazz generation.

Provides three main generation functions:
  - generate_head_melody: Composed head melody (simple, thematic)
  - generate_solo: Improvised solo with tension curve (melodic -> motivic -> sheets of sound)
  - generate_trading_fours: Trading fours with drums (4 bars melody, 4 bars silence)
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from coltrain.generation import NoteEvent, CCEvent, PitchBendEvent, BarFeel, TICKS_PER_QUARTER, TICKS_PER_8TH, TICKS_PER_16TH, TICKS_PER_BAR, ticks_per_bar

# Module-level rhythmic feel state (set per phrase in generator loop)
_current_feel: Optional[BarFeel] = None
from coltrain.theory.chord import ChordEvent, CHORD_TONES, TENSIONS
from coltrain.theory.scale import SCALES, CHORD_SCALE_MAP, get_scale_notes_midi

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base register (at tension = 0.0)
MELODY_LOW_BASE = 55    # G3
MELODY_HIGH_BASE = 84   # C6

# Extreme register (at tension = 1.0)
MELODY_LOW_EXTREME = 48  # C3
MELODY_HIGH_EXTREME = 91 # G6

# Legacy aliases (kept for backward compatibility with other modules)
MELODY_LOW = MELODY_LOW_EXTREME
MELODY_HIGH = MELODY_HIGH_EXTREME

# ---------------------------------------------------------------------------
# Digital patterns (Coltrane-style interval patterns from root)
# ---------------------------------------------------------------------------

DIGITAL_PATTERNS = [
    [0, 2, 4, 7],     # 1-2-3-5 ascending
    [0, 2, 4, 9],     # 1-2-3-6
    [7, 4, 2, 0],     # 5-3-2-1 descending
    [0, 4, 7, 12],    # triad arpeggio up
    [12, 7, 4, 0],    # triad arpeggio down
    [0, 4, 7, 11],    # maj7 arpeggio
    [0, 3, 7, 10],    # min7 arpeggio
    [0, 2, 4, 7, 9],  # major pentatonic
]

# ---------------------------------------------------------------------------
# Rhythmic cells -- pre-composed duration sequences with syncopation
# ---------------------------------------------------------------------------

RHYTHMIC_CELLS_SPARSE = [
    # Total: ~2 beats each. Suited for tier 1 / head.
    (2.0,),
    (1.5, 0.5),
    (0.5, 1.5),
    (1.0, 1.0),
    (1.0, 0.5, 0.5),
    (0.5, 0.5, 1.0),
    (0.5, 1.0, 0.5),
]

RHYTHMIC_CELLS_MEDIUM = [
    # Total: ~2 beats each. Suited for tier 2.
    (0.75, 0.25, 1.0),
    (0.5, 0.5, 0.5, 0.5),
    (0.25, 0.75, 0.5, 0.5),
    (0.5, 0.75, 0.75),
    (1.0, 0.25, 0.25, 0.5),
    (0.5, 0.25, 0.25, 0.5, 0.5),
    (0.75, 0.25, 0.5, 0.5),
]

RHYTHMIC_CELLS_DENSE = [
    # Total: ~1 beat each. Suited for tier 3.
    (0.25, 0.25, 0.25, 0.25),
    (0.25, 0.25, 0.5),
    (0.5, 0.25, 0.25),
    (0.75, 0.25),
    (0.25, 0.75),
]

RHYTHMIC_CELLS_TRIPLET = [
    # Total: 1 beat each. Triplet groupings.
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    (2.0 / 3.0, 1.0 / 3.0),
    (1.0 / 3.0, 2.0 / 3.0),
]

# ---------------------------------------------------------------------------
# Head phrase templates -- pre-composed 2-bar (8-beat) melodic shapes
# ---------------------------------------------------------------------------

# Each template defines a rhythmic skeleton + melodic contour for the head.
# rhythm: list of (beat_offset, duration) within an 8-beat (2-bar) phrase
# contour: per-note direction (+1=next chord tone up, -1=down, 0=snap/repeat,
#          +2=skip one chord tone up, -2=skip one down)

HEAD_PHRASE_TEMPLATES = [
    # 0: "Long Tone" — Impressions-style: held note, short response
    {"rhythm": [(0.0, 3.0), (4.0, 2.0), (6.5, 1.0)],
     "contour": [0, +1, -1]},

    # 1: "Dotted Call" — Blue Train-style: dotted-quarter syncopation
    {"rhythm": [(0.0, 1.5), (1.5, 1.5), (3.0, 1.0), (5.0, 1.5), (6.5, 1.0)],
     "contour": [0, +1, +1, -1, -1]},

    # 2: "Ascending Stab" — Stolen Moments-style: pickup stabs → long tone
    {"rhythm": [(0.5, 0.75), (1.5, 0.75), (4.0, 2.5)],
     "contour": [0, +1, -1]},

    # 3: "Repeated Note" — Equinox-style: rhythmic insistence on one pitch
    {"rhythm": [(0.0, 1.0), (1.5, 1.0), (3.0, 1.0), (5.0, 2.5)],
     "contour": [0, 0, 0, +1]},

    # 4: "Pickup Figure" — bebop-style: syncopated approach
    {"rhythm": [(0.5, 0.5), (1.0, 1.5), (4.0, 1.0), (5.5, 2.0)],
     "contour": [+1, +1, -1, -1]},

    # 5: "Pentatonic Rise" — Cantaloupe Island-style: stepwise climb + reset
    {"rhythm": [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.5), (6.0, 1.5)],
     "contour": [0, +1, +1, +1, -2]},

    # 6: "Call and Space" — Maiden Voyage-style: short call, long silence, resolve
    {"rhythm": [(0.0, 1.0), (1.0, 0.5), (5.0, 2.5)],
     "contour": [0, +1, -1]},

    # 7: "Syncopated Hits" — Moanin'-style: off-beat accents
    {"rhythm": [(0.5, 0.75), (2.0, 0.75), (3.5, 0.75), (5.5, 0.75), (7.0, 0.75)],
     "contour": [0, -1, +1, -1, +1]},

    # 8: "Descending Arc" — Round Midnight-style: sighing descent
    {"rhythm": [(0.0, 2.0), (2.5, 1.5), (4.5, 1.5), (6.5, 1.5)],
     "contour": [0, -1, -1, -1]},

    # 9: "Triplet Tag" — Moment's Notice-style: quick triplet → long tone
    {"rhythm": [(0.0, 2/3), (2/3, 2/3), (4/3, 2/3), (2.5, 2.5), (6.0, 1.5)],
     "contour": [0, +1, +1, -1, 0]},
]

# ---------------------------------------------------------------------------
# Motif system for motivic development
# ---------------------------------------------------------------------------


@dataclass
class Motif:
    """A short melodic idea defined by intervals and durations."""
    intervals: list   # relative semitone intervals from first note
    durations: list   # beat durations for each note

    def invert(self) -> "Motif":
        return Motif([-iv for iv in self.intervals], list(self.durations))

    def retrograde(self) -> "Motif":
        return Motif(list(reversed(self.intervals)), list(reversed(self.durations)))

    def augment(self) -> "Motif":
        return Motif(list(self.intervals), [d * 2 for d in self.durations])

    def diminish(self) -> "Motif":
        return Motif(list(self.intervals), [d * 0.5 for d in self.durations])

    def transpose(self, semitones: int) -> "Motif":
        return Motif([iv + semitones for iv in self.intervals], list(self.durations))

    def extend(self, extra_interval: int = 0, extra_duration: float = 0.5) -> "Motif":
        """Append one note to the motif (cumulative extension)."""
        last_iv = self.intervals[-1] if self.intervals else 0
        new_intervals = list(self.intervals) + [last_iv + extra_interval]
        new_durations = list(self.durations) + [extra_duration]
        return Motif(new_intervals, new_durations)


SEED_MOTIFS = [
    # Original motifs
    Motif([0, 2, 4, 7], [0.5, 0.5, 0.5, 0.5]),
    Motif([0, -1, -3, -5], [0.5, 0.5, 0.5, 0.5]),
    Motif([0, 4, 7, 4], [0.5, 0.5, 0.5, 0.5]),
    Motif([0, 7, 5, 3, 0], [0.25, 0.25, 0.25, 0.25, 1.0]),
    # Syncopated motifs with dotted rhythms
    Motif([0, 2, 5, 7], [0.75, 0.25, 0.5, 0.5]),
    Motif([0, -2, -3, 0, 4], [0.5, 0.25, 0.25, 0.5, 0.5]),
    Motif([0, 7, 12, 7], [0.25, 0.25, 1.0, 0.5]),
    Motif([0, 0, -1, 2, 4], [0.25, 0.25, 0.25, 0.25, 1.0]),
    Motif([0, 5, 4, 2], [1.0, 0.5, 0.75, 0.25]),
    Motif([0, -5, -3, -1, 0], [0.25, 0.25, 0.25, 0.25, 0.5]),
]

# ---------------------------------------------------------------------------
# Solo state -- carries context across phrase boundaries
# ---------------------------------------------------------------------------


@dataclass
class _PhraseMemo:
    """A remembered phrase for long-range callback development."""
    notes: List[NoteEvent]
    beat_position: float
    tension: float
    quality_score: float
    tier: int
    chord_root_pc: int


# ---------------------------------------------------------------------------
# Solo narrative -- phrase character archetypes and pre-planned intentions
# ---------------------------------------------------------------------------

PC_QUESTION = "question"
PC_ANSWER = "answer"
PC_EXCLAMATION = "exclamation"
PC_WHISPER = "whisper"
PC_CONTINUATION = "continuation"
PC_CONTRAST = "contrast"
PC_SILENCE = "silence"


@dataclass
class PhraseIntention:
    """Pre-planned intention for a single phrase in the solo narrative."""
    character: str = PC_QUESTION
    target_tier: int = 1
    phrase_beats_hint: float = 4.0
    register_target: float = 0.5
    density_bias: float = 0.0
    direction_bias: float = 0.0
    strategy_weights: Optional[Dict[str, float]] = None
    silence_after: float = 1.0
    velocity_offset: int = 0
    motif_action: str = "free"


@dataclass
class SoloNarrative:
    """Pre-planned narrative arc for an entire solo."""
    intentions: List[PhraseIntention] = field(default_factory=list)
    harmonic_landmarks: List[Tuple[float, str]] = field(default_factory=list)
    climax_beat: float = 0.0
    total_beats: float = 0.0

    def get_intention(self, phrase_index: int) -> Optional[PhraseIntention]:
        if phrase_index < len(self.intentions):
            return self.intentions[phrase_index]
        return None


@dataclass
class RunBlueprint:
    """Pre-planned architectural unit for a Tatum-style run."""
    purpose: str              # "bridge", "fill", "climax", "ornament", "turnaround"
    start_beat: float         # Where the run begins
    end_beat: float           # Where the target note lands
    target_pitch: int         # Guide tone of destination chord
    contour: str              # "sweep_up", "sweep_down", "arc_up_down", "arc_down_up", "cascade"
    scale_materials: List[Tuple[float, int, str]]  # [(beat_boundary, root_pc, quality), ...]
    timing_profile: str       # "decelerate", "accelerate", "even"
    intensity: float          # 0.0-1.0
    two_handed: bool = False
    executed: bool = False


@dataclass
class RunPlan:
    """Collection of pre-planned runs for one solo chorus."""
    blueprints: List[RunBlueprint] = field(default_factory=list)

    def get_active_run(self, beat: float) -> Optional[RunBlueprint]:
        """Return un-executed blueprint where start_beat <= beat+0.5 and beat < end_beat."""
        for bp in self.blueprints:
            if not bp.executed and bp.start_beat <= beat + 0.5 and beat < bp.end_beat:
                return bp
        return None

    def get_upcoming_run(self, beat: float, lookahead: float = 2.0) -> Optional[RunBlueprint]:
        """Return next un-executed blueprint within lookahead beats."""
        for bp in self.blueprints:
            if not bp.executed and bp.start_beat > beat and bp.start_beat <= beat + lookahead:
                return bp
        return None


@dataclass
class _SoloState:
    """Mutable state carried across phrase boundaries during solo generation."""
    pitch: int
    recent_pitches: List[int] = field(default_factory=list)
    active_motif: Optional[Motif] = None
    motif_play_count: int = 0
    motif_max_plays: int = 4
    phrase_count: int = 0
    last_phrase_length: float = 0.0
    substitute_tones: List[int] = field(default_factory=list)
    last_tension: float = 0.0
    last_key_center_pc: Optional[int] = None
    last_tier: int = 1
    last_form_section: str = ""
    target_queue: List[Tuple[float, int]] = field(default_factory=list)
    last_target_was_third: bool = False
    phrase_buffer: List[_PhraseMemo] = field(default_factory=list)
    callback_count: int = 0
    # --- Run blueprint fields (v13) ---
    run_plan: Optional[RunPlan] = None
    runs_executed: int = 0
    last_run_end_beat: float = -999.0
    # --- Narrative context fields (Solo Narrative Planner v6) ---
    narrative: Optional[SoloNarrative] = None
    phrase_length_history: List[float] = field(default_factory=list)
    strategy_history: List[str] = field(default_factory=list)
    direction_momentum: float = 0.0          # EMA of pitch direction, -1..+1
    register_ema: float = 66.0               # EMA of recent pitch (middle C area)
    beats_since_last_silence: float = 0.0
    last_silence_duration: float = 0.0
    last_phrase_ending_pitch: int = 0
    last_phrase_ending_interval: int = 0
    last_phrase_was_resolved: bool = True
    phrase_interval_history: List[int] = field(default_factory=list)
    phrase_cell_history: List[Tuple] = field(default_factory=list)
    _last_contour: str = ""                  # Last phrase's contour (for continuation/contrast)
    last_run_direction: Optional[int] = None  # Set after scalar/chromatic/arpeggio runs end
    beats_per_bar: int = 4  # Meter (4 for 4/4, 5 for 5/4, 7 for 7/4)
    harmonic_rhythm_speed: float = 0.0  # 0.0 = slow (≥3 beats), 1.0 = fast (≤1 beat)

    # --- Narrative context methods ---
    def record_phrase_length(self, length: float):
        self.phrase_length_history.append(length)
        if len(self.phrase_length_history) > 8:
            self.phrase_length_history = self.phrase_length_history[-8:]

    def record_strategy(self, strategy: str):
        self.strategy_history.append(strategy)
        if len(self.strategy_history) > 8:
            self.strategy_history = self.strategy_history[-8:]

    def update_direction_momentum(self, interval: int):
        """EMA of pitch direction. Positive = ascending, negative = descending."""
        direction = 1.0 if interval > 0 else (-1.0 if interval < 0 else 0.0)
        alpha = 0.3
        self.direction_momentum = alpha * direction + (1 - alpha) * self.direction_momentum

    def update_register_ema(self, pitch: int):
        """EMA of recent pitch for register tracking."""
        alpha = 0.2
        self.register_ema = alpha * pitch + (1 - alpha) * self.register_ema

    def reset_phrase_tracking(self):
        """Reset per-phrase accumulators at the start of each new phrase."""
        self.phrase_interval_history = []
        self.phrase_cell_history = []
        self.last_run_direction = None

    def advance_motif(self, tension: float,
                      motif_action: str = "free") -> Motif:
        """Get next motif with cumulative Coltrane-style development.

        motif_action overrides from narrative planner:
        - "new": force fresh motif from SEED_MOTIFS
        - "callback": extract motif from best phrase in buffer
        - "develop": skip to next development stage
        - "free": existing behavior

        Sequence: original → transpose up → extend → transpose extended → diminish.
        """
        # Tension-scaled max plays: patient at low tension, rapid cycling at high
        _max = max(3, min(6, int(6 - tension * 3)))
        _EXTEND_INTERVALS = [2, 3, 4, 5, 7]

        # Narrative overrides
        if motif_action == "new":
            self.active_motif = SEED_MOTIFS[self.phrase_count % len(SEED_MOTIFS)]
            self.motif_play_count = 0
            self.motif_max_plays = _max
            return self.active_motif
        elif motif_action == "callback" and self.phrase_buffer:
            best = max(self.phrase_buffer, key=lambda m: m.quality_score)
            if len(best.notes) >= 3:
                intervals = []
                for i in range(1, min(5, len(best.notes))):
                    intervals.append(best.notes[i].pitch - best.notes[i - 1].pitch)
                if intervals:
                    durations = [0.25] * len(intervals)
                    self.active_motif = Motif(intervals=tuple(intervals),
                                              durations=tuple(durations))
                    self.motif_play_count = 0
                    self.motif_max_plays = _max
                    return self.active_motif
        elif motif_action == "develop" and self.active_motif is not None:
            self.motif_play_count = min(self.motif_play_count + 1,
                                        self.motif_max_plays - 1)

        # Standard behavior
        if self.active_motif is None or self.motif_play_count >= self.motif_max_plays:
            self.active_motif = SEED_MOTIFS[self.phrase_count % len(SEED_MOTIFS)]
            self.motif_play_count = 0
            self.motif_max_plays = _max

        count = self.motif_play_count
        self.motif_play_count += 1

        m = self.active_motif
        if count == 0:
            return m
        elif count == 1:
            # Transpose: scale with tension (2 at low, 5 at high)
            return m.transpose(2 + int(tension * 3))
        elif count == 2:
            # Extend: cycle through interval options
            extra = _EXTEND_INTERVALS[self.phrase_count % len(_EXTEND_INTERVALS)]
            return m.extend(extra_interval=extra)
        elif count == 3:
            extra = _EXTEND_INTERVALS[self.phrase_count % len(_EXTEND_INTERVALS)]
            return m.extend(extra_interval=extra).transpose(3 + int(tension * 4))
        else:
            return m.diminish()

    def record_pitch(self, p: int):
        self.recent_pitches.append(p)
        if len(self.recent_pitches) > 20:
            self.recent_pitches = self.recent_pitches[-20:]

    def record_phrase(self, notes: List[NoteEvent], beat: float,
                      tension: float, tier: int, chord_root_pc: int):
        """Store a phrase for potential callback later."""
        if len(notes) < 2:
            return
        score = _evaluate_phrase_quality(notes, chord_root_pc)
        memo = _PhraseMemo(
            notes=list(notes), beat_position=beat, tension=tension,
            quality_score=score, tier=tier, chord_root_pc=chord_root_pc)
        self.phrase_buffer.append(memo)
        if len(self.phrase_buffer) > 12:
            self.phrase_buffer = self.phrase_buffer[-12:]


@dataclass
class _PhraseBlueprint:
    """Pre-planned structure for a single phrase — gives notes a destination."""
    goal_pitch: int                           # Chord tone to land on near phrase end
    contour: str                              # "arch", "ascending", "descending", "valley", "pendulum"
    interval_mode: str                        # "stepwise", "thirds", "wide"
    active_cell: Optional[List[int]] = None   # Interval pattern being developed
    cell_uses: int = 0                        # Times current cell has been sequenced


def _evaluate_phrase_quality(notes: List[NoteEvent], chord_root_pc: int) -> float:
    """Score a phrase 0.0-1.0 for callback worthiness."""
    if len(notes) < 2:
        return 0.0

    # Resolution score: does the last note land on a chord tone?
    last_pc = notes[-1].pitch % 12
    root_intervals = {0, 3, 4, 7, 10, 11}  # common chord tones
    interval_from_root = (last_pc - chord_root_pc) % 12
    resolution = 1.0 if interval_from_root in root_intervals else 0.3

    # Interval score: average interval 2-5 semitones is ideal
    intervals = [abs(notes[i+1].pitch - notes[i].pitch) for i in range(len(notes)-1)]
    avg_interval = sum(intervals) / len(intervals) if intervals else 0
    if 2 <= avg_interval <= 5:
        interval_score = 1.0
    elif avg_interval < 2:
        interval_score = avg_interval / 2.0
    else:
        interval_score = max(0.2, 1.0 - (avg_interval - 5) / 7.0)

    # Length score: 3-8 notes is ideal
    n = len(notes)
    if 3 <= n <= 8:
        length_score = 1.0
    elif n < 3:
        length_score = n / 3.0
    else:
        length_score = max(0.3, 1.0 - (n - 8) / 10.0)

    return 0.4 * resolution + 0.3 * interval_score + 0.3 * length_score


# ---------------------------------------------------------------------------
# Solo Narrative Pre-Planner
# ---------------------------------------------------------------------------

def _plan_solo_narrative(chords: List[ChordEvent], total_beats: float,
                         curve: 'TensionCurve', intensity: float,
                         coltrane: bool) -> SoloNarrative:
    """Pre-plan a narrative arc for the entire solo.

    Three passes:
    1. Scan chords for harmonic landmarks (key changes, resolutions, section boundaries)
    2. Find the climax beat by sampling the tension curve
    3. Generate phrase intentions based on solo position and landmarks
    """
    # --- Pass 1: Harmonic landscape ---
    landmarks: List[Tuple[float, str]] = []
    prev_key = chords[0].key_center_pc if chords else 0
    prev_section = chords[0].form_section if chords else ""

    for c in chords:
        # Key center change
        if c.key_center_pc != prev_key:
            landmarks.append((c.start_beat, "key_change"))
            prev_key = c.key_center_pc
        # Section boundary
        if c.form_section and c.form_section != prev_section:
            landmarks.append((c.start_beat, "section_boundary"))
            prev_section = c.form_section
        # ii-V resolution: dominant quality resolving down a fifth
        if c.quality in ("dom7", "7", "dom7alt") and c.function in ("V", "V7"):
            landmarks.append((c.start_beat + c.duration_beats, "resolution"))

    # --- Pass 2: Find climax beat ---
    climax_beat = total_beats * 0.7  # default
    peak_tension = 0.0
    for i in range(100):
        progress = i / 99.0
        t = curve(progress)
        if t > peak_tension:
            peak_tension = t
            climax_beat = progress * total_beats

    # --- Pass 3: Generate phrase intentions ---
    # Estimate ~total_beats/5 phrases (average 5 beats each)
    est_phrases = max(4, int(total_beats / 5))
    intentions: List[PhraseIntention] = []

    # Character weight tables by solo position
    #                       question, answer, exclamation, whisper, continuation, contrast, silence
    OPENING_W =             [0.30,    0.25,   0.05,        0.15,   0.10,          0.10,     0.05]
    DEVELOPMENT_W =         [0.10,    0.15,   0.10,        0.10,   0.35,          0.15,     0.05]
    BUILDING_W =            [0.08,    0.10,   0.25,        0.05,   0.25,          0.20,     0.07]
    CLIMAX_W =              [0.05,    0.05,   0.45,        0.02,   0.25,          0.15,     0.03]
    RESOLUTION_W =          [0.05,    0.35,   0.05,        0.25,   0.10,          0.10,     0.10]
    CHARACTERS = [PC_QUESTION, PC_ANSWER, PC_EXCLAMATION, PC_WHISPER,
                  PC_CONTINUATION, PC_CONTRAST, PC_SILENCE]

    # Landmark beats for quick lookup
    landmark_set = {b: kind for b, kind in landmarks}

    beat_cursor = 0.0
    for i in range(est_phrases):
        progress = i / max(1, est_phrases - 1)
        est_beat = progress * total_beats

        # Select weight table by position
        if progress < 0.15:
            weights = OPENING_W[:]
        elif progress < 0.35:
            weights = DEVELOPMENT_W[:]
        elif progress < 0.65:
            weights = BUILDING_W[:]
        elif progress < 0.80:
            weights = CLIMAX_W[:]
        else:
            weights = RESOLUTION_W[:]

        # Override near harmonic landmarks
        nearest_landmark = None
        for lb, lk in landmarks:
            if abs(lb - est_beat) < 6.0:  # within 6 beats
                nearest_landmark = lk
                break

        if nearest_landmark == "key_change":
            # Force contrast at key changes
            weights = [0.0, 0.0, 0.1, 0.0, 0.0, 0.85, 0.05]
        elif nearest_landmark == "resolution":
            # Force answer at resolutions
            weights = [0.0, 0.75, 0.0, 0.05, 0.1, 0.05, 0.05]
        elif nearest_landmark == "section_boundary":
            # Breath at section boundaries
            weights = [0.15, 0.15, 0.05, 0.15, 0.05, 0.15, 0.30]

        character = random.choices(CHARACTERS, weights=weights, k=1)[0]

        # Tier based on tension at this point
        t = curve(progress) if callable(curve) else 0.5
        if t < 0.35:
            target_tier = 1
        elif t < 0.7:
            target_tier = 2
        else:
            target_tier = 3

        # Phrase length hint based on character
        if character == PC_EXCLAMATION:
            phrase_hint = random.uniform(5.0, 10.0)
        elif character == PC_WHISPER:
            phrase_hint = random.uniform(1.5, 3.0)
        elif character == PC_SILENCE:
            phrase_hint = random.uniform(0.5, 1.5)
        elif character == PC_QUESTION:
            phrase_hint = random.uniform(2.5, 5.0)
        elif character == PC_ANSWER:
            phrase_hint = random.uniform(3.0, 6.0)
        else:
            phrase_hint = random.uniform(3.0, 7.0)

        # Register target: slow arc 0.4 → 0.9 at climax → 0.4
        climax_progress = climax_beat / max(1.0, total_beats)
        if progress < climax_progress:
            reg = 0.4 + 0.5 * (progress / max(0.01, climax_progress))
        else:
            remaining = 1.0 - climax_progress
            reg = 0.9 - 0.5 * ((progress - climax_progress) / max(0.01, remaining))
        reg = max(0.2, min(0.95, reg))

        # Direction bias: ascending early, neutral mid, descending late
        if progress < 0.3:
            dir_bias = 0.2
        elif progress < 0.7:
            dir_bias = 0.0
        else:
            dir_bias = -0.3

        # Density bias from character
        density_map = {
            PC_EXCLAMATION: 0.3, PC_WHISPER: -0.4, PC_SILENCE: -1.0,
            PC_QUESTION: -0.1, PC_ANSWER: 0.0, PC_CONTINUATION: 0.1,
            PC_CONTRAST: 0.0,
        }

        # Velocity offset from character
        vel_map = {
            PC_EXCLAMATION: 15, PC_WHISPER: -20, PC_SILENCE: 0,
            PC_QUESTION: 0, PC_ANSWER: 5, PC_CONTINUATION: 0,
            PC_CONTRAST: 8,
        }

        # Motif action
        if character == PC_CONTINUATION:
            motif_action = "develop"
        elif character == PC_CONTRAST:
            motif_action = "new"
        elif character == PC_ANSWER and i > 0:
            motif_action = "callback"
        else:
            motif_action = "free"

        # Silence after phrase
        if character == PC_SILENCE:
            silence_after = random.uniform(2.0, 6.0)
        elif character == PC_EXCLAMATION:
            silence_after = random.uniform(0.0, 0.5)
        elif character == PC_WHISPER:
            silence_after = random.uniform(1.0, 3.0)
        else:
            silence_after = random.uniform(0.5, 2.0)

        intentions.append(PhraseIntention(
            character=character,
            target_tier=target_tier,
            phrase_beats_hint=phrase_hint,
            register_target=reg,
            density_bias=density_map.get(character, 0.0),
            direction_bias=dir_bias,
            strategy_weights=None,
            silence_after=silence_after,
            velocity_offset=vel_map.get(character, 0),
            motif_action=motif_action,
        ))

    return SoloNarrative(
        intentions=intentions,
        harmonic_landmarks=landmarks,
        climax_beat=climax_beat,
        total_beats=total_beats,
    )


# ---------------------------------------------------------------------------
# Run Blueprint Planner (v13) -- pre-composed Tatum-style run opportunities
# ---------------------------------------------------------------------------


def _plan_run_opportunities(chords: List[ChordEvent],
                            narrative: SoloNarrative,
                            total_beats: float,
                            intensity: float,
                            curve: 'TensionCurve') -> RunPlan:
    """Scan chords and narrative to identify 3-8 run opportunities per solo.

    5 opportunity types:
    1. Bridge — before a major chord change
    2. Fill — after narrative silence phrases
    3. Climax — at the narrative climax beat
    4. Turnaround — before form section returns to "A"
    5. Section boundary — at harmonic landmarks
    """
    if not chords or total_beats <= 0:
        return RunPlan()

    opportunities: List[RunBlueprint] = []

    # Helper: estimate register at a given solo progress position
    def _est_register(progress: float) -> float:
        """Return estimated MIDI pitch at this solo position (55-84 range)."""
        climax_progress = narrative.climax_beat / max(1.0, total_beats)
        if progress < climax_progress:
            reg = 0.4 + 0.5 * (progress / max(0.01, climax_progress))
        else:
            remaining = 1.0 - climax_progress
            reg = 0.9 - 0.5 * ((progress - climax_progress) / max(0.01, remaining))
        reg = max(0.2, min(0.95, reg))
        return int(MELODY_LOW_BASE + reg * (MELODY_HIGH_BASE - MELODY_LOW_BASE))

    def _closest_guide_tone(target_pitch: int, root_pc: int, quality: str) -> int:
        """Find the closest guide tone (3rd or 7th) to a target pitch."""
        guides = _guide_tones_in_range(root_pc, quality,
                                       MELODY_LOW_EXTREME, MELODY_HIGH_EXTREME)
        if not guides:
            return target_pitch
        return min(guides, key=lambda g: abs(g - target_pitch))

    def _build_scale_materials(start_beat: float, end_beat: float) -> List[Tuple[float, int, str]]:
        """Build chord-based scale material list for a run span."""
        materials = []
        for c in chords:
            if c.end_beat <= start_beat:
                continue
            if c.start_beat >= end_beat:
                break
            boundary = max(start_beat, c.start_beat)
            materials.append((boundary, c.root_pc, c.quality))
        if not materials:
            # Fallback to chord at start
            ch = _get_chord_at_beat(chords, start_beat)
            if ch:
                materials.append((start_beat, ch.root_pc, ch.quality))
        return materials

    def _contour_for_direction(source_est: int, target: int) -> str:
        if target > source_est + 3:
            return "sweep_up"
        elif target < source_est - 3:
            return "sweep_down"
        return "arc_up_down"

    # 1. Bridge runs — before major chord changes
    for i, c in enumerate(chords):
        if c.start_beat < 8.0 or c.start_beat > total_beats - 4.0:
            continue
        prev = _get_chord_at_beat(chords, c.start_beat - 0.1)
        if prev is None or prev.root_pc == c.root_pc:
            continue
        # Check tension gate
        progress = c.start_beat / total_beats
        t = curve(progress) * intensity
        if t < 0.3:
            continue
        # Run starts 1.5-3 beats before chord change, lands on beat 1 of new chord
        run_dur = min(3.0, max(1.5, c.duration_beats * 0.75))
        start_b = c.start_beat - run_dur
        if start_b < 0:
            continue
        est_reg = _est_register(start_b / total_beats)
        target_p = _closest_guide_tone(est_reg, c.root_pc, c.quality)
        contour = _contour_for_direction(est_reg, target_p)
        materials = _build_scale_materials(start_b, c.start_beat + 0.5)
        # Bridge runs over dom7: 30% chance to inject tritone sub material
        if prev.quality in ("dom7", "7", "dom7alt") and len(materials) > 0:
            tritone_root = (prev.root_pc + 6) % 12
            # Insert at midpoint
            mid_beat = (start_b + c.start_beat) / 2
            if hash((start_b, prev.root_pc)) % 10 < 3:
                materials.insert(len(materials) // 2,
                                 (mid_beat, tritone_root, "dom7"))
        timing = "decelerate" if hash((start_b, c.root_pc)) % 5 < 3 else "even"
        opportunities.append(RunBlueprint(
            purpose="bridge", start_beat=start_b, end_beat=c.start_beat + 0.5,
            target_pitch=target_p, contour=contour,
            scale_materials=materials, timing_profile=timing,
            intensity=min(1.0, t + 0.1),
            two_handed=(t > 0.7),
        ))

    # 2. Fill runs — after narrative silence phrases
    for idx, intn in enumerate(narrative.intentions):
        if intn.character != PC_SILENCE:
            continue
        # Estimate the beat where silence ends
        est_beat = (idx / max(1, len(narrative.intentions) - 1)) * total_beats
        fill_start = est_beat + intn.silence_after
        if fill_start < 8.0 or fill_start > total_beats - 4.0:
            continue
        fill_end = fill_start + min(2.5, max(1.5, intn.silence_after))
        if fill_end > total_beats - 4.0:
            continue
        progress = fill_start / total_beats
        t = curve(progress) * intensity
        est_reg = _est_register(progress)
        ch = _get_chord_at_beat(chords, fill_end)
        if ch is None:
            continue
        target_p = _closest_guide_tone(est_reg, ch.root_pc, ch.quality)
        contour = "sweep_up" if hash(idx) % 2 == 0 else "sweep_down"
        materials = _build_scale_materials(fill_start, fill_end)
        opportunities.append(RunBlueprint(
            purpose="fill", start_beat=fill_start, end_beat=fill_end,
            target_pitch=target_p, contour=contour,
            scale_materials=materials, timing_profile="accelerate",
            intensity=min(1.0, t + 0.05),
        ))

    # 3. Climax run — at the narrative climax beat
    if narrative.climax_beat > 8.0 and narrative.climax_beat < total_beats - 4.0:
        climax_progress = narrative.climax_beat / total_beats
        t = curve(climax_progress) * intensity
        if t > 0.6:
            run_dur = min(6.0, max(3.0, t * 5.0))
            start_b = narrative.climax_beat - run_dur * 0.6
            end_b = narrative.climax_beat + run_dur * 0.4
            if start_b >= 8.0:
                est_reg = _est_register(climax_progress)
                ch = _get_chord_at_beat(chords, narrative.climax_beat)
                if ch:
                    target_p = _closest_guide_tone(
                        est_reg + 8, ch.root_pc, ch.quality)  # aim high for climax
                    materials = _build_scale_materials(start_b, end_b)
                    opportunities.append(RunBlueprint(
                        purpose="climax", start_beat=start_b, end_beat=end_b,
                        target_pitch=target_p, contour="cascade",
                        scale_materials=materials, timing_profile="even",
                        intensity=min(1.0, t + 0.15),
                        two_handed=True,
                    ))

    # 4. Turnaround runs — last 2-4 beats before form returns to "A"
    for beat_pos, kind in narrative.harmonic_landmarks:
        if kind != "section_boundary":
            continue
        if beat_pos < 8.0 or beat_pos > total_beats - 4.0:
            continue
        # Check if the section at this boundary is "A" (return)
        ch_at = _get_chord_at_beat(chords, beat_pos)
        if ch_at is None or ch_at.form_section != "A":
            continue
        progress = beat_pos / total_beats
        t = curve(progress) * intensity
        if t < 0.3:
            continue
        run_dur = min(4.0, max(2.0, t * 3.0))
        start_b = beat_pos - run_dur
        if start_b < 0:
            continue
        est_reg = _est_register(progress)
        target_p = _closest_guide_tone(est_reg, ch_at.root_pc, ch_at.quality)
        materials = _build_scale_materials(start_b, beat_pos + 0.5)
        opportunities.append(RunBlueprint(
            purpose="turnaround", start_beat=start_b, end_beat=beat_pos + 0.5,
            target_pitch=target_p, contour="arc_down_up",
            scale_materials=materials, timing_profile="decelerate",
            intensity=min(1.0, t + 0.1),
        ))

    # 5. Section boundary — connective tissue at harmonic landmarks
    for beat_pos, kind in narrative.harmonic_landmarks:
        if kind not in ("section_boundary", "key_change"):
            continue
        if beat_pos < 8.0 or beat_pos > total_beats - 4.0:
            continue
        # Avoid duplicate with turnaround (already targeting A returns)
        ch_at = _get_chord_at_beat(chords, beat_pos)
        if ch_at and ch_at.form_section == "A":
            continue  # already covered by turnaround
        progress = beat_pos / total_beats
        t = curve(progress) * intensity
        if t < 0.3:
            continue
        run_dur = min(2.5, max(1.5, t * 2.0))
        start_b = beat_pos - run_dur
        if start_b < 0:
            continue
        est_reg = _est_register(progress)
        if ch_at:
            target_p = _closest_guide_tone(est_reg, ch_at.root_pc, ch_at.quality)
        else:
            target_p = est_reg
        materials = _build_scale_materials(start_b, beat_pos + 0.5)
        opportunities.append(RunBlueprint(
            purpose="boundary", start_beat=start_b, end_beat=beat_pos + 0.5,
            target_pitch=target_p, contour="sweep_down",
            scale_materials=materials, timing_profile="decelerate",
            intensity=min(1.0, t),
        ))

    # --- Filtering ---
    # Sort by start_beat
    opportunities.sort(key=lambda bp: bp.start_beat)

    # Suppress during whisper/silence narrative intentions
    intention_beats = []
    for idx, intn in enumerate(narrative.intentions):
        if intn.character in (PC_WHISPER, PC_SILENCE):
            est_beat = (idx / max(1, len(narrative.intentions) - 1)) * total_beats
            intention_beats.append((est_beat, est_beat + intn.phrase_beats_hint + intn.silence_after))

    filtered = []
    for bp in opportunities:
        # No runs in first 8 beats or last 4 beats
        if bp.start_beat < 8.0 or bp.end_beat > total_beats - 4.0:
            continue
        # Clamp end_beat to total
        if bp.end_beat > total_beats:
            bp.end_beat = total_beats
        if bp.end_beat - bp.start_beat < 1.0:
            continue
        # Check whisper/silence suppression
        suppressed = False
        for ws_start, ws_end in intention_beats:
            if bp.start_beat < ws_end and bp.end_beat > ws_start:
                suppressed = True
                break
        if suppressed:
            continue
        filtered.append(bp)

    # Min spacing: no two runs within 8 beats
    selected: List[RunBlueprint] = []
    for bp in filtered:
        if selected and bp.start_beat - selected[-1].end_beat < 8.0:
            continue
        selected.append(bp)

    # Cap at 8 runs, keep best variety by purpose
    if len(selected) > 8:
        # Prioritize: climax first, then bridge, turnaround, boundary, fill
        priority = {"climax": 0, "bridge": 1, "turnaround": 2, "boundary": 3, "fill": 4}
        selected.sort(key=lambda bp: (priority.get(bp.purpose, 5), bp.start_beat))
        selected = selected[:8]
        selected.sort(key=lambda bp: bp.start_beat)

    return RunPlan(blueprints=selected)


def _generate_callback_phrase(memo: _PhraseMemo, current_beat: float,
                              chords, state: '_SoloState',
                              swing: bool) -> Tuple[List[NoteEvent], float]:
    """Generate a callback phrase from a stored memory with transformation."""
    # Cycle: exact → transpose → fragment → augment
    _TRANSFORMS = ["exact", "transpose", "fragment", "augment"]
    transform = _TRANSFORMS[state.callback_count % len(_TRANSFORMS)]

    orig_notes = memo.notes
    if not orig_notes:
        return [], current_beat

    # Compute time offset from original to current position
    orig_start_tick = orig_notes[0].start_tick
    new_start_tick = int(current_beat * TICKS_PER_QUARTER)
    tick_delta = new_start_tick - orig_start_tick

    # Pitch transposition based on current chord vs original chord
    cur_chord = _get_chord_at_beat(chords, current_beat)
    pitch_shift = 0
    if cur_chord is not None:
        pitch_shift = (cur_chord.root_pc - memo.chord_root_pc) % 12
        if pitch_shift > 6:
            pitch_shift -= 12  # Prefer small intervals

    new_notes = []
    beat_end = current_beat

    if transform == "exact":
        # Replay with slight jitter
        for n in orig_notes:
            new_tick = n.start_tick + tick_delta + random.randint(-12, 12)
            new_vel = max(1, min(127, n.velocity + random.randint(-8, 8)))
            new_notes.append(NoteEvent(
                pitch=n.pitch, start_tick=max(0, new_tick),
                duration_ticks=n.duration_ticks, velocity=new_vel))
    elif transform == "transpose":
        for n in orig_notes:
            new_tick = n.start_tick + tick_delta + random.randint(-8, 8)
            new_pitch = max(MELODY_LOW_BASE, min(MELODY_HIGH_BASE, n.pitch + pitch_shift))
            new_notes.append(NoteEvent(
                pitch=new_pitch, start_tick=max(0, new_tick),
                duration_ticks=n.duration_ticks, velocity=n.velocity))
    elif transform == "fragment":
        # First half only
        half = max(1, len(orig_notes) // 2)
        for n in orig_notes[:half]:
            new_tick = n.start_tick + tick_delta + random.randint(-8, 8)
            new_notes.append(NoteEvent(
                pitch=max(MELODY_LOW_BASE, min(MELODY_HIGH_BASE, n.pitch + pitch_shift)),
                start_tick=max(0, new_tick),
                duration_ticks=n.duration_ticks, velocity=n.velocity))
    elif transform == "augment":
        # Double durations (rhythmic stretching)
        extra_tick = 0
        for n in orig_notes:
            new_tick = n.start_tick + tick_delta + extra_tick
            new_dur = n.duration_ticks * 2
            new_notes.append(NoteEvent(
                pitch=max(MELODY_LOW_BASE, min(MELODY_HIGH_BASE, n.pitch + pitch_shift)),
                start_tick=max(0, new_tick),
                duration_ticks=new_dur, velocity=n.velocity))
            extra_tick += n.duration_ticks  # Accumulate the stretch

    if new_notes:
        last = new_notes[-1]
        beat_end = (last.start_tick + last.duration_ticks) / TICKS_PER_QUARTER
        state.pitch = new_notes[-1].pitch
        state.record_pitch(state.pitch)

    return new_notes, max(current_beat + 0.5, beat_end)


# ---------------------------------------------------------------------------
# Tension curve
# ---------------------------------------------------------------------------


class TensionCurve:
    """Maps position (0.0-1.0) to tension (0.0-1.0)."""

    CURVES = {
        "arc": lambda x: (
            math.sin(x / 0.75 * math.pi / 2)
            if x < 0.75
            else math.cos((x - 0.75) / 0.25 * math.pi / 2)
        ),
        "build": lambda x: x ** 0.6,
        "wave": lambda x: 0.5 * x + 0.5 * math.sin(x * 3 * math.pi) * 0.3 + 0.3,
        "plateau": lambda x: (
            min(0.75, x / 0.3 * 0.75) if x < 0.3
            else 0.75 if x < 0.8
            else 0.75 * (1.0 - (x - 0.8) / 0.2)
        ),
        "catharsis": lambda x: (
            0.2 + x / 0.6 * 0.3 if x < 0.6
            else 0.5 + (x - 0.6) / 0.15 * 0.5 if x < 0.75
            else 1.0 - (x - 0.75) / 0.25 * 0.9
        ),
    }

    def __init__(self, curve_name: str = "arc"):
        self.fn = self.CURVES.get(curve_name, self.CURVES["arc"])

    def __call__(self, progress: float) -> float:
        return max(0.0, min(1.0, self.fn(max(0.0, min(1.0, progress)))))


# ---------------------------------------------------------------------------
# MusicParams: interpolated from tension
# ---------------------------------------------------------------------------


@dataclass
class MusicParams:
    """Performance parameters derived from tension level."""
    note_density: float
    chromatic_prob: float
    rest_prob: float
    velocity_base: int
    velocity_range: int
    register_low: int
    register_high: int
    motif_complexity: float


def interpolate_params(tension: float, coltrane: bool = False) -> MusicParams:
    """Interpolate musical parameters from a tension value in [0, 1]."""
    t = max(0.0, min(1.0, tension))
    # Altissimo register in Coltrane mode
    high_extreme = MELODY_HIGH_EXTREME  # default 91 (G6)
    if coltrane:
        if t > 0.9:
            high_extreme = 99   # Eb7 — extreme altissimo
        elif t > 0.8:
            high_extreme = 96   # C7 — altissimo
    return MusicParams(
        note_density=0.9 + t * 2.0,
        chromatic_prob=0.08 + t * 0.35,
        rest_prob=0.42 - t * 0.20,
        velocity_base=int(60 + t * 45),
        velocity_range=int(15 + t * 20),
        register_low=int(MELODY_LOW_BASE - t * (MELODY_LOW_BASE - MELODY_LOW_EXTREME)),
        register_high=int(MELODY_HIGH_BASE + t * (high_extreme - MELODY_HIGH_BASE)),
        motif_complexity=t,
    )


# ---------------------------------------------------------------------------
# Helper functions -- pitch
# ---------------------------------------------------------------------------


def _chord_tones_in_range(root_pc: int, quality: str, low: int, high: int) -> List[int]:
    intervals = CHORD_TONES.get(quality, (0, 4, 7))
    result = []
    for midi_note in range(low, high + 1):
        interval = (midi_note % 12 - root_pc) % 12
        if interval in intervals:
            result.append(midi_note)
    return result


def _scale_tones_in_range(root_pc: int, quality: str, low: int, high: int,
                          tension: float = 0.0) -> List[int]:
    scale_names = CHORD_SCALE_MAP.get(quality, ["ionian"])
    scale_name = scale_names[0]
    # Altered scale on dominants at high tension
    if tension > 0.5 and quality in ("dom7", "7") and random.random() < tension * 0.3:
        scale_name = "altered"
    intervals = SCALES.get(scale_name, SCALES["ionian"])
    interval_set = set(intervals)
    result = []
    for midi_note in range(low, high + 1):
        interval = (midi_note % 12 - root_pc) % 12
        if interval in interval_set:
            result.append(midi_note)
    return result


def _nearest_chord_tone(current_midi: int, root_pc: int, quality: str,
                        low: int, high: int) -> int:
    tones = _chord_tones_in_range(root_pc, quality, low, high)
    if not tones:
        return max(low, min(high, current_midi))
    return min(tones, key=lambda t: abs(t - current_midi))


def _resolve_head_pitch(current_pitch: int, direction: int,
                        root_pc: int, quality: str,
                        low: int, high: int) -> int:
    """Resolve the next head melody pitch by stepping through chord tones.

    direction:  0 = snap to nearest chord tone (repeat/sustain)
               +1 = next chord tone above,  -1 = next below
               +2 = skip one chord tone up,  -2 = skip one down
    """
    tones = _chord_tones_in_range(root_pc, quality, low, high)
    if not tones:
        return max(low, min(high, current_pitch))

    if direction == 0:
        return min(tones, key=lambda t: abs(t - current_pitch))

    if direction > 0:
        above = [t for t in tones if t > current_pitch]
        skip = direction - 1
        if len(above) > skip:
            return above[skip]
        return above[-1] if above else tones[-1]
    else:
        below = [t for t in tones if t < current_pitch]
        below.reverse()  # nearest-first descending
        skip = abs(direction) - 1
        if len(below) > skip:
            return below[skip]
        return below[-1] if below else tones[0]


def _nearest_scale_tone(current_midi: int, root_pc: int, quality: str,
                        low: int, high: int, tension: float = 0.0) -> int:
    tones = _scale_tones_in_range(root_pc, quality, low, high, tension=tension)
    if not tones:
        return max(low, min(high, current_midi))
    return min(tones, key=lambda t: abs(t - current_midi))


def _guide_tones_in_range(root_pc: int, quality: str, low: int, high: int) -> List[int]:
    """Return 3rds and 7ths of the chord within [low, high]."""
    intervals = CHORD_TONES.get(quality, (0, 4, 7))
    guide_intervals = set()
    if len(intervals) > 1:
        guide_intervals.add(intervals[1])
    if len(intervals) > 3:
        guide_intervals.add(intervals[3])
    result = []
    for midi_note in range(low, high + 1):
        interval = (midi_note % 12 - root_pc) % 12
        if interval in guide_intervals:
            result.append(midi_note)
    return result


def _extensions_in_range(root_pc: int, quality: str, low: int, high: int) -> List[int]:
    tension_intervals = TENSIONS.get(quality, [])
    result = []
    for midi_note in range(low, high + 1):
        interval = (midi_note % 12 - root_pc) % 12
        for t in tension_intervals:
            if interval == t % 12:
                result.append(midi_note)
                break
    return result


def _substitute_key_tones(key_center_pc: int, low: int, high: int) -> List[int]:
    sub_keys = [(key_center_pc + 4) % 12, (key_center_pc + 8) % 12]
    result = []
    for sub_key in sub_keys:
        for interval in (0, 4, 7):
            pc = (sub_key + interval) % 12
            for midi_note in range(low, high + 1):
                if midi_note % 12 == pc and midi_note not in result:
                    result.append(midi_note)
    result.sort()
    return result


def _choose_target_pitch(current_pitch: int, root_pc: int, quality: str,
                         low: int, high: int, tension: float,
                         beat_in_bar: float,
                         substitute_tones: Optional[List[int]] = None,
                         target_queue: Optional[List[Tuple[float, int]]] = None,
                         current_beat: float = 0.0,
                         blueprint: Optional['_PhraseBlueprint'] = None,
                         phrase_progress: float = 0.0,
                         state: Optional['_SoloState'] = None) -> int:
    """Choose a target pitch with intervallic variety based on tension."""
    result = _choose_target_pitch_core(
        current_pitch, root_pc, quality, low, high, tension,
        beat_in_bar, substitute_tones, target_queue, current_beat,
        blueprint=blueprint, phrase_progress=phrase_progress,
        state=state)

    # Final interval guard: prevent jumps > 10 semitones (sounds random)
    _MAX_LEAP = 10
    if abs(result - current_pitch) > _MAX_LEAP:
        chord_tones = _chord_tones_in_range(root_pc, quality, low, high)
        close = [t for t in chord_tones
                 if abs(t - current_pitch) <= _MAX_LEAP and t != current_pitch]
        if close:
            # Pick the closest to the intended target direction
            return min(close, key=lambda t: abs(t - result))
    return result


def _choose_target_pitch_core(current_pitch: int, root_pc: int, quality: str,
                               low: int, high: int, tension: float,
                               beat_in_bar: float,
                               substitute_tones: Optional[List[int]] = None,
                               target_queue: Optional[List[Tuple[float, int]]] = None,
                               current_beat: float = 0.0,
                               blueprint: Optional['_PhraseBlueprint'] = None,
                               phrase_progress: float = 0.0,
                               state: Optional['_SoloState'] = None) -> int:
    """Score-based pitch selection — evaluates all candidates simultaneously."""

    # --- Early exit 1: Voice-leading target within 0.5 beats (deterministic) ---
    vl_target_pitch = None
    vl_target_distance = float('inf')
    if target_queue:
        for tgt_beat, tgt_pitch in target_queue:
            if tgt_beat > current_beat + 1e-9:
                vl_target_distance = tgt_beat - current_beat
                vl_target_pitch = max(low, min(high, tgt_pitch))
                if vl_target_distance <= 0.5:
                    return vl_target_pitch
                break

    # --- Early exit 2: Goal pitch forced approach at phrase end ---
    if blueprint and blueprint.goal_pitch and phrase_progress > 0.85:
        goal = blueprint.goal_pitch
        scale_tones = _scale_tones_in_range(root_pc, quality, low, high, tension=tension)
        if scale_tones:
            direction = 1 if goal > current_pitch else -1
            candidates = [s for s in scale_tones
                          if 0 < abs(s - current_pitch) <= 3
                          and (s - current_pitch) * direction > 0]
            if candidates:
                return min(candidates, key=lambda s: abs(s - goal))
        step = 1 if goal > current_pitch else -1
        return max(low, min(high, current_pitch + step))

    # --- Build candidate pool ---
    chord_tones = _chord_tones_in_range(root_pc, quality, low, high)
    if not chord_tones:
        return max(low, min(high, current_pitch))

    scale_tones = _scale_tones_in_range(root_pc, quality, low, high, tension=tension)
    guide_tones_set = set(_guide_tones_in_range(root_pc, quality, low, high))
    chord_tones_set = set(chord_tones)

    pool = set(chord_tones + scale_tones)
    # Add substitute tones if Coltrane mode, early in phrase
    sub_tones_set = set()
    if substitute_tones and phrase_progress < 0.50:
        nearby_subs = [t for t in substitute_tones
                       if low <= t <= high and abs(t - current_pitch) <= 10]
        pool.update(nearby_subs)
        sub_tones_set = set(nearby_subs)

    # Filter: within reach, exclude current pitch
    candidates = [c for c in pool
                  if low <= c <= high and c != current_pitch
                  and abs(c - current_pitch) <= 12]
    if not candidates:
        return min(chord_tones, key=lambda t: abs(t - current_pitch))

    # --- Scoring context ---
    is_strong_beat = beat_in_bar < 0.1 or abs(beat_in_bar - 2.0) < 0.1
    goal_pitch = blueprint.goal_pitch if blueprint else 0
    contour = blueprint.contour if blueprint else ""
    interval_mode = blueprint.interval_mode if blueprint else "stepwise"
    pitch_range = max(1, high - low)
    expected_dir = _contour_direction(contour, phrase_progress) if contour else 0

    # Recent pitches for anti-repetition
    recent = state.recent_pitches[-3:] if state else []
    # Last interval for leap resolution
    last_intervals = state.phrase_interval_history[-2:] if state else []

    # --- Score each candidate ---
    best_score = -999.0
    best_pitch = candidates[0]
    tied = []

    for c in candidates:
        score = 0.0
        interval = c - current_pitch
        abs_interval = abs(interval)

        # 1. Chord tone bonus (+12)
        if c in chord_tones_set:
            score += 12.0

        # 2. Guide tone bonus (+15 on strong beats, +5 otherwise)
        if c in guide_tones_set:
            score += 15.0 if is_strong_beat else 5.0

        # 3. Contour alignment (+10 agree, -5 disagree)
        if expected_dir != 0:
            if interval > 0 and expected_dir > 0:
                score += 10.0
            elif interval < 0 and expected_dir < 0:
                score += 10.0
            elif interval > 0 and expected_dir < 0:
                score -= 5.0
            elif interval < 0 and expected_dir > 0:
                score -= 5.0

        # 4. Goal proximity (+8 * progress)
        if goal_pitch and phrase_progress > 0.1:
            dist_to_goal = abs(c - goal_pitch)
            score += 8.0 * phrase_progress * (1.0 - dist_to_goal / pitch_range)

        # 5. Interval mode fit (+6 preferred, +3 close)
        if interval_mode == "stepwise":
            if 1 <= abs_interval <= 3:
                score += 6.0
            elif 4 <= abs_interval <= 5:
                score += 3.0
        elif interval_mode == "thirds":
            if 3 <= abs_interval <= 5:
                score += 6.0
            elif abs_interval in (1, 2, 6, 7):
                score += 3.0
        elif interval_mode == "wide":
            if 5 <= abs_interval <= 10:
                score += 6.0
            elif 3 <= abs_interval <= 4:
                score += 3.0

        # 6. Voice-leading smoothness (+4, scaled)
        score += 4.0 * max(0.0, 1.0 - abs_interval / 12.0)

        # 7. Register penalty (-8 near extremes)
        if c < low + 3 or c > high - 3:
            score -= 8.0

        # 8. Substitute tone bonus (+7 * tension, Coltrane color)
        if c in sub_tones_set:
            score += 7.0 * tension

        # 9. Leap resolution: Tatum-style mirror leap OR stepwise
        if last_intervals:
            last_iv = last_intervals[-1]
            if abs(last_iv) > 5:
                # Mirror leap: 5-12st opposite direction (+10)
                if 5 <= abs_interval <= 12 and (interval * last_iv) < 0:
                    score += 10.0
                # Stepwise opposite: still rewarded but less (+5)
                elif abs_interval <= 3 and (interval * last_iv) < 0:
                    score += 5.0
            elif len(last_intervals) >= 2:
                # After two same-direction steps: reward contrasting leap
                if (0 < abs(last_intervals[-1]) <= 3
                        and 0 < abs(last_intervals[-2]) <= 3
                        and (last_intervals[-1] > 0) == (last_intervals[-2] > 0)):
                    if abs_interval >= 5 and (interval * last_intervals[-1]) < 0:
                        score += 3.0

        # 10. Anti-repetition (-3 per recent occurrence)
        for rp in recent:
            if c == rp:
                score -= 3.0

        # 11. Voice-leading target pull (if within 2 beats)
        if vl_target_pitch is not None and vl_target_distance <= 2.0:
            proximity = 1.0 - vl_target_distance / 2.0
            target_closeness = max(0.0, 1.0 - abs(c - vl_target_pitch) / 12.0)
            score += 30.0 * proximity * target_closeness

        # 12. Fast harmonic rhythm: chord-tone arpeggiation bias
        hr_speed = state.harmonic_rhythm_speed if state else 0.0
        if hr_speed > 0.1:
            if c in chord_tones_set:
                score += 8.0 * hr_speed
            if c in guide_tones_set:
                score += 6.0 * hr_speed
            # 3rd-based motion bonus (Giant Steps cell: m3/M3)
            if abs_interval in (3, 4):
                score += 10.0 * hr_speed
            elif abs_interval > 7:
                score -= 5.0 * hr_speed

        # 13. Interval anti-repetition: penalize oscillation patterns
        if last_intervals:
            candidate_interval = interval
            last_iv = last_intervals[-1]
            # Exact same interval: -6
            if candidate_interval == last_iv:
                score -= 6.0
            # Same magnitude, any direction (bouncing): -3
            elif abs(candidate_interval) == abs(last_iv) and abs(candidate_interval) >= 3:
                score -= 3.0
            # Oscillation: A, -A, A pattern: -12
            if len(last_intervals) >= 2:
                prev_prev_iv = last_intervals[-2]
                if (candidate_interval == prev_prev_iv
                        and last_iv == -prev_prev_iv
                        and abs(candidate_interval) >= 3):
                    score -= 12.0
                # Near-oscillation (similar magnitudes bouncing): -8
                elif (abs(abs(candidate_interval) - abs(prev_prev_iv)) <= 1
                      and (candidate_interval > 0) == (prev_prev_iv > 0)
                      and (last_iv > 0) != (candidate_interval > 0)
                      and abs(candidate_interval) >= 4):
                    score -= 8.0

        # Track best
        if score > best_score + 0.5:
            best_score = score
            best_pitch = c
            tied = [c]
        elif abs(score - best_score) <= 0.5:
            tied.append(c)

    # Tiebreak: random among tied candidates (the one acceptable random)
    if len(tied) > 1:
        return random.choice(tied)
    return best_pitch


# ---------------------------------------------------------------------------
# Helper functions -- harmony and anticipation
# ---------------------------------------------------------------------------


def _get_chord_at_beat(chords: List[ChordEvent], beat: float) -> Optional[ChordEvent]:
    for chord in chords:
        if chord.start_beat <= beat < chord.end_beat:
            return chord
    if chords:
        return chords[-1]
    return None


def _get_next_chord(chords: List[ChordEvent], beat: float) -> Optional[ChordEvent]:
    """Return the next chord AFTER the current beat position."""
    for chord in chords:
        if chord.start_beat > beat + 1e-9:
            return chord
    return None


def _beats_until_chord_change(chords: List[ChordEvent], beat: float) -> float:
    """Return how many beats until the next chord change."""
    current = _get_chord_at_beat(chords, beat)
    if current is None:
        return float('inf')
    return current.end_beat - beat


def _build_target_queue(current_beat: float, chords: List[ChordEvent],
                        low: int, high: int, current_pitch: int,
                        prefer_third: bool = True) -> List[Tuple[float, int]]:
    """Look ahead 2-4 chord changes and identify guide-tone targets.

    Returns a list of (beat_position, target_midi_pitch) sorted by beat,
    alternating between 3rds and 7ths for smooth voice-leading.
    """
    targets: List[Tuple[float, int]] = []
    scan_beat = current_beat
    ref_pitch = current_pitch
    use_third = prefer_third

    for _ in range(4):
        nxt = _get_next_chord(chords, scan_beat)
        if nxt is None:
            break
        guides = _guide_tones_in_range(nxt.root_pc, nxt.quality, low, high)
        if not guides:
            scan_beat = nxt.start_beat
            continue

        # Separate 3rds and 7ths
        intervals = CHORD_TONES.get(nxt.quality, (0, 4, 7))
        third_int = intervals[1] if len(intervals) > 1 else 4
        seventh_int = intervals[3] if len(intervals) > 3 else None

        thirds = [g for g in guides if (g % 12 - nxt.root_pc) % 12 == third_int]
        sevenths = [g for g in guides
                    if seventh_int is not None
                    and (g % 12 - nxt.root_pc) % 12 == seventh_int]

        # Pick based on alternation preference, falling back to closest guide
        if use_third and thirds:
            chosen = min(thirds, key=lambda t: abs(t - ref_pitch))
        elif not use_third and sevenths:
            chosen = min(sevenths, key=lambda t: abs(t - ref_pitch))
        else:
            chosen = min(guides, key=lambda t: abs(t - ref_pitch))

        targets.append((nxt.start_beat, chosen))
        ref_pitch = chosen
        use_third = not use_third
        scan_beat = nxt.start_beat

    return targets


def _approach_next_chord(current_pitch: int, chords: List[ChordEvent],
                         beat: float, low: int, high: int) -> Optional[int]:
    """If close to a chord change, return a chromatic approach note
    targeting the next chord's 3rd or 7th."""
    beats_left = _beats_until_chord_change(chords, beat)
    if beats_left > 1.0:
        return None
    next_chord = _get_next_chord(chords, beat)
    if next_chord is None:
        return None
    next_guides = _guide_tones_in_range(next_chord.root_pc, next_chord.quality, low, high)
    if not next_guides:
        return None
    target_guide = min(next_guides, key=lambda t: abs(t - current_pitch))
    if current_pitch >= target_guide:
        approach = target_guide + 1
    else:
        approach = target_guide - 1
    if low <= approach <= high:
        return approach
    return None


# ---------------------------------------------------------------------------
# Phrase Blueprint -- pre-planning for within-phrase coherence
# ---------------------------------------------------------------------------


def _plan_phrase_blueprint(chords: List[ChordEvent], phrase_start: float,
                           phrase_end: float, current_pitch: int,
                           low: int, high: int, tension: float,
                           tier: int,
                           intention: Optional[PhraseIntention] = None,
                           last_contour: str = "",
                           harmonic_rhythm_speed: float = 0.0) -> _PhraseBlueprint:
    """Pre-plan a phrase: pick a goal pitch, contour shape, and interval mode.

    The goal pitch is a guide tone (3rd/7th) of the chord active at the last
    strong beat of the phrase — giving the phrase a destination to land on.
    """
    # --- Goal pitch: find guide tone at phrase end ---
    # Last strong beat in phrase (quantize to 2-beat grid)
    last_strong = phrase_start
    b = phrase_start
    while b < phrase_end - 0.5:
        if b % 2.0 < 0.1:
            last_strong = b
        b += 1.0
    # Also consider the beat just before phrase end
    if phrase_end - 1.0 > phrase_start:
        candidate = phrase_end - 1.0
        if candidate % 2.0 < 0.5:
            last_strong = max(last_strong, candidate)

    end_chord = _get_chord_at_beat(chords, max(phrase_start, last_strong))
    goal_pitch = current_pitch  # fallback
    if end_chord is not None:
        guides = _guide_tones_in_range(end_chord.root_pc, end_chord.quality, low, high)
        if guides:
            goal_pitch = min(guides, key=lambda t: abs(t - current_pitch))

    # --- Contour: map from intention character ---
    if intention is not None:
        char = intention.character
        contour_map = {
            PC_QUESTION: "ascending",
            PC_ANSWER: "descending",
            PC_EXCLAMATION: "arch",
            PC_WHISPER: "pendulum",
            PC_CONTINUATION: last_contour if last_contour else "ascending",
            PC_CONTRAST: _opposite_contour(last_contour),
            PC_SILENCE: "descending",
        }
        contour = contour_map.get(char, "arch")
    else:
        contour = random.choice(["arch", "ascending", "descending"])

    # --- Interval mode: from tier + tension + harmonic rhythm ---
    if harmonic_rhythm_speed > 0.3:
        interval_mode = "thirds"        # arpeggiation on fast changes
    elif tier >= 2 and tension > 0.7:
        interval_mode = "wide"
    elif tension < 0.4:
        interval_mode = "stepwise"
    else:
        interval_mode = "thirds"

    return _PhraseBlueprint(
        goal_pitch=goal_pitch,
        contour=contour,
        interval_mode=interval_mode,
    )


def _opposite_contour(contour: str) -> str:
    """Return the opposite contour shape for contrast phrases."""
    opposites = {
        "ascending": "descending", "descending": "ascending",
        "arch": "valley", "valley": "arch",
        "pendulum": "arch",
    }
    return opposites.get(contour, "arch")


def _contour_direction(contour: str, progress: float) -> int:
    """Return expected melodic direction from contour and phrase progress.

    +1 = ascending, -1 = descending, 0 = neutral.
    """
    if contour == "ascending":
        return +1 if progress < 0.80 else -1
    elif contour == "descending":
        return -1
    elif contour == "arch":
        return +1 if progress < 0.50 else -1
    elif contour == "valley":
        return -1 if progress < 0.40 else +1
    elif contour == "pendulum":
        return +1 if int(progress * 10) % 2 == 0 else -1
    return 0


def _detect_cell(phrase_intervals: List[int], blueprint: _PhraseBlueprint) -> None:
    """After 3+ notes, check if the last 3 intervals form an interesting melodic cell.

    An 'interesting' cell has at least 2 distinct interval sizes and at least
    one interval > 1 semitone. Stores it in blueprint.active_cell if found.
    """
    if blueprint.active_cell is not None:
        return  # Already have a cell for this phrase
    if len(phrase_intervals) < 3:
        return

    last3 = phrase_intervals[-3:]
    # Must have at least 2 distinct absolute interval sizes
    abs_intervals = [abs(i) for i in last3]
    if len(set(abs_intervals)) < 2:
        return
    # Must have at least one interval > 1 semitone
    if max(abs_intervals) <= 1:
        return
    # Not all zeros
    if all(i == 0 for i in last3):
        return

    blueprint.active_cell = list(last3)
    blueprint.cell_uses = 0


def _apply_cell_sequence(blueprint: _PhraseBlueprint, current_pitch: int,
                         root_pc: int, quality: str,
                         low: int, high: int,
                         tension: float) -> Optional[List[int]]:
    """If a melodic cell is active and hasn't been overused, sequence it.

    Applies the stored interval pattern from the current pitch, snapping
    each resulting note to the nearest scale tone. Returns a list of pitches
    or None if no cell is active / already used enough.
    """
    if blueprint.active_cell is None:
        return None
    if blueprint.cell_uses >= 2:
        blueprint.active_cell = None  # Let new ideas emerge
        return None

    scale_tones = _scale_tones_in_range(root_pc, quality, low, high, tension=tension)
    if not scale_tones:
        return None

    pitches = []
    p = current_pitch
    for interval in blueprint.active_cell:
        p = p + interval
        p = max(low, min(high, p))
        # Snap to nearest scale tone
        nearest = min(scale_tones, key=lambda s: abs(s - p))
        pitches.append(nearest)
        p = nearest

    blueprint.cell_uses += 1
    return pitches


# ---------------------------------------------------------------------------
# Helper functions -- timing and rhythm
# ---------------------------------------------------------------------------


def _beat_to_tick(beat: float) -> int:
    return int(beat * TICKS_PER_QUARTER)


def _apply_swing(tick: int, swing_ratio: float = 0.667) -> int:
    depth = _current_feel.swing_depth if _current_feel else 1.0
    beat_pos = tick % TICKS_PER_QUARTER
    beat_start = tick - beat_pos
    if beat_pos >= TICKS_PER_8TH:
        swing_point = int(TICKS_PER_QUARTER * swing_ratio)
        offset_within_offbeat = beat_pos - TICKS_PER_8TH
        remaining = TICKS_PER_QUARTER - swing_point
        if TICKS_PER_8TH > 0:
            scaled_offset = int(offset_within_offbeat * remaining / TICKS_PER_8TH)
        else:
            scaled_offset = 0
        result = beat_start + swing_point + scaled_offset
    else:
        swing_point = int(TICKS_PER_QUARTER * swing_ratio)
        if TICKS_PER_8TH > 0:
            scaled = int(beat_pos * swing_point / TICKS_PER_8TH)
        else:
            scaled = 0
        result = beat_start + scaled
    # Scale swing displacement by depth
    straight = beat_start + beat_pos
    result = straight + int((result - straight) * depth)
    # Apply push/pull offset
    if _current_feel and _current_feel.offset_bias != 0.0:
        result += int(_current_feel.offset_bias * 15)
    return result


def _humanize(tick: int, amount: int = 10, beat: float = -1.0) -> int:
    """Add timing variation with Tatum-style downbeat/upbeat bias.

    Downbeats pulled ahead (crisp), upbeats pushed behind (lazy).
    """
    spread = _current_feel.timing_spread if _current_feel else 1.0
    scaled = max(1, int(amount * spread))
    base_jitter = random.randint(-scaled, scaled)
    if beat >= 0.0:
        frac = beat % 1.0
        if frac < 0.15 or frac > 0.85:
            bias = -random.randint(3, 8)   # downbeat: pull ahead
        elif 0.35 < frac < 0.65:
            bias = random.randint(5, 15)   # upbeat: push behind
        else:
            bias = 0
        return max(0, tick + base_jitter + bias)
    return max(0, tick + base_jitter)


# ---------------------------------------------------------------------------
# Context-Aware Decision Functions (Solo Narrative Planner v6)
# ---------------------------------------------------------------------------

def _choose_phrase_length(tier: int, tension: float,
                          harmonic_rhythm_fast: bool,
                          state: '_SoloState',
                          intention: Optional[PhraseIntention] = None) -> float:
    """Deterministic phrase length selection based on context.

    Uses intention hint as base, with alternating jitter and context-driven
    adjustments instead of random choice.
    """
    # Base from intention or tier defaults
    if intention is not None:
        base = intention.phrase_beats_hint
        # Wire density_bias into phrase length
        base -= intention.density_bias * 1.5
    else:
        tier_defaults = {1: 6.0, 2: 5.0, 3: 4.0}
        base = tier_defaults.get(tier, 5.0)

    # Anti-repetition: deterministic alternation instead of random jitter
    hist = state.phrase_length_history
    if len(hist) >= 2 and abs(hist[-1] - hist[-2]) < 1.0:
        jitter = 1.5 if state.phrase_count % 2 == 0 else -1.5
        base += jitter

    # Context-driven adjustments
    if state.beats_since_last_silence > 8:
        base *= 0.85  # Shorten after long run
    elif state.beats_since_last_silence == 0.0 and state.phrase_count > 0:
        base *= 1.15  # Stretch after silence
    if not state.last_phrase_was_resolved:
        base *= 0.85  # Shorter answer to unresolved tension

    # Character shaping
    if intention is not None:
        if intention.character == PC_EXCLAMATION:
            progress = state.phrase_count / max(1, 20)
            base += progress * 2.0
        elif intention.character == PC_WHISPER:
            base = min(base, 3.0)

    # Harmonic rhythm: longer phrases when chords move fast (continuous playing)
    if harmonic_rhythm_fast:
        base *= 1.5

    # Climax phrases can exceed normal tier max (Tatum marathon phrasing)
    climax_stretch = 0.0
    if intention is not None and intention.character == PC_EXCLAMATION:
        progress = state.phrase_count / max(1, 20)
        if progress > 0.6:  # late in solo = climax territory
            climax_stretch = 4.0 * (progress - 0.6) / 0.4  # up to +4 beats

    # Clamp to tier maximums
    tier_max = {1: 16.0, 2: 10.0, 3: 6.0}
    return max(1.5, min(base, tier_max.get(tier, 8.0) + climax_stretch))


def _choose_context_rhythmic_cell(tension: float,
                                  state: Optional['_SoloState'] = None,
                                  intention: Optional[PhraseIntention] = None) -> Tuple[float, ...]:
    """Deterministic rhythmic cell selection with anti-repetition.

    Uses tension + density_bias to pick pool, then selects highest-contrast
    cell vs. recent history.
    """
    # Effective tension shifted by narrative density_bias
    eff = tension + (intention.density_bias * 0.3 if intention else 0.0)
    eff = max(0.0, min(1.0, eff))

    # Deterministic pool selection based on effective tension
    if eff < 0.30:
        pool = RHYTHMIC_CELLS_SPARSE
    elif eff < 0.55:
        pool = RHYTHMIC_CELLS_MEDIUM
    elif eff < 0.80:
        pool = RHYTHMIC_CELLS_TRIPLET
    else:
        pool = RHYTHMIC_CELLS_DENSE

    if state is None or not state.phrase_cell_history:
        cell = random.choice(pool)
        if state is not None:
            state.phrase_cell_history.append(cell)
        return cell

    last_cell = state.phrase_cell_history[-1]
    # Pick highest-contrast cell deterministically
    best_cell = pool[0]
    best_weight = -1.0
    tied = []
    for c in pool:
        if c == last_cell:
            w = 0.1
        else:
            dur_diff = abs(sum(c) - sum(last_cell))
            w = 0.5 + dur_diff
        if w > best_weight + 0.01:
            best_weight = w
            best_cell = c
            tied = [c]
        elif abs(w - best_weight) <= 0.01:
            tied.append(c)

    cell = random.choice(tied) if len(tied) > 1 else best_cell
    state.phrase_cell_history.append(cell)
    if len(state.phrase_cell_history) > 6:
        state.phrase_cell_history = state.phrase_cell_history[-6:]
    return cell


def _choose_strategy(strategies: List[str], base_weights: List[float],
                     state: '_SoloState',
                     intention: Optional[PhraseIntention] = None) -> str:
    """Context-aware strategy selection with recency penalty.

    Reduces weight of recently used strategies by 40% per recent use (last 3).
    Boosts last-used strategy 2x when character is "continuation".
    """
    weights = list(base_weights)

    # Recency penalty: reduce weight for recently used strategies
    recent = state.strategy_history[-3:] if state.strategy_history else []
    for i, s in enumerate(strategies):
        penalty_count = recent.count(s)
        if penalty_count > 0:
            weights[i] *= 0.6 ** penalty_count

    # Continuation bonus: boost last-used strategy
    if (intention is not None and intention.character == PC_CONTINUATION
            and state.strategy_history):
        last_strat = state.strategy_history[-1]
        for i, s in enumerate(strategies):
            if s == last_strat:
                weights[i] *= 2.0

    # Intention override
    if intention is not None and intention.strategy_weights:
        for i, s in enumerate(strategies):
            if s in intention.strategy_weights:
                weights[i] = intention.strategy_weights[s]

    # Ensure at least minimal weight
    weights = [max(0.01, w) for w in weights]

    chosen = random.choices(strategies, weights=weights, k=1)[0]
    state.record_strategy(chosen)
    return chosen


def _choose_direction(state: '_SoloState',
                      intention: Optional[PhraseIntention] = None,
                      blueprint: Optional['_PhraseBlueprint'] = None,
                      phrase_progress: float = 0.0) -> int:
    """Deterministic melodic direction from context.

    Priority: register extremes → monotonic run reversal → blueprint contour →
    narrative bias → momentum continuation → default ascending.
    Returns +1 (ascending) or -1 (descending).
    """
    mid_register = (MELODY_LOW_BASE + MELODY_HIGH_BASE) / 2.0

    # 1. Register extreme override (highest priority)
    if state.register_ema > mid_register + 12:
        return -1
    if state.register_ema < mid_register - 12:
        return +1

    # 2. Monotonic run > 6 notes: force reversal
    if _contour_check(state.recent_pitches, max_monotonic=6):
        return -1 if state.direction_momentum > 0 else +1

    # 3. Blueprint contour direction
    if blueprint and blueprint.contour:
        cd = _contour_direction(blueprint.contour, phrase_progress)
        if cd != 0:
            return cd

    # 4. Narrative direction bias
    if intention is not None and abs(intention.direction_bias) > 0.1:
        return +1 if intention.direction_bias > 0 else -1

    # 5. Momentum continuation
    if abs(state.direction_momentum) > 0.3:
        return +1 if state.direction_momentum > 0 else -1

    # 6. Default
    return +1


def _should_insert_strategic_silence(state: '_SoloState', tension: float,
                                     intention: Optional[PhraseIntention] = None,
                                     harmonic_rhythm_fast: bool = False) -> bool:
    """Decide whether to insert strategic silence before the next phrase.

    Considers: intention character, silence starvation, tension level,
    harmonic rhythm speed.
    """
    # PC_SILENCE intention: always insert
    if intention is not None and intention.character == PC_SILENCE:
        return True

    # Starvation pressure: probability rises after 16+ beats without silence
    starvation = state.beats_since_last_silence
    if starvation < 12.0:
        starvation_prob = 0.0
    elif starvation < 32.0:
        starvation_prob = (starvation - 12.0) / 40.0  # 0→0.5 over 12-32 beats
    else:
        starvation_prob = 0.5

    # Tension suppression: high tension reduces silence probability
    tension_mod = -0.30 * tension

    # Fast harmonic rhythm suppression: keep playing through changes
    harmony_mod = -0.40 if harmonic_rhythm_fast else 0.0

    # Base probability from intention
    if intention is not None:
        char_probs = {
            PC_WHISPER: 0.25, PC_QUESTION: 0.10, PC_ANSWER: 0.05,
            PC_EXCLAMATION: 0.02, PC_CONTINUATION: 0.05,
            PC_CONTRAST: 0.15, PC_SILENCE: 1.0,
        }
        base = char_probs.get(intention.character, 0.08)
    else:
        base = 0.08

    # Density contrast: high probability after sustained dense playing
    if (state.beats_since_last_silence > 20.0
            and state.phrase_count > 8
            and tension > 0.5):
        density_prob = 0.30
    else:
        density_prob = 0.0

    prob = max(0.0, min(0.8, base + starvation_prob + tension_mod + harmony_mod + density_prob))
    return random.random() < prob


def _post_phrase_rest(state: '_SoloState', tension: float, tier: int,
                      intention: Optional[PhraseIntention] = None,
                      params: Optional[dict] = None,
                      harmonic_rhythm_fast: bool = False) -> float:
    """Determine rest duration after a phrase.

    Uses intention.silence_after as base when available, with anti-repetition.
    Falls back to tier-based logic.
    """
    if intention is not None:
        base_rest = intention.silence_after
    else:
        # Tier-based defaults
        tier_rests = {1: 2.0, 2: 1.5, 3: 0.5}
        base_rest = tier_rests.get(tier, 1.0)
        # Reduce rest at high tension
        base_rest *= max(0.3, 1.0 - tension * 0.6)

    # Anti-repetition: alternating jitter when about to repeat rest duration
    if (len(state.phrase_length_history) >= 2 and
            abs(state.last_silence_duration - base_rest) < 0.3):
        base_rest += 0.5 if (state.phrase_count % 2 == 0) else -0.25

    # Fast harmonic rhythm: very short breaths to maintain continuity
    if harmonic_rhythm_fast:
        base_rest *= 0.3

    rest = max(0.25, base_rest)
    state.last_silence_duration = rest
    state.beats_since_last_silence = 0.0
    return rest


def _tier3_register_shift(state: '_SoloState', low: int, high: int,
                           intention: Optional[PhraseIntention] = None) -> Optional[int]:
    """Decide whether to do a register shift in tier 3 phrases.

    Returns new pitch if shift should happen, None otherwise.
    Uses deterministic period gating and target-seeking magnitude.
    """
    # Period-based gating from character
    if intention is not None:
        char_periods = {
            PC_EXCLAMATION: 4, PC_WHISPER: 20, PC_CONTRAST: 5,
        }
        period = char_periods.get(intention.character, 7)
    else:
        period = 7

    if state.phrase_count % period != 0:
        return None

    center = (low + high) / 2.0

    # Direction toward narrative register target
    if intention is not None:
        target_pitch = low + intention.register_target * (high - low)
        dist = abs(target_pitch - state.pitch)
        magnitude = max(8, min(14, int(dist * 0.8)))
        if state.pitch < target_pitch - 5:
            return min(high, state.pitch + magnitude)
        elif state.pitch > target_pitch + 5:
            return max(low, state.pitch - magnitude)

    # Default: counter-momentum toward register center
    dist_to_center = abs(state.register_ema - center)
    magnitude = max(8, min(14, int(dist_to_center * 0.5)))
    if state.direction_momentum <= 0:
        return min(high, state.pitch + magnitude)
    else:
        return max(low, state.pitch - magnitude)


# ---------------------------------------------------------------------------
# Question-Answer Phrasing (Solo Narrative Planner v6)
# ---------------------------------------------------------------------------

def _record_phrase_ending(state: '_SoloState', phrase_notes: List[NoteEvent],
                          chords: List[ChordEvent], beat: float):
    """Record how a phrase ended for question-answer continuity."""
    if not phrase_notes:
        return
    last_note = phrase_notes[-1]
    state.last_phrase_ending_pitch = last_note.pitch

    # Compute last interval
    if len(phrase_notes) >= 2:
        state.last_phrase_ending_interval = (
            phrase_notes[-1].pitch - phrase_notes[-2].pitch)
    else:
        state.last_phrase_ending_interval = 0

    # Check resolution: did we end on a chord tone?
    chord = _get_chord_at_beat(chords, beat)
    if chord is not None:
        end_pc = last_note.pitch % 12
        chord_tone_pcs = set()
        for interval in CHORD_TONES.get(chord.quality, [0, 4, 7]):
            chord_tone_pcs.add((chord.root_pc + interval) % 12)
        state.last_phrase_was_resolved = end_pc in chord_tone_pcs
    else:
        state.last_phrase_was_resolved = True


def _qa_opening_adjustment(state: '_SoloState',
                           intention: Optional[PhraseIntention],
                           current_pitch: int,
                           low: int, high: int) -> Optional[int]:
    """Suggest a starting pitch for the new phrase based on how the last one ended.

    Returns adjusted pitch or None to keep current.
    """
    if state.phrase_count < 1:
        return None  # No history yet

    ending = state.last_phrase_ending_pitch
    if ending == 0:
        return None

    if intention is None:
        return None

    char = intention.character
    last_interval = state.last_phrase_ending_interval

    # Previous unresolved + current "answer" → step toward resolution
    if not state.last_phrase_was_resolved and char == PC_ANSWER:
        # Step opposite to the unresolved leap direction
        nudge = -1 if last_interval > 0 else (1 if last_interval < 0 else 0)
        return max(low, min(high, ending + nudge))

    # Previous big ascending leap → answer phrases start nearby and descend
    if last_interval > 5 and char == PC_ANSWER:
        nudge = -min(3, abs(last_interval) // 3)
        return max(low, min(high, ending + nudge))

    # Previous big descending leap → exclamation phrases leap back up
    if last_interval < -5 and char == PC_EXCLAMATION:
        leap_back = min(10, max(4, abs(last_interval)))
        return max(low, min(high, ending + leap_back))

    # Continuation: continue in direction of momentum
    if char == PC_CONTINUATION:
        nudge = 1 if state.direction_momentum >= 0 else -1
        return max(low, min(high, ending + nudge))

    # Contrast: jump toward the opposite register extreme
    if char == PC_CONTRAST:
        center = (low + high) // 2
        dist = abs(ending - center)
        jump = max(10, min(18, dist + 4))
        if ending > center:
            return max(low, ending - jump)
        else:
            return min(high, ending + jump)

    return None


# ---------------------------------------------------------------------------
# Within-Phrase Note Coherence (Solo Narrative Planner v6)
# ---------------------------------------------------------------------------


def _run_exit_pitch(state: '_SoloState', current_pitch: int,
                    root_pc: int, quality: str,
                    low: int, high: int, tension: float) -> Optional[int]:
    """Choose the first note after a run ends, based on Tatum run-exit data.

    35% bounce (leap 5-10st opposite), 29% turn (step 1-3st opposite),
    36% continue (step 1-4st same direction).
    Returns None if no run just ended.
    """
    if state.last_run_direction is None:
        return None

    direction = state.last_run_direction
    state.last_run_direction = None  # consume: one-shot

    scale_tones = _scale_tones_in_range(root_pc, quality, low, high, tension=tension)
    chord_tones = _chord_tones_in_range(root_pc, quality, low, high)
    all_tones = sorted(set(scale_tones + chord_tones))

    r = random.random()
    if r < 0.35:
        # Bounce: leap 5-10st opposite direction
        opp = -direction
        candidates = [t for t in all_tones
                      if 5 <= abs(t - current_pitch) <= 10
                      and (t - current_pitch) * opp > 0]
    elif r < 0.64:
        # Turn: step 1-3st opposite direction
        opp = -direction
        candidates = [t for t in all_tones
                      if 1 <= abs(t - current_pitch) <= 3
                      and (t - current_pitch) * opp > 0]
    else:
        # Continue: step 1-4st same direction
        candidates = [t for t in all_tones
                      if 1 <= abs(t - current_pitch) <= 4
                      and (t - current_pitch) * direction > 0]

    if candidates:
        return min(candidates, key=lambda t: abs(t - current_pitch))
    return None


def _apply_leap_resolution(state: '_SoloState', pitch: int,
                           scale_tones: List[int]) -> int:
    """After a large leap, resolve with mirror leap (60%) or stepwise (40%).

    Tatum data: leap → LEAP → leap_back is the #1 pattern.
    60% pass through to let scoring choose mirror leap.
    40% force stepwise opposite (original behavior).
    """
    hist = state.phrase_interval_history
    if not hist:
        return pitch

    last_interval = hist[-1]

    # Large leap: probabilistic resolution (allows mirror leaps)
    if abs(last_interval) > 5:
        # 60% let scoring handle it (mirror leap can win with +10)
        if random.random() < 0.60:
            return pitch
        # 40% force stepwise opposite
        direction = -1 if last_interval > 0 else 1
        candidates = [s for s in scale_tones
                      if 0 < abs(s - pitch) <= 3 and (s - pitch) * direction > 0]
        if candidates:
            return min(candidates, key=lambda s: abs(s - pitch))

    # Two consecutive steps same direction → always contrast
    if len(hist) >= 2:
        if (0 < abs(hist[-1]) <= 3 and 0 < abs(hist[-2]) <= 3 and
                (hist[-1] > 0) == (hist[-2] > 0)):
            direction = -1 if hist[-1] > 0 else 1
            candidates = [s for s in scale_tones
                          if 5 <= abs(s - pitch) <= 10 and (s - pitch) * direction > 0]
            if candidates:
                # Pick middle candidate deterministically
                return candidates[len(candidates) // 2]

    return pitch


# ---------------------------------------------------------------------------
# Tatum solo dyad punctuation
# ---------------------------------------------------------------------------

# Common dyad intervals below melody: minor 3rd, major 3rd, minor 6th, major 6th
_DYAD_INTERVALS_BELOW = (3, 4, 8, 9)


def _tatum_dyad_below(melody_pitch: int, root_pc: int, quality: str,
                       low: int) -> Optional[int]:
    """Find the best dyad tone below a melody note, Tatum-style.

    Picks from 3rd, 4th, 8th, or 9th below — whichever is closest to
    a chord tone of the current harmony.
    """
    chord_intervals = set(CHORD_TONES.get(quality, (0, 4, 7)))
    best_pitch = None
    best_distance = 999

    for interval_below in _DYAD_INTERVALS_BELOW:
        candidate = melody_pitch - interval_below
        if candidate < low:
            continue
        candidate_pc = (candidate % 12 - root_pc) % 12
        distance = min(abs(candidate_pc - ct) for ct in chord_intervals) if chord_intervals else 99
        if distance < best_distance:
            best_distance = distance
            best_pitch = candidate

    # Only return if reasonably harmonic (within 1 semitone of a chord tone)
    if best_pitch is not None and best_distance <= 1:
        return best_pitch
    return None


def _shape_phrase_velocity(base_velocity: int, phrase_progress: float,
                           intention: Optional[PhraseIntention] = None) -> int:
    """Shape velocity within a phrase based on character.

    Answer → decrescendo, Exclamation → crescendo, Whisper → flat soft.
    """
    if intention is None:
        return base_velocity

    char = intention.character
    offset = intention.velocity_offset

    if char == PC_ANSWER:
        # Decrescendo through phrase
        vel_mod = int(-15 * phrase_progress)
    elif char == PC_EXCLAMATION:
        # Crescendo through phrase
        vel_mod = int(15 * phrase_progress)
    elif char == PC_WHISPER:
        # Flat and soft
        vel_mod = -15
    elif char == PC_QUESTION:
        # Slight rise then fall
        if phrase_progress < 0.6:
            vel_mod = int(8 * (phrase_progress / 0.6))
        else:
            vel_mod = int(8 * (1.0 - phrase_progress) / 0.4)
    else:
        # Accent-start (Tatum default): strong attack, settling through phrase
        # +12 at start → 0 at 25% → -5 at end
        if phrase_progress < 0.25:
            vel_mod = int(12 * (1.0 - phrase_progress / 0.25))
        else:
            vel_mod = int(-5 * (phrase_progress - 0.25) / 0.75)

    return max(1, min(127, base_velocity + vel_mod + offset))


def _record_note_context(state: '_SoloState', pitch: int, prev_pitch: int):
    """Record interval and update momentum/register EMAs after each note."""
    interval = pitch - prev_pitch
    state.phrase_interval_history.append(interval)
    state.update_direction_momentum(interval)
    state.update_register_ema(pitch)


def _choose_rhythmic_cell(tension: float) -> Tuple[float, ...]:
    """Select a rhythmic cell appropriate for the current tension level.

    Piano-jazz calibrated: sparse/medium cells dominate.  Dense (16th-note)
    cells are rare even at high tension so the solo breathes.
    """
    roll = random.random()
    if tension < 0.3:
        pool = RHYTHMIC_CELLS_SPARSE if roll > 0.10 else RHYTHMIC_CELLS_MEDIUM
    elif tension < 0.65:
        # Mostly sparse/medium; very occasional dense lick
        if roll < 0.30:
            pool = RHYTHMIC_CELLS_SPARSE
        elif roll < 0.92:
            pool = RHYTHMIC_CELLS_MEDIUM
        else:
            pool = RHYTHMIC_CELLS_DENSE
    else:
        # High tension: medium dominates, some dense/triplet colour
        if roll < 0.15:
            pool = RHYTHMIC_CELLS_SPARSE
        elif roll < 0.60:
            pool = RHYTHMIC_CELLS_MEDIUM
        elif roll < 0.80:
            pool = RHYTHMIC_CELLS_TRIPLET
        else:
            pool = RHYTHMIC_CELLS_DENSE
    return random.choice(pool)


def _choose_velocity(params: MusicParams, beat_in_bar: float = 0.0,
                     intention: Optional[PhraseIntention] = None,
                     beats_per_bar: int = 4) -> int:
    """Beat-position velocity with accent pattern + tiny humanization jitter.

    Downbeat: +8, mid-bar accent: +4, secondary beats: -2, offbeats: -6.
    """
    base = params.velocity_base

    # Beat-position accent pattern (generalized for any meter)
    beat_frac = beat_in_bar % float(beats_per_bar)
    mid_bar = beats_per_bar // 2
    if beat_frac < 0.1:                        # Downbeat (beat 1)
        base += 8
    elif abs(beat_frac - float(mid_bar)) < 0.5:  # Mid-bar accent
        base += 4
    elif beat_frac > beats_per_bar - 0.5:      # Last beat (pickup)
        base -= 2
    elif abs(beat_frac - round(beat_frac)) < 0.15:  # On-beat
        base -= 2
    else:                                      # Offbeats
        base -= 6

    # Narrative velocity offset
    if intention is not None:
        base += intention.velocity_offset

    # Tiny humanization jitter (±3 instead of full range)
    base += random.randint(-3, 3)
    return max(1, min(127, base))


# ---------------------------------------------------------------------------
# Contour and melodic helpers
# ---------------------------------------------------------------------------


def _contour_check(recent_pitches: List[int], max_monotonic: int = 6) -> bool:
    if len(recent_pitches) < max_monotonic:
        return False
    tail = recent_pitches[-max_monotonic:]
    ascending = all(tail[i] <= tail[i + 1] for i in range(len(tail) - 1))
    descending = all(tail[i] >= tail[i + 1] for i in range(len(tail) - 1))
    return ascending or descending


def _chromatic_enclosure(target_midi: int, from_above: bool = True) -> List[Tuple[int, float]]:
    if from_above:
        return [(target_midi + 1, 0.25), (target_midi - 1, 0.25), (target_midi, 0.5)]
    else:
        return [(target_midi - 1, 0.25), (target_midi + 1, 0.25), (target_midi, 0.5)]


def _choose_run_timing() -> str:
    """Select run timing profile matching Tatum data: 40% decel, 25% accel, 35% even."""
    r = random.random()
    if r < 0.40:
        return "decelerate"
    elif r < 0.65:
        return "accelerate"
    return "even"


def _run_durations(length: int, timing: str,
                   dur_start: float, dur_end: float, dur_even: float) -> List[float]:
    """Compute per-note durations for a run with timing variation."""
    if timing == "decelerate":
        return [dur_start + (dur_end - dur_start) * i / max(1, length - 1)
                for i in range(length)]
    elif timing == "accelerate":
        return [dur_end + (dur_start - dur_end) * i / max(1, length - 1)
                for i in range(length)]
    return [dur_even] * length


def _scalar_run(current_midi: int, direction: int, root_pc: int, quality: str,
                length: int, low: int, high: int,
                tension: float = 0.0,
                timing: str = "even") -> List[Tuple[int, float]]:
    scale_tones = _scale_tones_in_range(root_pc, quality, low, high, tension=tension)
    if not scale_tones:
        return [(current_midi, 0.30)]
    start = min(scale_tones, key=lambda t: abs(t - current_midi))
    idx = scale_tones.index(start)
    durations = _run_durations(length, timing, 0.20, 0.40, 0.30)
    result = []
    for i in range(length):
        if 0 <= idx < len(scale_tones):
            result.append((scale_tones[idx], durations[i]))
            idx += direction
        else:
            break
    if not result:
        result.append((start, 0.30))
    return result


def _arpeggio_run(current_midi: int, direction: int, root_pc: int, quality: str,
                  low: int, high: int) -> List[Tuple[int, float]]:
    chord_notes = _chord_tones_in_range(root_pc, quality, low, high)
    if not chord_notes:
        return [(current_midi, 0.5)]
    start = min(chord_notes, key=lambda t: abs(t - current_midi))
    idx = chord_notes.index(start)
    result = []
    max_notes = random.randint(3, 6)
    for _ in range(max_notes):
        if 0 <= idx < len(chord_notes):
            result.append((chord_notes[idx], 0.5))
            idx += direction
        else:
            break
    if not result:
        result.append((start, 0.5))
    return result


def _digital_pattern_fragment(root_pc: int, current_midi: int,
                              low: int, high: int) -> List[Tuple[int, float]]:
    pattern = random.choice(DIGITAL_PATTERNS)
    root_candidates = [n for n in range(low, high + 1) if n % 12 == root_pc]
    if not root_candidates:
        return [(current_midi, 0.5)]
    root_midi = min(root_candidates, key=lambda r: abs(r - current_midi))
    result = []
    for interval in pattern:
        note = root_midi + interval
        clamped = max(low, min(high, note))
        result.append((clamped, 0.5))
    return result


def _chromatic_run(current_midi: int, low: int, high: int,
                   length: int = 0, direction: int = 0,
                   timing: str = "even") -> List[Tuple[int, float]]:
    """Generate a pure chromatic run ignoring harmony — true sheets of sound.

    Runs chromatically for `length` notes, reversing at register boundaries.
    Timing varies: decelerate (cascade settling), accelerate, or even.
    """
    if length <= 0:
        length = 8
    if direction == 0:
        direction = 1 if current_midi < (low + high) // 2 else -1
    durations = _run_durations(length, timing, 0.15, 0.30, 0.22)
    result: List[Tuple[int, float]] = []
    pitch = current_midi
    for i in range(length):
        pitch += direction
        if pitch > high:
            direction = -1
            pitch = high - 1
        elif pitch < low:
            direction = 1
            pitch = low + 1
        result.append((max(low, min(high, pitch)), durations[i]))
    return result


# ---------------------------------------------------------------------------
# Run Builder (v13) -- backward-planned, forward-executed Tatum runs
# ---------------------------------------------------------------------------


def _build_run_notes(blueprint: RunBlueprint,
                     chords: List[ChordEvent],
                     state: '_SoloState',
                     params: 'MusicParams',
                     swing: bool) -> Tuple[List[NoteEvent], float]:
    """Build a pre-planned run from a RunBlueprint into NoteEvents.

    Core algorithm: backward planning (pitch sequence from target), forward execution.
    Returns (notes, new_beat_position).
    """
    low = params.register_low
    high = params.register_high

    # Climax runs: temporarily widen register for Tatum-style keyboard sweeps
    if blueprint.purpose == "climax" and blueprint.intensity > 0.7:
        low = max(36, low - 12)    # extend down one octave (to C2)
        high = min(96, high + 5)   # extend up slightly (to C7)

    source_pitch = state.pitch
    target_pitch = max(low, min(high, blueprint.target_pitch))
    available_beats = blueprint.end_beat - blueprint.start_beat
    if available_beats < 0.5:
        blueprint.executed = True
        return [], blueprint.end_beat

    # --- Note count based on available time and run speed ---
    is_fill = blueprint.purpose == "fill"
    if is_fill:
        avg_dur = 0.30  # 8th-note-ish fills
    else:
        avg_dur = 0.22  # 16th-note-ish runs
    max_notes = 36 if blueprint.purpose == "climax" else 24
    note_count = max(6, min(max_notes, int(available_beats / avg_dur)))

    # --- Build pitch sequence backward from target ---
    pitches = _build_run_pitch_sequence(
        blueprint, note_count, source_pitch, target_pitch,
        low, high, chords)

    # --- Durations with timing profile ---
    if is_fill:
        durations = _run_durations(len(pitches), blueprint.timing_profile,
                                   0.25, 0.50, 0.30)
    else:
        durations = _run_durations(len(pitches), blueprint.timing_profile,
                                   0.15, 0.35, 0.22)

    # --- Velocity arc: crescendo through body, decrescendo last 2-3 notes ---
    vel_base = params.velocity_base + 5
    vel_range = params.velocity_range
    velocities = []
    n = len(pitches)
    landing_zone = max(2, n // 8)  # last ~12% for decrescendo
    for i in range(n):
        if i < n - landing_zone:
            # Crescendo through body
            body_progress = i / max(1, n - landing_zone - 1)
            vel = vel_base + int(body_progress * vel_range * 0.6)
        else:
            # Decrescendo for graceful landing
            landing_progress = (i - (n - landing_zone)) / max(1, landing_zone - 1)
            vel = vel_base + int(vel_range * 0.6) - int(landing_progress * vel_range * 0.3)
        velocities.append(max(30, min(120, vel)))

    # --- Emit NoteEvents ---
    notes: List[NoteEvent] = []
    beat = blueprint.start_beat
    for i, (pitch, dur, vel) in enumerate(zip(pitches, durations, velocities)):
        if beat >= blueprint.end_beat:
            break
        tick = _beat_to_tick(beat)
        if swing:
            tick = _apply_swing(tick)
        tick = _humanize(tick, amount=5, beat=beat)
        dur_ticks = max(1, int(dur * TICKS_PER_QUARTER * 0.85))  # slight staccato
        notes.append(NoteEvent(
            pitch=max(low, min(high, pitch)),
            start_tick=max(0, tick),
            duration_ticks=dur_ticks,
            velocity=vel,
            channel=0,
        ))

        # Two-handed dyads: every other note when enabled
        if blueprint.two_handed and i % 2 == 0:
            ch = _get_chord_at_beat(chords, beat)
            if ch:
                dyad = _tatum_dyad_below(pitch, ch.root_pc, ch.quality, low)
                if dyad is not None:
                    notes.append(NoteEvent(
                        pitch=dyad,
                        start_tick=max(0, tick),
                        duration_ticks=dur_ticks,
                        velocity=max(25, vel - 10),
                        channel=0,
                    ))

        beat += dur

    # --- Post-run state updates ---
    if notes:
        state.pitch = notes[-1].pitch
        state.record_pitch(state.pitch)
        # Set run direction for run-exit behavior
        if len(pitches) >= 2:
            state.last_run_direction = 1 if pitches[-1] > pitches[-2] else -1

    blueprint.executed = True
    state.runs_executed += 1
    state.last_run_end_beat = beat
    final_beat = max(blueprint.end_beat, beat)
    return notes, final_beat


def _build_run_pitch_sequence(blueprint: RunBlueprint,
                              note_count: int,
                              source_pitch: int,
                              target_pitch: int,
                              low: int, high: int,
                              chords: List[ChordEvent]) -> List[int]:
    """Build pitch sequence for a run, backward from target, then reverse.

    Threads through scale materials, uses extensions every 4-5th note,
    respects contour shape.
    """
    materials = blueprint.scale_materials
    if not materials:
        # Fallback: chromatic between source and target
        direction = 1 if target_pitch > source_pitch else -1
        return list(range(source_pitch, target_pitch + direction, direction))[:note_count]

    # Build full scale pool per material segment
    # Each segment: (beat_boundary, root_pc, quality, scale_tones_asc)
    segments = []
    for i, (beat_b, root_pc, quality) in enumerate(materials):
        next_beat = materials[i + 1][0] if i + 1 < len(materials) else blueprint.end_beat
        tones = _scale_tones_in_range(root_pc, quality, low, high)
        if not tones:
            tones = list(range(low, high + 1))
        # Add tension tones (9ths, 11ths, 13ths)
        tension_intervals = TENSIONS.get(quality, ())
        tension_tones = set()
        for midi_note in range(low, high + 1):
            interval = (midi_note % 12 - root_pc) % 12
            if interval in tension_intervals:
                tension_tones.add(midi_note)
        segments.append({
            "beat_start": beat_b,
            "beat_end": next_beat,
            "root_pc": root_pc,
            "quality": quality,
            "scale": sorted(set(tones)),
            "tensions": sorted(tension_tones),
        })

    # Distribute notes across segments proportionally by beat duration
    total_dur = blueprint.end_beat - blueprint.start_beat
    notes_per_seg = []
    remaining_notes = note_count
    for i, seg in enumerate(segments):
        seg_dur = seg["beat_end"] - seg["beat_start"]
        if i == len(segments) - 1:
            notes_per_seg.append(remaining_notes)
        else:
            n = max(1, int(note_count * seg_dur / max(0.01, total_dur)))
            n = min(n, remaining_notes - (len(segments) - i - 1))
            notes_per_seg.append(max(1, n))
            remaining_notes -= notes_per_seg[-1]

    # Build backward from target, then reverse
    backward_pitches = [target_pitch]
    current = target_pitch

    # Determine contour direction for backward traversal
    if blueprint.contour in ("sweep_up", "cascade"):
        backward_dir = -1  # going backward from top means descending = going down
    elif blueprint.contour == "sweep_down":
        backward_dir = 1   # going backward from bottom means ascending
    elif blueprint.contour == "arc_up_down":
        backward_dir = 1   # second half descending, backward = ascending
    elif blueprint.contour == "arc_down_up":
        backward_dir = -1  # second half ascending, backward = descending
    else:
        backward_dir = -1 if target_pitch > source_pitch else 1

    # Walk backward through segments (reversed)
    extension_counter = 0
    for seg_idx in range(len(segments) - 1, -1, -1):
        seg = segments[seg_idx]
        seg_notes = notes_per_seg[seg_idx]
        scale = seg["scale"]
        tensions = seg["tensions"]

        if not scale:
            continue

        for _ in range(seg_notes - (1 if seg_idx == len(segments) - 1 else 0)):
            if len(backward_pitches) >= note_count:
                break

            extension_counter += 1
            # Extension threading: every 4-5th note, use tension tone
            use_extension = (extension_counter % 5 == 0
                             and tensions
                             and blueprint.intensity > 0.3)

            if use_extension:
                # Find nearest tension tone in the right direction
                candidates = [t for t in tensions
                              if (backward_dir < 0 and t < current)
                              or (backward_dir > 0 and t > current)]
                if candidates:
                    next_p = min(candidates, key=lambda t: abs(t - current))
                else:
                    next_p = min(tensions, key=lambda t: abs(t - current))
            else:
                # Step through scale
                candidates = [t for t in scale
                              if (backward_dir < 0 and t < current)
                              or (backward_dir > 0 and t > current)]
                if candidates:
                    # Pick the nearest step
                    next_p = min(candidates, key=lambda t: abs(t - current))
                else:
                    # Ran out of scale in this direction — wrap/reverse
                    next_p = min(scale, key=lambda t: abs(t - current))
                    if next_p == current and len(scale) > 1:
                        backward_dir = -backward_dir
                        candidates = [t for t in scale
                                      if (backward_dir < 0 and t < current)
                                      or (backward_dir > 0 and t > current)]
                        if candidates:
                            next_p = min(candidates, key=lambda t: abs(t - current))

            # Transition zone: at segment boundaries, allow chromatic passing
            if seg_idx > 0 and len(backward_pitches) >= note_count - 3:
                # Within last 3 notes approaching segment boundary: chromatic step OK
                step = -1 if backward_dir < 0 else 1
                chromatic_candidate = current + step
                if low <= chromatic_candidate <= high:
                    next_p = chromatic_candidate

            current = max(low, min(high, next_p))
            backward_pitches.append(current)

    # Reverse to get forward sequence
    pitches = list(reversed(backward_pitches))

    # Apply contour shaping (arc / cascade adjustments)
    if blueprint.contour == "arc_up_down" and len(pitches) > 4:
        # Peak at 55-65% of run
        peak_idx = int(len(pitches) * 0.6)
        peak_pitch = max(pitches[:peak_idx + 1]) if peak_idx > 0 else pitches[0]
        # Ensure the peak is actually the highest point
        for i in range(peak_idx + 1, len(pitches)):
            if pitches[i] > peak_pitch:
                pitches[i] = max(low, pitches[i] - (pitches[i] - peak_pitch))
    elif blueprint.contour == "cascade" and len(pitches) > 4:
        # Sweep up to 75%, rapid descent 25%
        turn_idx = int(len(pitches) * 0.75)
        # After turn, force descending
        for i in range(turn_idx + 1, len(pitches)):
            if i > 0 and pitches[i] >= pitches[i - 1]:
                pitches[i] = max(low, pitches[i - 1] - 1)

    # Ensure we start near source pitch: adjust first few pitches
    if pitches and abs(pitches[0] - source_pitch) > 12:
        # Smooth entry: interpolate first 3 notes toward source
        blend_count = min(3, len(pitches))
        for i in range(blend_count):
            alpha = i / max(1, blend_count)
            pitches[i] = int(source_pitch + alpha * (pitches[i] - source_pitch))

    return pitches[:note_count]


def _select_tier(tension: float, phrase_count: int = 0) -> int:
    """Select generation tier with deterministic cycling near boundaries.

    Piano-jazz calibrated: favours space (tier 1) across most of the
    tension range.  Tier 3 (sheets-of-sound) only appears above 0.8.
    In transition zones, cycles between tiers based on phrase_count.
    """
    if tension < 0.45:
        return 1
    elif tension < 0.65:
        tier2_prob = (tension - 0.45) / 0.20  # 0.0 at 0.45, 1.0 at 0.65
        period = max(1, int(1.0 / max(0.05, tier2_prob)))
        return 2 if (phrase_count % period == 0) else 1
    elif tension < 0.80:
        return 2
    elif tension < 0.92:
        tier3_prob = (tension - 0.80) / 0.12
        period = max(1, int(1.0 / max(0.05, tier3_prob)))
        return 3 if (phrase_count % period == 0) else 2
    else:
        return 3


# ---------------------------------------------------------------------------
# Tier 1: Melodic -- guide tones, voice leading, syncopated rhythm
# ---------------------------------------------------------------------------


def _generate_tier1_phrase(current_beat: float, phrase_beats: float,
                           chords: List[ChordEvent], params: MusicParams,
                           swing: bool, state: _SoloState,
                           tension: float,
                           intention: Optional[PhraseIntention] = None) -> Tuple[List[NoteEvent], float]:
    notes = []
    beat = current_beat
    phrase_end = current_beat + phrase_beats
    note_count = 0

    low = params.register_low
    high = params.register_high

    # Pre-plan phrase blueprint for coherent destination + contour
    blueprint = _plan_phrase_blueprint(
        chords, current_beat, phrase_end, state.pitch,
        low, high, tension, tier=1, intention=intention,
        last_contour=getattr(state, '_last_contour', ""),
        harmonic_rhythm_speed=state.harmonic_rhythm_speed)
    state._last_contour = blueprint.contour

    while beat < phrase_end - 1e-9:
        beat_before = beat  # Stall guard

        chord = _get_chord_at_beat(chords, beat)
        if chord is None:
            beat += 1.0
            continue

        low = params.register_low
        high = params.register_high

        # Context-aware rhythmic cell selection
        cell = _choose_context_rhythmic_cell(tension, state, intention)

        for dur in cell:
            if beat + dur > phrase_end + 1e-9:
                # Try to fit a truncated note
                remaining = phrase_end - beat
                if remaining < 0.125:
                    break
                dur = remaining

            prev_pitch = state.pitch
            phrase_progress = (beat - current_beat) / max(0.1, phrase_beats)

            # Melodic cell sequencing: if a cell is active, use it
            cell_pitches = _apply_cell_sequence(
                blueprint, state.pitch, chord.root_pc, chord.quality,
                low, high, tension)
            if cell_pitches:
                # Play the sequenced cell
                for cp in cell_pitches:
                    if beat + dur > phrase_end + 1e-9:
                        remaining = phrase_end - beat
                        if remaining < 0.125:
                            break
                        dur = remaining
                    tick = _beat_to_tick(beat)
                    if swing:
                        tick = _apply_swing(tick)
                    tick = _humanize(tick, beat=beat)
                    dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                    vel = _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention)
                    vel = _shape_phrase_velocity(vel, phrase_progress, intention)
                    notes.append(NoteEvent(
                        pitch=cp, start_tick=tick,
                        duration_ticks=dur_ticks, velocity=vel,
                    ))
                    _record_note_context(state, cp, state.pitch)
                    state.pitch = cp
                    state.record_pitch(cp)
                    beat += dur
                    note_count += 1
                    phrase_progress = (beat - current_beat) / max(0.1, phrase_beats)
                    if beat >= phrase_end - 1e-9:
                        break
                    # Get next dur from cell
                    cell = _choose_context_rhythmic_cell(tension, state, intention)
                    dur = cell[0]
                break  # Exit the cell for-loop after sequencing

            # Harmonic anticipation: use when near chord change and past mid-phrase
            approach = _approach_next_chord(state.pitch, chords, beat, low, high)
            near_change = _beats_until_chord_change(chords, beat) < 1.5
            if approach is not None and near_change and phrase_progress > 0.4:
                target = approach
            else:
                target = _choose_target_pitch(
                    state.pitch, chord.root_pc, chord.quality,
                    low, high, tension, beat % float(state.beats_per_bar),
                    substitute_tones=getattr(state, 'substitute_tones', None),
                    target_queue=state.target_queue, current_beat=beat,
                    blueprint=blueprint, phrase_progress=phrase_progress,
                    state=state)

            # Leap resolution: resolve large leaps stepwise
            scale_tones = _scale_tones_in_range(chord.root_pc, chord.quality, low, high, tension=tension)
            target = _apply_leap_resolution(state, target, scale_tones)

            # Context-aware contour direction (deterministic)
            direction = _choose_direction(state, intention, blueprint, phrase_progress)
            if _contour_check(state.recent_pitches):
                candidates = _chord_tones_in_range(chord.root_pc, chord.quality, low, high)
                shifted = [t for t in candidates
                           if (t > state.pitch) == (direction > 0)]
                if shifted:
                    target = shifted[0] if direction > 0 else shifted[-1]

            tick = _beat_to_tick(beat)
            if swing:
                tick = _apply_swing(tick)
            tick = _humanize(tick, beat=beat)
            dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
            vel = _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention)

            # Phrase-internal velocity shaping
            vel = _shape_phrase_velocity(vel, phrase_progress, intention)

            notes.append(NoteEvent(
                pitch=target, start_tick=tick,
                duration_ticks=dur_ticks, velocity=vel,
            ))

            # Tatum dyad punctuation: ~13% of sustained notes get a chord tone below
            if dur >= 0.5 and vel > 50 and random.random() < 0.13:
                dyad_pitch = _tatum_dyad_below(target, chord.root_pc, chord.quality,
                                                params.register_low)
                if dyad_pitch is not None:
                    notes.append(NoteEvent(
                        pitch=dyad_pitch, start_tick=tick,
                        duration_ticks=dur_ticks,
                        velocity=max(1, vel - 10),
                        channel=0,
                    ))

            # Record context for next note
            _record_note_context(state, target, prev_pitch)
            state.pitch = target
            state.record_pitch(target)
            beat += dur
            note_count += 1

            # Detect melodic cells after 3+ notes
            _detect_cell(state.phrase_interval_history, blueprint)

            if beat >= phrase_end - 1e-9:
                break

        # Stall guard: if no progress was made, advance to end
        if beat <= beat_before + 1e-9:
            beat = phrase_end

    return notes, beat


# ---------------------------------------------------------------------------
# Tier 2: Motivic Development -- motifs, enclosures, runs, syncopation
# ---------------------------------------------------------------------------


def _generate_tier2_phrase(current_beat: float, phrase_beats: float,
                           chords: List[ChordEvent], params: MusicParams,
                           swing: bool, coltrane: bool,
                           state: _SoloState,
                           tension: float,
                           intention: Optional[PhraseIntention] = None) -> Tuple[List[NoteEvent], float]:
    notes = []
    beat = current_beat
    phrase_end = current_beat + phrase_beats

    # Motif development: get the next transformation of the active motif
    motif = state.advance_motif(
        tension,
        motif_action=intention.motif_action if intention else "free")

    # Apply motif if complexity allows and it fits
    if params.motif_complexity > 0.2 and beat + sum(motif.durations) <= phrase_end + 1e-9:
        chord = _get_chord_at_beat(chords, beat)
        if chord is not None:
            low = params.register_low
            high = params.register_high
            motif_start = _nearest_chord_tone(state.pitch, chord.root_pc, chord.quality, low, high)

            for iv, dur in zip(motif.intervals, motif.durations):
                if beat + dur > phrase_end + 1e-9:
                    break
                note_midi = max(low, min(high, motif_start + iv))
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick, beat=beat)
                dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                vel = _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention)
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                state.pitch = note_midi
                state.record_pitch(note_midi)
                beat += dur

    # Continue with connective strategies
    strategies = ["enclosure", "scalar_run", "arpeggio", "digital_pattern", "direct"]
    weights = [0.25, 0.20, 0.15, 0.20, 0.20]
    if coltrane:
        weights = [0.15, 0.20, 0.15, 0.35, 0.15]

    low = params.register_low
    high = params.register_high

    # Pre-plan phrase blueprint for coherent destination
    blueprint = _plan_phrase_blueprint(
        chords, current_beat, phrase_end, state.pitch,
        low, high, tension, tier=2, intention=intention,
        last_contour=getattr(state, '_last_contour', ""),
        harmonic_rhythm_speed=state.harmonic_rhythm_speed)
    state._last_contour = blueprint.contour

    # Lock strategy for the entire phrase (Improvement 4)
    strategy = _choose_strategy(strategies, weights, state, intention)

    while beat < phrase_end - 1e-9:
        beat_before = beat  # Stall guard

        chord = _get_chord_at_beat(chords, beat)
        if chord is None:
            beat += 0.5
            continue

        low = params.register_low
        high = params.register_high
        phrase_progress = (beat - current_beat) / max(0.1, phrase_beats)

        # Run exit behavior: if a run just ended, choose exit pitch first
        run_exit = _run_exit_pitch(state, state.pitch, chord.root_pc, chord.quality,
                                   low, high, tension)
        if run_exit is not None:
            cell = _choose_context_rhythmic_cell(tension, state, intention)
            dur = cell[0]
            if beat + dur <= phrase_end + 1e-9:
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick, beat=beat)
                dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                vel = _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention)
                notes.append(NoteEvent(
                    pitch=run_exit, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                _record_note_context(state, run_exit, state.pitch)
                state.pitch = run_exit
                state.record_pitch(run_exit)
                beat += dur
            continue

        # Chromatic passing tone insertion
        if random.random() < params.chromatic_prob and notes:
            approach = _approach_next_chord(state.pitch, chords, beat, low, high)
            if approach is not None:
                cdur = 0.25
                if beat + cdur <= phrase_end + 1e-9:
                    tick = _beat_to_tick(beat)
                    if swing:
                        tick = _apply_swing(tick)
                    tick = _humanize(tick, beat=beat)
                    notes.append(NoteEvent(
                        pitch=approach, start_tick=tick,
                        duration_ticks=max(30, int(cdur * TICKS_PER_QUARTER)),
                        velocity=max(1, _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention) - 10),
                    ))
                    state.pitch = approach
                    state.record_pitch(approach)
                    beat += cdur
                    continue

        if strategy == "enclosure":
            # Target next chord's guide tone if near a change
            approach = _approach_next_chord(state.pitch, chords, beat, low, high)
            if approach is not None:
                target = approach
            else:
                target = _choose_target_pitch(
                    state.pitch, chord.root_pc, chord.quality,
                    low, high, tension, beat % float(state.beats_per_bar),
                    substitute_tones=getattr(state, 'substitute_tones', None),
                    target_queue=state.target_queue, current_beat=beat,
                    blueprint=blueprint, phrase_progress=phrase_progress,
                    state=state)
            enc_notes = _chromatic_enclosure(target, from_above=(state.pitch > target))
            for note_midi, dur in enc_notes:
                if beat + dur > phrase_end + 1e-9:
                    break
                note_midi = max(low, min(high, note_midi))
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick, beat=beat)
                dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                vel = _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention)
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                state.pitch = note_midi
                state.record_pitch(note_midi)
                beat += dur

        elif strategy == "scalar_run":
            direction = _choose_direction(state, intention, blueprint, phrase_progress)
            remaining = phrase_end - beat
            length = max(3, min(9, int(remaining / 0.3)))
            timing = _choose_run_timing()
            run = _scalar_run(state.pitch, direction, chord.root_pc, chord.quality, length, low, high, tension=tension, timing=timing)
            run_emitted = 0
            for note_midi, dur in run:
                if beat + dur > phrase_end + 1e-9:
                    break
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick, beat=beat)
                dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                vel = _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention)
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                state.pitch = note_midi
                state.record_pitch(note_midi)
                beat += dur
                run_emitted += 1
            if run_emitted >= 3:
                state.last_run_direction = direction

        elif strategy == "arpeggio":
            direction = _choose_direction(state, intention, blueprint, phrase_progress)
            run = _arpeggio_run(state.pitch, direction, chord.root_pc, chord.quality, low, high)
            run_emitted = 0
            for note_midi, dur in run:
                if beat + dur > phrase_end + 1e-9:
                    break
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick, beat=beat)
                dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                vel = _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention)
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                state.pitch = note_midi
                state.record_pitch(note_midi)
                beat += dur
                run_emitted += 1
            if run_emitted >= 3:
                state.last_run_direction = direction

        elif strategy == "digital_pattern":
            fragment = _digital_pattern_fragment(chord.root_pc, state.pitch, low, high)
            for note_midi, dur in fragment:
                if beat + dur > phrase_end + 1e-9:
                    break
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick, beat=beat)
                dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                vel = _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention)
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                state.pitch = note_midi
                state.record_pitch(note_midi)
                beat += dur

        elif strategy == "direct":
            target = _choose_target_pitch(
                state.pitch, chord.root_pc, chord.quality,
                low, high, tension, beat % float(state.beats_per_bar),
                substitute_tones=getattr(state, 'substitute_tones', None),
                target_queue=state.target_queue, current_beat=beat,
                blueprint=blueprint, phrase_progress=phrase_progress,
                state=state)
            cell = _choose_context_rhythmic_cell(tension, state, intention)
            dur = cell[0]
            if beat + dur > phrase_end + 1e-9:
                dur = phrase_end - beat
                if dur < 0.125:
                    beat = phrase_end
                    break
            tick = _beat_to_tick(beat)
            if swing:
                tick = _apply_swing(tick)
            tick = _humanize(tick, beat=beat)
            dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
            vel = _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention)
            notes.append(NoteEvent(
                pitch=target, start_tick=tick,
                duration_ticks=dur_ticks, velocity=vel,
            ))

            # Tatum dyad punctuation
            if dur >= 0.5 and vel > 50 and random.random() < 0.13:
                dyad_pitch = _tatum_dyad_below(target, chord.root_pc, chord.quality,
                                                low)
                if dyad_pitch is not None:
                    notes.append(NoteEvent(
                        pitch=dyad_pitch, start_tick=tick,
                        duration_ticks=dur_ticks,
                        velocity=max(1, vel - 10),
                        channel=0,
                    ))

            state.pitch = target
            state.record_pitch(target)
            beat += dur

        # Stall guard: if no progress was made, advance to end
        if beat <= beat_before + 1e-9:
            beat = phrase_end

    return notes, beat


# ---------------------------------------------------------------------------
# Tier 3: Sheets of Sound -- rapid notes, register shifts, digital patterns
# ---------------------------------------------------------------------------


def _generate_tier3_phrase(current_beat: float, phrase_beats: float,
                           chords: List[ChordEvent], params: MusicParams,
                           swing: bool, coltrane: bool,
                           state: _SoloState,
                           tension: float,
                           intention: Optional[PhraseIntention] = None) -> Tuple[List[NoteEvent], float]:
    notes = []
    beat = current_beat
    phrase_end = current_beat + phrase_beats

    strategies = ["digital", "rapid_arpeggio", "enclosure_run", "chromatic_run"]
    weights = [0.25, 0.20, 0.30, 0.25]
    if coltrane:
        weights = [0.30, 0.20, 0.15, 0.35]

    low = params.register_low
    high = params.register_high

    # Pre-plan phrase blueprint (lighter touch for tier 3 — goal pitch only)
    blueprint = _plan_phrase_blueprint(
        chords, current_beat, phrase_end, state.pitch,
        low, high, tension, tier=3, intention=intention,
        last_contour=getattr(state, '_last_contour', ""),
        harmonic_rhythm_speed=state.harmonic_rhythm_speed)
    state._last_contour = blueprint.contour

    vel_base = max(90, params.velocity_base)
    vel_range = params.velocity_range

    while beat < phrase_end - 1e-9:
        # Guard: need at least a 16th note
        if phrase_end - beat < 0.25 - 1e-9:
            break

        beat_before = beat  # Stall guard
        phrase_progress = (beat - current_beat) / max(0.1, phrase_beats)

        chord = _get_chord_at_beat(chords, beat)
        if chord is None:
            beat += 0.25
            continue

        low = params.register_low
        high = params.register_high

        # Run exit behavior: if a run just ended, choose exit pitch first
        run_exit = _run_exit_pitch(state, state.pitch, chord.root_pc, chord.quality,
                                   low, high, tension)
        if run_exit is not None:
            dur = 0.25  # 16th note in tier 3
            if beat + dur <= phrase_end + 1e-9:
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick, amount=5, beat=beat)
                dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                vel = _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention)
                notes.append(NoteEvent(
                    pitch=run_exit, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                state.pitch = run_exit
                state.record_pitch(run_exit)
                beat += dur
            continue

        # Context-aware register shift
        reg_shift = _tier3_register_shift(state, low, high, intention)
        if reg_shift is not None:
            state.pitch = reg_shift

        # Context-aware rhythmic cell and strategy selection
        cell = _choose_context_rhythmic_cell(tension, state, intention)
        strategy = _choose_strategy(strategies, weights, state, intention)

        if strategy == "digital":
            fragment = _digital_pattern_fragment(chord.root_pc, state.pitch, low, high)
            for (note_midi, _), dur in zip(fragment, cell):
                if beat + dur > phrase_end + 1e-9:
                    break
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick, amount=5, beat=beat)
                dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                vel = _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention)
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                state.pitch = note_midi
                state.record_pitch(note_midi)
                beat += dur

        elif strategy == "rapid_arpeggio":
            chord_notes = _chord_tones_in_range(chord.root_pc, chord.quality, low, high)
            ext_notes = _extensions_in_range(chord.root_pc, chord.quality, low, high)
            all_tones = sorted(set(chord_notes + ext_notes))
            if not all_tones:
                all_tones = [state.pitch]
            start_note = min(all_tones, key=lambda t: abs(t - state.pitch))
            idx = all_tones.index(start_note)
            direction = _choose_direction(state, intention, blueprint, phrase_progress)
            remaining = phrase_end - beat
            avg_dur = sum(cell) / len(cell) if cell else 0.25
            run_length = max(4, min(8, int(remaining / avg_dur)))
            ra_emitted = 0
            for i in range(run_length):
                dur = cell[i % len(cell)]
                if beat + dur > phrase_end + 1e-9:
                    break
                if 0 <= idx < len(all_tones):
                    note_midi = all_tones[idx]
                else:
                    direction = -direction
                    idx = max(0, min(len(all_tones) - 1, idx))
                    note_midi = all_tones[idx]
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick, amount=5, beat=beat)
                dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                vel = _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention)
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                state.pitch = note_midi
                state.record_pitch(note_midi)
                beat += dur
                idx += direction
                ra_emitted += 1
            if ra_emitted >= 3:
                state.last_run_direction = direction

        elif strategy == "enclosure_run":
            chord_notes = _chord_tones_in_range(chord.root_pc, chord.quality, low, high)
            if not chord_notes:
                chord_notes = [state.pitch]
            targets = sorted(chord_notes, key=lambda t: abs(t - state.pitch))[:3]
            # Reverse order on odd phrases for variety (nearest-first vs farthest-first)
            if state.phrase_count % 2 == 1:
                targets = list(reversed(targets))
            for target in targets:
                if beat >= phrase_end - 1e-9:
                    break
                enc_pitches = [target + 1, target - 1, target]
                for j, ep in enumerate(enc_pitches):
                    dur = cell[j % len(cell)] if j < len(cell) else 0.25
                    if beat + dur > phrase_end + 1e-9:
                        break
                    ep_clamped = max(low, min(high, ep))
                    tick = _beat_to_tick(beat)
                    if swing:
                        tick = _apply_swing(tick)
                    tick = _humanize(tick, amount=5, beat=beat)
                    dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                    vel = _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention)
                    notes.append(NoteEvent(
                        pitch=ep_clamped, start_tick=tick,
                        duration_ticks=dur_ticks, velocity=vel,
                    ))
                    state.pitch = ep_clamped
                    state.record_pitch(ep_clamped)
                    beat += dur

        elif strategy == "chromatic_run":
            cr_dir = _choose_direction(state, intention, blueprint, phrase_progress)
            cr_timing = _choose_run_timing()
            run = _chromatic_run(state.pitch, low, high, direction=cr_dir, timing=cr_timing)
            cr_emitted = 0
            for note_midi, dur in run:
                if beat + dur > phrase_end + 1e-9:
                    break
                tick = _beat_to_tick(beat)
                if swing:
                    tick = _apply_swing(tick)
                tick = _humanize(tick, amount=5, beat=beat)
                dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
                vel = _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention)
                notes.append(NoteEvent(
                    pitch=note_midi, start_tick=tick,
                    duration_ticks=dur_ticks, velocity=vel,
                ))
                state.pitch = note_midi
                state.record_pitch(note_midi)
                beat += dur
                cr_emitted += 1
            if cr_emitted >= 3:
                state.last_run_direction = cr_dir

        # Stall guard: if no progress was made, advance to end
        if beat <= beat_before + 1e-9:
            beat = phrase_end

    return notes, beat


# ---------------------------------------------------------------------------
# Extended phrase generators
# ---------------------------------------------------------------------------


def _generate_pentatonic_super_phrase(current_beat: float, phrase_beats: float,
                                      chords: List[ChordEvent], params: MusicParams,
                                      swing: bool, coltrane: bool,
                                      state: _SoloState,
                                      tension: float,
                                      intention: Optional[PhraseIntention] = None,
) -> Tuple[List[NoteEvent], float]:
    notes = []
    beat = current_beat
    phrase_end = current_beat + phrase_beats

    chord = _get_chord_at_beat(chords, beat)
    if chord is None:
        return notes, beat

    low = params.register_low
    high = params.register_high

    root = chord.root_pc
    q = chord.quality
    if q in ("maj7", "maj", "6"):
        penta_root = (root + 2) % 12
    elif q in ("dom7", "7"):
        penta_root = (root + (10 if tension > 0.5 else 2)) % 12  # blues at high tension
    elif q in ("min7", "min", "min6"):
        penta_root = (root + 3) % 12
    else:
        penta_root = root

    penta_intervals = (0, 2, 4, 7, 9)
    penta_notes = [m for m in range(low, high + 1)
                   if (m % 12 - penta_root) % 12 in penta_intervals]

    if not penta_notes:
        return _generate_tier2_phrase(current_beat, phrase_beats, chords,
                                      params, swing, coltrane, state, tension)

    start_note = min(penta_notes, key=lambda t: abs(t - state.pitch))
    idx = penta_notes.index(start_note)
    direction = _choose_direction(state, intention)

    remaining = phrase_end - beat
    num_notes = max(6, min(10, int(remaining / 0.4)))
    for i in range(num_notes):
        if beat >= phrase_end - 1e-9:
            break
        if 0 <= idx < len(penta_notes):
            note_midi = penta_notes[idx]
        else:
            direction = -direction
            idx = max(0, min(len(penta_notes) - 1, idx))
            note_midi = penta_notes[idx]

        cell = _choose_rhythmic_cell(tension)
        dur = cell[0]
        if beat + dur > phrase_end + 1e-9:
            remaining = phrase_end - beat
            if remaining < 0.125:
                break
            dur = remaining

        tick = _beat_to_tick(beat)
        if swing:
            tick = _apply_swing(tick)
        tick = _humanize(tick, beat=beat)
        dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
        vel = _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention)
        notes.append(NoteEvent(
            pitch=note_midi, start_tick=tick,
            duration_ticks=dur_ticks, velocity=vel,
        ))
        state.pitch = note_midi
        state.record_pitch(note_midi)
        beat += dur
        idx += direction

    return notes, beat


def _generate_call_response_phrase(current_beat: float, phrase_beats: float,
                                    chords: List[ChordEvent], params: MusicParams,
                                    swing: bool, coltrane: bool,
                                    state: _SoloState,
                                    tension: float,
                                    intention: Optional[PhraseIntention] = None) -> Tuple[List[NoteEvent], float]:
    notes = []
    beat = current_beat
    phrase_end = current_beat + phrase_beats
    half_beats = phrase_beats / 2.0

    # --- Call ---
    call_notes_data = []
    call_end = beat + half_beats
    call_note_count = max(3, min(6, int(half_beats / 0.6)))
    call_played = 0

    while beat < call_end - 1e-9 and call_played < call_note_count:
        chord = _get_chord_at_beat(chords, beat)
        if chord is None:
            beat += 0.5
            continue

        low = params.register_low
        high = params.register_high
        target = _choose_target_pitch(
            state.pitch, chord.root_pc, chord.quality,
            low, high, tension, beat % float(state.beats_per_bar),
            substitute_tones=getattr(state, 'substitute_tones', None),
            target_queue=state.target_queue, current_beat=beat,
            state=state)

        cell = _choose_rhythmic_cell(tension)
        dur = cell[0]
        if beat + dur > call_end + 1e-9:
            remaining = call_end - beat
            if remaining < 0.125:
                break
            dur = remaining

        tick = _beat_to_tick(beat)
        if swing:
            tick = _apply_swing(tick)
        tick = _humanize(tick, beat=beat)
        dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
        vel = _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention)
        notes.append(NoteEvent(
            pitch=target, start_tick=tick,
            duration_ticks=dur_ticks, velocity=vel,
        ))
        call_notes_data.append((target, dur))
        state.pitch = target
        state.record_pitch(target)
        beat += dur
        call_played += 1

    # --- Response ---
    if len(call_notes_data) < 2:
        return notes, beat

    call_pitches = [p for p, d in call_notes_data]
    call_durs = [d for p, d in call_notes_data]
    call_intervals = [call_pitches[i + 1] - call_pitches[i]
                      for i in range(len(call_pitches) - 1)]

    response_end = min(beat + half_beats, phrase_end)

    # Cycle through transforms: transpose → invert → echo
    cycle = state.phrase_count % 3
    if cycle == 0:
        # Transpose: counter-momentum, magnitude scaled by tension
        base_shift = 3 + int(tension * 2)
        shift = -base_shift if state.direction_momentum >= 0 else base_shift
        resp_pitches = [max(MELODY_LOW, min(MELODY_HIGH, p + shift)) for p in call_pitches]
        resp_durs = list(call_durs)
    elif cycle == 1:
        # Invert: mirror the intervals
        inverted_intervals = [-iv for iv in call_intervals]
        resp_pitches = [call_pitches[0]]
        for iv in inverted_intervals:
            resp_pitches.append(max(MELODY_LOW, min(MELODY_HIGH, resp_pitches[-1] + iv)))
        resp_durs = list(call_durs)
    else:
        # Echo: exact repeat with delay
        resp_pitches = list(call_pitches)
        resp_durs = list(call_durs)
        beat += 0.5

    for resp_pitch, dur in zip(resp_pitches, resp_durs):
        if beat >= response_end - 1e-9:
            break
        if beat + dur > response_end + 1e-9:
            remaining = response_end - beat
            if remaining < 0.125:
                break
            dur = remaining
        tick = _beat_to_tick(beat)
        if swing:
            tick = _apply_swing(tick)
        tick = _humanize(tick, beat=beat)
        dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
        vel = _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention)
        notes.append(NoteEvent(
            pitch=resp_pitch, start_tick=tick,
            duration_ticks=dur_ticks, velocity=vel,
        ))
        state.pitch = resp_pitch
        state.record_pitch(resp_pitch)
        beat += dur

    return notes, beat


def _generate_triplet_phrase(current_beat: float, phrase_beats: float,
                              chords: List[ChordEvent], params: MusicParams,
                              swing: bool, coltrane: bool,
                              state: _SoloState,
                              tension: float,
                              intention: Optional[PhraseIntention] = None) -> Tuple[List[NoteEvent], float]:
    notes = []
    beat = current_beat
    phrase_end = current_beat + phrase_beats
    triplet_dur_ticks = TICKS_PER_QUARTER // 3
    triplet_dur_beats = 1.0 / 3.0
    remaining = phrase_end - beat
    num_groups = max(2, min(4, int(remaining / 1.2)))

    for group_idx in range(num_groups):
        if beat >= phrase_end - 1e-9:
            break
        chord = _get_chord_at_beat(chords, beat)
        if chord is None:
            beat += 1.0
            continue
        low = params.register_low
        high = params.register_high
        chord_notes = _chord_tones_in_range(chord.root_pc, chord.quality, low, high)
        if not chord_notes:
            beat += 1.0
            continue

        start_note = min(chord_notes, key=lambda t: abs(t - state.pitch))
        idx = chord_notes.index(start_note)
        # Alternate direction per group, seeded from momentum
        base_dir = 1 if state.direction_momentum >= 0 else -1
        direction = base_dir if (group_idx % 2 == 0) else -base_dir

        for note_i in range(3):
            if beat + triplet_dur_beats > phrase_end + 1e-9:
                break
            if 0 <= idx < len(chord_notes):
                note_midi = chord_notes[idx]
            else:
                direction = -direction
                idx = max(0, min(len(chord_notes) - 1, idx))
                note_midi = chord_notes[idx]
            tick = _beat_to_tick(beat)
            if swing:
                tick = _apply_swing(tick)
            tick = _humanize(tick, beat=beat)
            vel = _choose_velocity(params, beat_in_bar=beat % float(state.beats_per_bar), intention=intention)
            notes.append(NoteEvent(
                pitch=note_midi, start_tick=tick,
                duration_ticks=triplet_dur_ticks, velocity=vel,
            ))
            state.pitch = note_midi
            state.record_pitch(note_midi)
            beat += triplet_dur_beats
            idx += direction

        if group_idx < num_groups - 1 and group_idx % 2 == 0 and tension < 0.7:
            beat += 0.5

    return notes, beat


# ---------------------------------------------------------------------------
# Grace notes / ornaments post-processor
# ---------------------------------------------------------------------------


def _add_grace_notes(notes: List[NoteEvent], probability: float = 0.12) -> List[NoteEvent]:
    """Add chromatic grace notes and mordents to a melody line.

    For each note at random, insert one of:
    - Single grace below (40%): chromatic lower neighbor
    - Single grace above (25%): chromatic upper neighbor
    - Double grace (20%): lower then upper neighbor approach
    - Mordent (15%): note -> upper neighbor -> note, very fast

    Grace notes are softer and very short (32nd-note equivalent).
    """
    grace_dur = TICKS_PER_QUARTER // 8  # 32nd note = 60 ticks
    result: List[NoteEvent] = []

    for note in notes:
        if random.random() >= probability:
            result.append(note)
            continue

        grace_vel = max(1, note.velocity - 15)
        roll = random.random()

        if roll < 0.40:
            # Single grace below: chromatic lower neighbor
            grace_pitch = max(0, note.pitch - 1)
            result.append(NoteEvent(
                pitch=grace_pitch,
                start_tick=max(0, note.start_tick - grace_dur),
                duration_ticks=grace_dur,
                velocity=grace_vel,
                channel=note.channel,
            ))
            result.append(note)

        elif roll < 0.65:
            # Single grace above: chromatic upper neighbor
            grace_pitch = min(127, note.pitch + 1)
            result.append(NoteEvent(
                pitch=grace_pitch,
                start_tick=max(0, note.start_tick - grace_dur),
                duration_ticks=grace_dur,
                velocity=grace_vel,
                channel=note.channel,
            ))
            result.append(note)

        elif roll < 0.85:
            # Double grace: lower then upper neighbor approach
            lower = max(0, note.pitch - 1)
            upper = min(127, note.pitch + 1)
            result.append(NoteEvent(
                pitch=lower,
                start_tick=max(0, note.start_tick - grace_dur * 2),
                duration_ticks=grace_dur,
                velocity=grace_vel,
                channel=note.channel,
            ))
            result.append(NoteEvent(
                pitch=upper,
                start_tick=max(0, note.start_tick - grace_dur),
                duration_ticks=grace_dur,
                velocity=grace_vel,
                channel=note.channel,
            ))
            result.append(note)

        else:
            # Mordent: note -> upper neighbor -> note (very fast)
            upper = min(127, note.pitch + 1)
            mordent_dur = grace_dur
            # Shorten the main note to make room
            shortened = max(grace_dur, note.duration_ticks - mordent_dur * 2)
            result.append(NoteEvent(
                pitch=note.pitch,
                start_tick=note.start_tick,
                duration_ticks=mordent_dur,
                velocity=note.velocity,
                channel=note.channel,
            ))
            result.append(NoteEvent(
                pitch=upper,
                start_tick=note.start_tick + mordent_dur,
                duration_ticks=mordent_dur,
                velocity=grace_vel,
                channel=note.channel,
            ))
            result.append(NoteEvent(
                pitch=note.pitch,
                start_tick=note.start_tick + mordent_dur * 2,
                duration_ticks=shortened,
                velocity=note.velocity,
                channel=note.channel,
            ))

    result.sort(key=lambda n: n.start_tick)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _extract_section_spans(
    chords: List[ChordEvent], total_beats: float,
) -> List[Tuple[str, float, float]]:
    """Extract contiguous form section spans from chord annotations.

    Returns e.g. [("A", 0.0, 32.0), ("A", 32.0, 64.0), ("B", 64.0, 96.0), ("A", 96.0, 128.0)]
    """
    if not chords or not chords[0].form_section:
        return []

    spans: List[Tuple[str, float, float]] = []
    current_label = chords[0].form_section
    span_start = 0.0

    for chord in chords:
        if chord.form_section and chord.form_section != current_label:
            spans.append((current_label, span_start, chord.start_beat))
            current_label = chord.form_section
            span_start = chord.start_beat

    # Final span
    if current_label:
        spans.append((current_label, span_start, total_beats))

    return spans


def _generate_section_melody(
    chords: List[ChordEvent], sec_start: float, sec_end: float,
    start_pitch: int, swing: bool, beats_per_bar: int = 4,
) -> Tuple[List[NoteEvent], int]:
    """Generate head melody for a single form section using phrase templates.

    Picks one rhythmic/melodic template and repeats it across 2-bar chunks,
    alternating question (original contour) and answer (inverted tail, truncated)
    to create composed-sounding heads with clear rhythmic identity.

    Returns (notes, final_pitch).
    """
    section_beats = sec_end - sec_start
    notes: List[NoteEvent] = []
    pitch = start_pitch

    # Edge case: very short section — just play one held chord tone
    if section_beats < 8.0:
        chord = _get_chord_at_beat(chords, sec_start)
        if chord:
            pitch = _resolve_head_pitch(pitch, 0, chord.root_pc, chord.quality,
                                        MELODY_LOW_BASE, MELODY_HIGH_BASE)
        tick = _beat_to_tick(sec_start)
        if swing:
            tick = _apply_swing(tick)
        dur_ticks = max(30, int(section_beats * 0.8 * TICKS_PER_QUARTER))
        notes.append(NoteEvent(pitch=pitch, start_tick=tick,
                               duration_ticks=dur_ticks, velocity=85))
        return notes, pitch

    # Pick a random phrase template for this section
    template = random.choice(HEAD_PHRASE_TEMPLATES)
    rhythm = template["rhythm"]
    contour = template["contour"]
    num_notes = len(rhythm)

    # Split section into 2-bar chunks
    chunk_size = 2.0 * beats_per_bar
    num_chunks = max(1, int(section_beats / chunk_size))
    mid_register = (MELODY_LOW_BASE + MELODY_HIGH_BASE) // 2

    for chunk_idx in range(num_chunks):
        chunk_start = sec_start + chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, sec_end)
        actual_beats = chunk_end - chunk_start
        is_answer = (chunk_idx % 2 == 1)

        # Register re-centering on question chunks if pitch has drifted
        if not is_answer and abs(pitch - mid_register) > 10:
            chord = _get_chord_at_beat(chords, chunk_start)
            if chord:
                center = _nearest_chord_tone(mid_register, chord.root_pc,
                                             chord.quality, MELODY_LOW_BASE,
                                             MELODY_HIGH_BASE)
                pitch = _nearest_chord_tone((pitch + center) // 2,
                                            chord.root_pc, chord.quality,
                                            MELODY_LOW_BASE, MELODY_HIGH_BASE)

        # Determine which notes to play in this chunk
        # Answer: drop the final note for breathing room, invert tail contour
        note_count = max(2, num_notes - 1) if is_answer else num_notes
        # Velocity offset: answers are slightly softer
        vel_offset = -6 if is_answer else 0
        # Half-point for contour inversion in answers
        invert_from = num_notes // 2

        for note_idx in range(note_count):
            beat_offset, duration = rhythm[note_idx]

            # Skip notes that fall outside a shortened final chunk
            if beat_offset >= actual_beats:
                continue

            absolute_beat = chunk_start + beat_offset
            # Clamp duration to chunk boundary
            max_dur = chunk_end - absolute_beat
            dur = min(duration, max_dur)
            if dur < 0.1:
                continue

            # Get chord at this beat
            chord = _get_chord_at_beat(chords, absolute_beat)
            if chord is None:
                continue

            # Resolve contour direction (invert tail for answers)
            direction = contour[note_idx]
            if is_answer and note_idx >= invert_from:
                direction = -direction

            pitch = _resolve_head_pitch(pitch, direction, chord.root_pc,
                                        chord.quality, MELODY_LOW_BASE,
                                        MELODY_HIGH_BASE)

            # Convert to MIDI event
            tick = _beat_to_tick(absolute_beat)
            if swing:
                tick = _apply_swing(tick)
            tick = _humanize(tick, beat=absolute_beat)
            dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))

            # Velocity: first note of chunk is louder (downbeat emphasis)
            if note_idx == 0:
                vel = random.randint(82, 95) + vel_offset
            else:
                vel = random.randint(70, 85) + vel_offset
            vel = max(1, min(127, vel))

            notes.append(NoteEvent(
                pitch=pitch, start_tick=tick,
                duration_ticks=dur_ticks, velocity=vel,
            ))

    return notes, pitch


def _vary_melody(
    original: List[NoteEvent], new_sec_start: float, orig_sec_start: float,
) -> List[NoteEvent]:
    """Replay a stored section melody with slight variations (pitch, timing, velocity)."""
    if not original:
        return []

    tick_offset = int((new_sec_start - orig_sec_start) * TICKS_PER_QUARTER)
    varied: List[NoteEvent] = []

    for note in original:
        pitch = note.pitch
        if random.random() < 0.15:
            pitch += random.choice([-2, -1, 1, 2])
            pitch = max(MELODY_LOW_BASE, min(MELODY_HIGH_BASE, pitch))

        tick = note.start_tick + tick_offset + random.randint(-15, 15)
        tick = max(0, tick)

        vel = max(1, min(127, note.velocity + random.randint(-8, 8)))

        varied.append(NoteEvent(
            pitch=pitch, start_tick=tick,
            duration_ticks=note.duration_ticks, velocity=vel,
        ))

    return varied


def generate_head_melody(chords: List[ChordEvent], total_beats: float,
                         swing: bool = True,
                         beats_per_bar: int = 4) -> List[NoteEvent]:
    """Generate a composed head melody with form-aware A-section recall."""
    if not chords:
        return []

    # Extract form section spans from chord annotations
    section_spans = _extract_section_spans(chords, total_beats)

    mid_pitch = (MELODY_LOW_BASE + MELODY_HIGH_BASE) // 2
    pitch = _nearest_chord_tone(mid_pitch, chords[0].root_pc, chords[0].quality,
                                MELODY_LOW_BASE, MELODY_HIGH_BASE)

    # No section info — flat generation (backward compat)
    if not section_spans:
        notes, _ = _generate_section_melody(chords, 0.0, total_beats, pitch, swing,
                                               beats_per_bar=beats_per_bar)
        notes.sort(key=lambda n: n.start_tick)
        notes = _add_grace_notes(notes, probability=0.12)
        return notes

    # Section-aware generation: store first occurrence of each label
    notes: List[NoteEvent] = []
    section_melodies: Dict[str, Tuple[List[NoteEvent], float]] = {}  # label -> (notes, sec_start)

    for label, sec_start, sec_end in section_spans:
        sec_chords = [c for c in chords
                      if c.start_beat < sec_end and c.end_beat > sec_start]

        if label in section_melodies:
            # Recall: replay with variation
            orig_notes, orig_start = section_melodies[label]
            varied = _vary_melody(orig_notes, sec_start, orig_start)
            notes.extend(varied)
            if varied:
                pitch = varied[-1].pitch
        else:
            # Compose: generate new melody for this section
            sec_notes, pitch = _generate_section_melody(
                sec_chords, sec_start, sec_end, pitch, swing,
                beats_per_bar=beats_per_bar)
            section_melodies[label] = (sec_notes, sec_start)
            notes.extend(sec_notes)

    notes.sort(key=lambda n: n.start_tick)
    notes = _add_grace_notes(notes, probability=0.12)
    return notes


def generate_solo(chords: List[ChordEvent], total_beats: float,
                  tension_curve: str = "arc", swing: bool = True,
                  coltrane: bool = False, seed: Optional[int] = None,
                  intensity: float = 1.0,
                  bar_feel: Optional[list] = None,
                  beats_per_bar: int = 4) -> List[NoteEvent]:
    """Generate an improvised jazz solo with tension-driven phrasing.

    *intensity* (0–1) scales the tension curve output so that lower-
    intensity solo sections never reach the upper tiers.
    """
    if seed is not None:
        random.seed(seed)
    if not chords:
        return []

    intensity = max(0.0, min(1.0, intensity))
    curve = TensionCurve(tension_curve)
    notes: List[NoteEvent] = []

    mid_pitch = (MELODY_LOW_BASE + MELODY_HIGH_BASE) // 2
    state = _SoloState(
        pitch=_nearest_chord_tone(mid_pitch, chords[0].root_pc, chords[0].quality,
                                  MELODY_LOW_BASE, MELODY_HIGH_BASE),
        beats_per_bar=beats_per_bar,
    )
    state.record_pitch(state.pitch)

    # Pre-plan the solo narrative arc
    narrative = _plan_solo_narrative(chords, total_beats, curve, intensity, coltrane)
    state.narrative = narrative

    # Pre-plan Tatum-style run blueprints
    run_plan = _plan_run_opportunities(chords, narrative, total_beats, intensity, curve)
    state.run_plan = run_plan

    beat = 0.0
    while beat < total_beats:
        global _current_feel
        bar_idx = int(beat) // beats_per_bar
        _current_feel = bar_feel[bar_idx] if bar_feel and bar_idx < len(bar_feel) else None

        progress = beat / total_beats if total_beats > 0 else 0.0
        tension = curve(progress) * intensity

        # Fetch narrative intention for this phrase
        intention = narrative.get_intention(state.phrase_count)
        state.reset_phrase_tracking()

        # Form section awareness: detect A/B/C transitions
        _fs_chord = _get_chord_at_beat(chords, beat)
        current_form_section = _fs_chord.form_section if _fs_chord else ""
        if (state.last_form_section and current_form_section
                and current_form_section != state.last_form_section
                and state.phrase_count > 0):
            # Breath at form section boundary (more space at low tension)
            beat += max(1.0, 2.0 - tension)
            state.last_form_section = current_form_section
            continue

        # Structural silence at tension transitions (always breathe, scale by delta)
        tension_delta = tension - state.last_tension
        if state.phrase_count > 0:
            if tension_delta > 0.15:
                # Breath before escalation — duration proportional to delta
                beat += 2.0 + tension_delta * 4.0
                state.last_tension = tension
                continue
            elif tension_delta < -0.2:
                # Brief pause on de-escalation
                beat += 1.0 + abs(tension_delta) * 2.0
                state.last_tension = tension
                continue
        state.last_tension = tension

        params = interpolate_params(tension, coltrane=coltrane)

        # Smooth tier selection with probability blending
        tier = _select_tier(tension, state.phrase_count)

        # Narrative target_tier nudge (replaces random bridge boost)
        if intention is not None and intention.target_tier != tier:
            if intention.target_tier > tier:
                tier = min(3, tier + 1)
            elif intention.target_tier < tier:
                tier = max(1, tier - 1)

        # Phrase length -- context-aware with narrative intention
        beats_until_change = _beats_until_chord_change(chords, beat)
        harmonic_rhythm_fast = beats_until_change <= 2.0
        state.harmonic_rhythm_speed = max(0.0, min(1.0, (3.0 - beats_until_change) / 2.0))

        phrase_beats = _choose_phrase_length(
            tier, tension, harmonic_rhythm_fast, state, intention)

        # Strategic silence: context-aware (replaces fixed 16% chance)
        if _should_insert_strategic_silence(state, tension, intention,
                                            harmonic_rhythm_fast=harmonic_rhythm_fast):
            # Silence duration scales with tension and preceding density
            density_factor = max(0.5, min(2.0, state.beats_since_last_silence / 16.0))
            silence_dur = (4.0 + (1.0 - tension) * 4.0) * density_factor
            silence_dur = min(silence_dur, 12.0)  # cap at 12 beats (3 bars)
            beat += silence_dur
            state.beats_since_last_silence = 0.0
            state.last_silence_duration = silence_dur
            continue

        # Clamp to remaining beats
        phrase_beats = min(phrase_beats, total_beats - beat)
        if phrase_beats < 0.5:
            break

        # Coltrane mode: key center awareness — populate substitute tones
        if coltrane:
            chord = _get_chord_at_beat(chords, beat)
            if chord is not None and chord.key_center_pc != chord.root_pc:
                state.substitute_tones = _substitute_key_tones(
                    chord.key_center_pc,
                    params.register_low,
                    params.register_high)
            else:
                state.substitute_tones = []

        # Build voice-leading target queue for this phrase
        state.target_queue = _build_target_queue(
            beat, chords, params.register_low, params.register_high,
            state.pitch, prefer_third=not state.last_target_was_third)
        if state.target_queue:
            state.last_target_was_third = not state.last_target_was_third

        # Long-range memory: callback every 8th phrase after phrase 6
        if (state.phrase_count > 6
                and state.phrase_count % 8 == 0
                and len(state.phrase_buffer) >= 3):
            # Pick highest quality phrase deterministically
            chosen = max(state.phrase_buffer, key=lambda m: m.quality_score)
            phrase_notes, beat = _generate_callback_phrase(
                chosen, beat, chords, state, swing)
            if phrase_notes:
                notes.extend(phrase_notes)
                state.phrase_count += 1
                state.callback_count += 1
                beat += 1.0  # fixed breath after callback
                continue

        # Run blueprint interception (v13): execute pre-planned Tatum runs
        active_run = state.run_plan.get_active_run(beat) if state.run_plan else None
        if active_run is not None:
            if intention and intention.character in (PC_WHISPER, PC_SILENCE):
                active_run.executed = True  # narrative override — suppress run
            else:
                run_notes, beat = _build_run_notes(
                    active_run, chords, state, params, swing)
                if run_notes:
                    notes.extend(run_notes)
                    _cur_chord = _get_chord_at_beat(chords, beat)
                    _cur_root_pc = _cur_chord.root_pc if _cur_chord else 0
                    state.record_phrase(run_notes, beat, tension, tier, _cur_root_pc)
                    _record_phrase_ending(state, run_notes, chords, beat)
                    state.record_phrase_length(active_run.end_beat - active_run.start_beat)
                    state.beats_since_last_silence += (active_run.end_beat - active_run.start_beat)
                    state.phrase_count += 1
                    beat += 0.5 + (1.0 - tension) * 0.5  # breath after run
                    beat = max(beat, active_run.end_beat)  # forward progress
                    continue

        # Upcoming run awareness: truncate phrase to leave room for approaching run
        upcoming_run = state.run_plan.get_upcoming_run(beat) if state.run_plan else None
        if upcoming_run:
            max_beats = upcoming_run.start_beat - beat - 0.5
            if max_beats > 0.5:
                phrase_beats = min(phrase_beats, max_beats)
            else:
                beat = upcoming_run.start_beat
                continue

        # Q-A opening: adjust starting pitch based on how last phrase ended
        qa_pitch = _qa_opening_adjustment(state, intention, state.pitch,
                                          params.register_low, params.register_high)
        if qa_pitch is not None:
            state.pitch = qa_pitch

        # Generate phrase based on tier
        if tier == 1:
            phrase_notes, beat = _generate_tier1_phrase(
                beat, phrase_beats, chords, params, swing, state, tension,
                intention=intention)
        elif tier == 2:
            sub_type = _choose_strategy(
                ["motivic", "pentatonic", "call_response", "triplet"],
                [0.35, 0.25, 0.20, 0.20], state, intention)
            if sub_type == "pentatonic" and tension > 0.5:
                phrase_notes, beat = _generate_pentatonic_super_phrase(
                    beat, phrase_beats, chords, params, swing, coltrane, state, tension,
                    intention=intention)
            elif sub_type == "call_response":
                phrase_notes, beat = _generate_call_response_phrase(
                    beat, phrase_beats, chords, params, swing, coltrane, state, tension,
                    intention=intention)
            elif sub_type == "triplet" and tension > 0.5:
                phrase_notes, beat = _generate_triplet_phrase(
                    beat, phrase_beats, chords, params, swing, coltrane, state, tension,
                    intention=intention)
            else:
                phrase_notes, beat = _generate_tier2_phrase(
                    beat, phrase_beats, chords, params, swing, coltrane, state, tension,
                    intention=intention)
        else:
            phrase_notes, beat = _generate_tier3_phrase(
                beat, phrase_beats, chords, params, swing, coltrane, state, tension,
                intention=intention)

        notes.extend(phrase_notes)
        state.phrase_count += 1

        # Record phrase for long-range memory callbacks
        _cur_chord = _get_chord_at_beat(chords, beat)
        _cur_root_pc = _cur_chord.root_pc if _cur_chord is not None else 0
        state.record_phrase(phrase_notes, beat, tension, tier, _cur_root_pc)

        # Narrative context: record phrase ending and length
        _record_phrase_ending(state, phrase_notes, chords, beat)
        state.record_phrase_length(phrase_beats)
        state.beats_since_last_silence += phrase_beats

        # Rhythmic displacement — always when tension > 0.5, amount scales
        if tension > 0.5:
            beat += 0.25 + (tension - 0.5) * 0.5

        # Ghost notes (Coltrane mode: every 6th phrase when 0.3 < t < 0.6)
        if coltrane and 0.3 < tension < 0.6 and state.phrase_count % 6 == 0:
            num_ghost = 1 + int(tension * 2)
            ghost_beat = beat
            ghost_pitch = state.pitch
            ghost_dir = 1 if state.direction_momentum >= 0 else -1
            ghost_vel = 25 + int(tension * 10)
            for _ in range(num_ghost):
                if ghost_beat >= total_beats:
                    break
                ghost_pitch = max(MELODY_LOW, min(MELODY_HIGH, ghost_pitch + ghost_dir))
                ghost_tick = _beat_to_tick(ghost_beat)
                if swing:
                    ghost_tick = _apply_swing(ghost_tick)
                notes.append(NoteEvent(
                    pitch=ghost_pitch, start_tick=ghost_tick,
                    duration_ticks=TICKS_PER_16TH, velocity=ghost_vel,
                ))
                ghost_beat += 0.25

        # Context-aware post-phrase rest (replaces mandatory breath + strategic rests)
        rest_dur = _post_phrase_rest(state, tension, tier, intention,
                                     harmonic_rhythm_fast=harmonic_rhythm_fast)

        # Always breathe on key center changes, proportional to key distance
        current_chord = _get_chord_at_beat(chords, beat)
        current_kc = current_chord.key_center_pc if current_chord else None
        if (state.last_key_center_pc is not None and current_kc is not None
                and current_kc != state.last_key_center_pc):
            semitone_dist = min(abs(current_kc - state.last_key_center_pc),
                                12 - abs(current_kc - state.last_key_center_pc))
            rest_dur = max(rest_dur, 1.5 + semitone_dist * 0.15)

        beat += rest_dur
        state.beats_since_last_silence += rest_dur
        state.last_key_center_pc = current_kc
        state.last_tier = tier
        state.last_form_section = current_form_section

    notes.sort(key=lambda n: n.start_tick)
    notes = _add_grace_notes(notes, probability=0.10)
    return notes


def generate_trading_fours(chords: List[ChordEvent], total_beats: float,
                           intensity: float = 0.6,
                           beats_per_bar: int = 4) -> List[NoteEvent]:
    """Generate a trading-fours section (4 bars melody, 4 bars drums)."""
    if not chords:
        return []

    notes: List[NoteEvent] = []
    beat = 0.0
    phrase_size = 8.0 * beats_per_bar   # 4 bars melody + 4 bars drums
    melody_beats = 4.0 * beats_per_bar  # 4 bars of melody

    mid_pitch = (MELODY_LOW_BASE + MELODY_HIGH_BASE) // 2
    pitch = _nearest_chord_tone(mid_pitch, chords[0].root_pc, chords[0].quality,
                                MELODY_LOW_BASE, MELODY_HIGH_BASE)
    recent_pitches = [pitch]
    params = interpolate_params(min(1.0, intensity + 0.15))

    while beat < total_beats:
        melody_end = min(beat + melody_beats, total_beats)
        melody_section_beats = melody_end - beat
        if melody_section_beats <= 0:
            break

        current_beat = beat
        while current_beat < melody_end - 1e-9:
            chord = _get_chord_at_beat(chords, current_beat)
            if chord is None:
                current_beat += 0.5
                continue

            low = params.register_low
            high = params.register_high

            # Use _choose_target_pitch for variety
            target = _choose_target_pitch(
                pitch, chord.root_pc, chord.quality,
                low, high, tension=intensity, beat_in_bar=current_beat % float(beats_per_bar))

            # Use rhythmic cells
            cell = _choose_rhythmic_cell(tension=intensity)
            dur = cell[0]

            if _contour_check(recent_pitches):
                direction = -1 if len(recent_pitches) >= 2 and recent_pitches[-1] > recent_pitches[-2] else 1
                candidates = _chord_tones_in_range(chord.root_pc, chord.quality, low, high)
                if direction > 0:
                    above = [t for t in candidates if t > pitch]
                    if above:
                        target = above[0]
                else:
                    below = [t for t in candidates if t < pitch]
                    if below:
                        target = below[-1]

            if current_beat + dur > melody_end:
                remaining = melody_end - current_beat
                if remaining < 0.125:
                    break
                dur = remaining

            tick = _beat_to_tick(current_beat)
            tick = _apply_swing(tick)
            tick = _humanize(tick, beat=current_beat)
            dur_ticks = max(30, int(dur * TICKS_PER_QUARTER))
            vel = _choose_velocity(params, beat_in_bar=current_beat % float(beats_per_bar))

            notes.append(NoteEvent(
                pitch=target, start_tick=tick,
                duration_ticks=dur_ticks, velocity=vel,
            ))

            pitch = target
            recent_pitches.append(pitch)
            if len(recent_pitches) > 20:
                recent_pitches = recent_pitches[-20:]
            current_beat += dur

        beat += phrase_size

    notes.sort(key=lambda n: n.start_tick)
    return notes


# ---------------------------------------------------------------------------
# Expression: pitch bends (blue notes, grace notes) and CC curves for melody
# ---------------------------------------------------------------------------

MELODY_CHANNEL = 0


def _is_blue_note(pitch: int, chord) -> bool:
    """Check if pitch is a b3, b5, or b7 over a major/dominant chord."""
    if not hasattr(chord, "root_pc") or not hasattr(chord, "quality"):
        return False
    q = chord.quality
    if q not in ("maj7", "dom7", "7", "maj", "6"):
        return False
    root = chord.root_pc
    pc = pitch % 12
    interval = (pc - root) % 12
    # b3 = 3, b5 = 6, b7 = 10 (over major/dominant)
    return interval in (3, 6, 10)


def _chord_at_tick(chords: list, tick: int):
    """Find the chord active at a given tick."""
    for c in reversed(chords):
        c_start = int(getattr(c, "start_beat", 0) * TICKS_PER_QUARTER)
        c_dur = int(getattr(c, "duration_beats", 4) * TICKS_PER_QUARTER)
        if c_start <= tick < c_start + c_dur:
            return c
    return chords[0] if chords else None


def generate_melody_expression(
    notes: List[NoteEvent],
    chords: list,
    channel: int = MELODY_CHANNEL,
    beats_per_bar: int = 4,
) -> tuple:
    """Generate pitch bends and CC curves for expressive melody.

    - Blue note bends: b3/b5/b7 over major/dominant → bend down 300-500
    - Grace note slides: 8% of notes, fast bend -1024→0 over 30 ticks
    - CC1 vibrato: sustained notes > 1 beat, oscillate CC1 after half-beat delay
    - CC11 phrase dynamics: rise 80→120 over first 60%, fall 120→70 over last 40%

    Returns:
        (List[PitchBendEvent], List[CCEvent])
    """
    bends: List[PitchBendEvent] = []
    ccs: List[CCEvent] = []

    if not notes:
        return bends, ccs

    # --- CC11 phrase dynamics ---
    # Group notes into phrases (gap > 1 bar = new phrase)
    phrases: List[List[NoteEvent]] = []
    tpb = ticks_per_bar(beats_per_bar)
    current_phrase: List[NoteEvent] = []
    for note in notes:
        if current_phrase:
            prev = current_phrase[-1]
            gap = note.start_tick - (prev.start_tick + prev.duration_ticks)
            if gap > tpb:
                phrases.append(current_phrase)
                current_phrase = []
        current_phrase.append(note)
    if current_phrase:
        phrases.append(current_phrase)

    for phrase in phrases:
        phrase_start = phrase[0].start_tick
        phrase_end = phrase[-1].start_tick + phrase[-1].duration_ticks
        phrase_dur = phrase_end - phrase_start
        if phrase_dur < tpb:
            continue

        # Rise 80→120 over first 60%, fall 120→70 over last 40%
        # One CC11 per bar boundary — natural, gradual dynamic arc
        n_bars = max(1, phrase_dur // tpb)
        if n_bars < 2:
            # Short phrase: just set a flat value
            ccs.append(CCEvent(cc_number=11, value=100,
                               start_tick=phrase_start, channel=channel))
            continue
        for bar_idx in range(n_bars + 1):
            t = phrase_start + bar_idx * tpb
            if t > phrase_end:
                break
            frac = bar_idx / n_bars
            if frac < 0.6:
                val = int(80 + (120 - 80) * (frac / 0.6))
            else:
                val = int(120 - (120 - 70) * ((frac - 0.6) / 0.4))
            ccs.append(CCEvent(cc_number=11, value=max(40, min(127, val)),
                               start_tick=t, channel=channel))

    # --- Per-note expression ---
    # Filter: when multiple notes share a tick (block chords from head
    # harmonization), only apply expression to the highest pitch (melody on top).
    _melody_top: dict = {}
    for n in notes:
        key = n.start_tick // 10  # group within 10-tick window
        if key not in _melody_top or n.pitch > _melody_top[key].pitch:
            _melody_top[key] = n
    _top_note_set = set(id(n) for n in _melody_top.values())

    for i, note in enumerate(notes):
        if id(note) not in _top_note_set:
            continue  # skip harmony notes below melody
        dur = note.duration_ticks
        start = note.start_tick

        chord = _chord_at_tick(chords, start)

        # Blue note bends (65% probability)
        if chord and _is_blue_note(note.pitch, chord) and random.random() < 0.65:
            bend_amount = random.randint(-500, -300)
            # Bend at note start, gradually resolve to 0 over first third
            bends.append(PitchBendEvent(value=bend_amount, start_tick=start, channel=channel))
            resolve_tick = start + max(20, dur // 3)
            resolve_steps = max(2, (resolve_tick - start) // 15)
            for s in range(1, resolve_steps + 1):
                t = start + int(s * (resolve_tick - start) / resolve_steps)
                val = int(bend_amount * (1.0 - s / resolve_steps))
                bends.append(PitchBendEvent(value=val, start_tick=t, channel=channel))
            bends.append(PitchBendEvent(value=0, start_tick=resolve_tick, channel=channel))

        # Grace note slides (8% of notes)
        elif random.random() < 0.08:
            slide_dur = min(30, dur // 4)
            if slide_dur >= 10:
                ramp_start = max(0, start - slide_dur)
                steps = max(2, slide_dur // 8)
                for s in range(steps):
                    t = ramp_start + int(s * slide_dur / steps)
                    val = int(-1024 * (1.0 - s / steps))
                    bends.append(PitchBendEvent(value=val, start_tick=t, channel=channel))
                bends.append(PitchBendEvent(value=0, start_tick=start, channel=channel))

        # CC1 vibrato on sustained notes (> 1 beat)
        if dur > TICKS_PER_QUARTER:
            delay_ticks = TICKS_PER_QUARTER // 2  # half-beat delay
            vib_start = start + delay_ticks
            vib_end = start + dur
            # ~5Hz vibrato at 480 tpq, 140bpm → ~18 ticks per cycle
            # Use ~30 ticks per oscillation for moderate vibrato
            cycle_ticks = 30
            t = vib_start
            phase = 0
            while t < vib_end:
                val = int(35 + 15 * math.sin(phase))  # oscillate 20-50
                ccs.append(CCEvent(cc_number=1, value=max(0, min(127, val)),
                                   start_tick=t, channel=channel))
                t += cycle_ticks
                phase += math.pi / 2  # quarter cycle per step
            # Reset vibrato off at note end
            ccs.append(CCEvent(cc_number=1, value=0, start_tick=vib_end, channel=channel))

    return bends, ccs
