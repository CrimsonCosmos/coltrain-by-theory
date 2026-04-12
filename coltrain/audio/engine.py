"""MIDI-to-audio rendering engine for Coltrain.

Orchestrates all synths, applies expression data (pitch bends, CCs),
and produces a mixed stereo output.
"""

import numpy as np

from coltrain.generation import NoteEvent, CCEvent, PitchBendEvent, TICKS_PER_QUARTER
from . import SAMPLE_RATE
from .bass_synth import UprightBassSynth
from .piano_synth import PianoSynth
from .drum_synth import DrumSynth
from .mixer import mix_to_stereo, write_wav, write_mp3


class AudioEngine:
    """Render MIDI tracks to audio using synthesis."""

    def __init__(self, tempo: int, sample_rate: int = SAMPLE_RATE):
        self.tempo = tempo
        self.sr = sample_rate
        self.bass_synth = UprightBassSynth(sample_rate)
        self.piano_synth = PianoSynth(sample_rate)
        self.drum_synth = DrumSynth(sample_rate)

    def tick_to_sample(self, tick: int) -> int:
        """Convert MIDI tick to sample position."""
        seconds = (tick / TICKS_PER_QUARTER) * (60.0 / self.tempo)
        return int(seconds * self.sr)

    def render(
        self,
        tracks: dict,
        cc_events: dict = None,
        pitch_bend_events: dict = None,
        output_path: str = "output.wav",
        format: str = "wav",
    ) -> str:
        """Render all tracks to audio.

        Args:
            tracks: Dict mapping track name to list of NoteEvent.
            cc_events: Dict mapping track name to list of CCEvent.
            pitch_bend_events: Dict mapping track name to list of PitchBendEvent.
            output_path: Base output path (extension replaced by format).
            format: 'wav' or 'mp3'.

        Returns:
            Path to the written audio file.
        """
        cc_events = cc_events or {}
        pitch_bend_events = pitch_bend_events or {}

        # Find total duration in samples
        max_tick = 0
        for note_list in tracks.values():
            for note in note_list:
                end_tick = note.start_tick + note.duration_ticks
                if end_tick > max_tick:
                    max_tick = end_tick
        # Add 2 seconds of tail for reverb/decay
        total_samples = self.tick_to_sample(max_tick) + 2 * self.sr

        print(f"  Rendering {total_samples / self.sr:.1f}s of audio...")

        # Render each instrument
        instrument_buffers = {}

        # --- Melody (piano) ---
        melody_notes = tracks.get("melody", [])
        if melody_notes:
            print(f"    Piano: {len(melody_notes)} notes...", flush=True)
            melody_pb = self._build_pb_curve(
                pitch_bend_events.get("melody", []), total_samples
            )
            melody_sustain = self._build_sustain_map(
                cc_events.get("melody", [])
            )
            instrument_buffers["melody"] = self._render_piano(
                melody_notes, total_samples, melody_sustain
            )

        # --- Bass ---
        bass_notes = tracks.get("bass", [])
        if bass_notes:
            print(f"    Bass: {len(bass_notes)} notes...", flush=True)
            bass_pb = self._build_pb_curve(
                pitch_bend_events.get("bass", []), total_samples
            )
            bass_expr = self._build_cc_curve(
                cc_events.get("bass", []), 11, total_samples, default=100
            )
            instrument_buffers["bass"] = self._render_bass(
                bass_notes, total_samples, bass_pb, bass_expr
            )

        # --- Drums ---
        drum_notes = tracks.get("drums", [])
        if drum_notes:
            print(f"    Drums: {len(drum_notes)} notes...", flush=True)
            cc4_events = [
                cc for cc in cc_events.get("drums", []) if cc.cc_number == 4
            ]
            instrument_buffers["drums"] = self._render_drums(
                drum_notes, total_samples, cc4_events
            )

        # Mix and write
        print("    Mixing and mastering...", flush=True)
        stereo = mix_to_stereo(instrument_buffers, self.sr)

        # Determine output path
        import os
        base = os.path.splitext(output_path)[0]
        if format == "mp3":
            out_path = base + ".mp3"
            write_mp3(stereo, out_path, self.sr)
        else:
            out_path = base + ".wav"
            write_wav(stereo, out_path, self.sr)

        return out_path

    # ------------------------------------------------------------------
    # Instrument renderers
    # ------------------------------------------------------------------

    def _render_piano(
        self, notes: list, total_samples: int, sustain_map: dict
    ) -> np.ndarray:
        """Render piano notes into a mono buffer."""
        buf = np.zeros(total_samples, dtype=np.float64)
        for note in notes:
            start = self.tick_to_sample(note.start_tick)
            end = self.tick_to_sample(note.start_tick + note.duration_ticks)
            dur = max(end - start, int(0.05 * self.sr))

            # Check if sustain pedal is held at this tick
            sustain = self._is_sustained(note.start_tick, sustain_map)

            rendered = self.piano_synth.render_note(
                note.pitch, dur, note.velocity, sustain=sustain
            )
            end_idx = min(start + len(rendered), total_samples)
            buf[start:end_idx] += rendered[: end_idx - start]
        return buf

    def _render_bass(
        self, notes: list, total_samples: int,
        pb_curve: np.ndarray, expr_curve: np.ndarray,
    ) -> np.ndarray:
        """Render bass notes into a mono buffer."""
        buf = np.zeros(total_samples, dtype=np.float64)
        for note in notes:
            start = self.tick_to_sample(note.start_tick)
            end = self.tick_to_sample(note.start_tick + note.duration_ticks)
            dur = max(end - start, int(0.05 * self.sr))

            # Slice pitch bend and expression for this note's duration
            note_pb = None
            if pb_curve is not None:
                pb_start = min(start, len(pb_curve) - 1)
                pb_end = min(start + dur, len(pb_curve))
                if pb_end > pb_start:
                    note_pb = pb_curve[pb_start:pb_end]

            note_expr = None
            if expr_curve is not None:
                ex_start = min(start, len(expr_curve) - 1)
                ex_end = min(start + dur, len(expr_curve))
                if ex_end > ex_start:
                    # Normalize CC11 (0-127) to 0-1
                    note_expr = expr_curve[ex_start:ex_end] / 127.0

            rendered = self.bass_synth.render_note(
                note.pitch, dur, note.velocity,
                pitch_bend_curve=note_pb, expression_curve=note_expr,
            )
            end_idx = min(start + len(rendered), total_samples)
            buf[start:end_idx] += rendered[: end_idx - start]
        return buf

    def _render_drums(
        self, notes: list, total_samples: int, cc4_events: list
    ) -> np.ndarray:
        """Render drum notes into a mono buffer."""
        buf = np.zeros(total_samples, dtype=np.float64)

        # Build CC4 lookup: sorted list of (tick, value)
        cc4_sorted = sorted(
            [(cc.start_tick, cc.value) for cc in cc4_events],
            key=lambda x: x[0],
        )

        for note in notes:
            start = self.tick_to_sample(note.start_tick)

            # Find CC4 value at this tick
            cc4_val = 0
            for tick, val in cc4_sorted:
                if tick <= note.start_tick:
                    cc4_val = val
                else:
                    break

            rendered = self.drum_synth.render_hit(
                note.pitch, note.velocity, cc4_value=cc4_val
            )
            end_idx = min(start + len(rendered), total_samples)
            if end_idx > start:
                buf[start:end_idx] += rendered[: end_idx - start]
        return buf

    # ------------------------------------------------------------------
    # CC / pitch bend helpers
    # ------------------------------------------------------------------

    def _build_pb_curve(
        self, events: list, total_samples: int
    ) -> np.ndarray:
        """Build a per-sample pitch bend curve in semitones.

        MIDI pitch bend range is typically +/- 2 semitones.
        Values -8192 to +8191 map to -2 to +2 semitones.
        """
        curve = np.zeros(total_samples, dtype=np.float64)
        if not events:
            return curve

        sorted_events = sorted(events, key=lambda e: e.start_tick)
        current_val = 0.0
        prev_sample = 0

        for ev in sorted_events:
            sample = min(self.tick_to_sample(ev.start_tick), total_samples - 1)
            # Fill from prev to current with the previous value
            if sample > prev_sample:
                curve[prev_sample:sample] = current_val
            current_val = (ev.value / 8192.0) * 2.0  # Convert to semitones
            prev_sample = sample

        # Fill remainder
        if prev_sample < total_samples:
            curve[prev_sample:] = current_val

        return curve

    def _build_cc_curve(
        self, events: list, cc_number: int, total_samples: int,
        default: int = 0,
    ) -> np.ndarray:
        """Build a per-sample CC curve (0-127)."""
        curve = np.full(total_samples, default, dtype=np.float64)
        if not events:
            return curve

        cc_only = sorted(
            [e for e in events if e.cc_number == cc_number],
            key=lambda e: e.start_tick,
        )
        current_val = float(default)
        prev_sample = 0

        for ev in cc_only:
            sample = min(self.tick_to_sample(ev.start_tick), total_samples - 1)
            if sample > prev_sample:
                curve[prev_sample:sample] = current_val
            current_val = float(ev.value)
            prev_sample = sample

        if prev_sample < total_samples:
            curve[prev_sample:] = current_val

        return curve

    def _build_sustain_map(self, events: list) -> dict:
        """Build a sustain pedal state map from CC64 events.

        Returns dict with tick keys and bool values (True = pedal down).
        """
        sustain = {}
        if not events:
            return sustain
        for ev in events:
            if ev.cc_number == 64:
                sustain[ev.start_tick] = ev.value >= 64
        return sustain

    def _is_sustained(self, tick: int, sustain_map: dict) -> bool:
        """Check if sustain pedal is held at a given tick."""
        if not sustain_map:
            return False
        # Find most recent sustain event at or before this tick
        held = False
        for t in sorted(sustain_map.keys()):
            if t <= tick:
                held = sustain_map[t]
            else:
                break
        return held
