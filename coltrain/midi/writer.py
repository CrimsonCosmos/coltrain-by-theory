"""MIDI file writer for Coltrain.

Converts generated NoteEvent lists into a standard MIDI file using the mido library.
Handles multi-track output with proper delta-time conversion, overlapping notes,
and correct channel/program assignments for each instrument.
"""

import os
from typing import Dict, List

import mido

from ..generation import NoteEvent, CCEvent, PitchBendEvent, TICKS_PER_QUARTER

# ---------------------------------------------------------------------------
# Instrument configuration
# ---------------------------------------------------------------------------

# MIDI program numbers (General MIDI)
PROGRAMS = {
    "melody": 0,     # Acoustic Grand Piano (right hand)
    "piano": 0,      # Acoustic Grand Piano (left hand)
    "bass": 32,      # Acoustic Bass
    "drums": None,   # Drums use channel 9, no program change needed
}

# Alternate lead instruments
LEAD_PROGRAMS = {
    "sax": 65,       # Alto Sax
    "trumpet": 56,   # Trumpet
    "piano": 0,      # Acoustic Grand Piano
}

# MIDI channels
CHANNELS = {
    "melody": 0,
    "piano": 1,
    "bass": 2,
    "drums": 9,
}

# Default stereo pan positions (CC10: 0=hard left, 64=center, 127=hard right)
DEFAULT_PAN = {
    "melody": 45,    # Right hand piano — left of center
    "piano": 82,     # Left hand comping — right of center
    "bass": 58,      # Slightly left of center
    "drums": 70,     # Slightly right of center
}

# Track ordering
TRACK_ORDER = ("melody", "piano", "bass", "drums")


# ---------------------------------------------------------------------------
# Core MIDI writer
# ---------------------------------------------------------------------------


def write_midi(
    tracks: Dict[str, List[NoteEvent]],
    output_path: str,
    tempo: int = 140,
    lead_instrument: str = "sax",
    cc_events: Dict[str, List] = None,
    pitch_bend_events: Dict[str, List] = None,
) -> int:
    """Write a multi-track MIDI file from generated NoteEvent data.

    Args:
        tracks: Dict mapping track name to list of NoteEvent.
                Expected keys: 'melody', 'piano', 'bass', 'drums'.
        output_path: File path for the .mid output.
        tempo: BPM (beats per minute).
        lead_instrument: Lead instrument choice ('sax', 'trumpet', 'piano').

    Returns:
        Total number of notes written across all tracks.
    """
    mid = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_QUARTER)

    # Track 0: tempo/meta track
    tempo_track = mido.MidiTrack()
    mid.tracks.append(tempo_track)
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
    tempo_track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    tempo_track.append(mido.MetaMessage("track_name", name="Coltrain", time=0))

    # Override melody program if a different lead instrument was chosen
    programs = dict(PROGRAMS)
    if lead_instrument in LEAD_PROGRAMS:
        programs["melody"] = LEAD_PROGRAMS[lead_instrument]

    total_notes = 0

    for track_name in TRACK_ORDER:
        note_events = tracks.get(track_name, [])
        if not note_events:
            # Still create an empty track for consistency
            track = mido.MidiTrack()
            mid.tracks.append(track)
            track.append(mido.MetaMessage("track_name", name=track_name, time=0))
            continue

        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("track_name", name=track_name, time=0))

        channel = CHANNELS.get(track_name, 0)

        # Program change
        if track_name == "drums":
            track.append(mido.Message("program_change", program=32,
                                       channel=channel, time=0))
        else:
            program = programs.get(track_name, 0)
            track.append(mido.Message("program_change", program=program,
                                       channel=channel, time=0))

        # Build a list of all MIDI events (note_on and note_off) with absolute times
        midi_events = []
        for ne in note_events:
            # Clamp values
            pitch = max(0, min(127, ne.pitch))
            velocity = max(1, min(127, ne.velocity))
            # Pull drums back in the mix (classic jazz balance)
            if track_name == "drums":
                velocity = max(1, min(127, int(velocity * 0.55)))
            start = max(0, ne.start_tick)
            end = max(start + 1, start + ne.duration_ticks)

            midi_events.append((start, "note_on", pitch, velocity, channel))
            midi_events.append((end, "note_off", pitch, 0, channel))

        # Merge CC events for this track
        if cc_events and track_name in cc_events:
            for cc in cc_events[track_name]:
                cc_num = max(0, min(127, cc.cc_number))
                cc_val = max(0, min(127, cc.value))
                cc_ch = cc.channel
                midi_events.append((max(0, cc.start_tick), "control_change", cc_num, cc_val, cc_ch))

        # Merge pitch bend events for this track
        if pitch_bend_events and track_name in pitch_bend_events:
            for pb in pitch_bend_events[track_name]:
                pb_val = max(-8192, min(8191, pb.value))
                pb_ch = pb.channel
                midi_events.append((max(0, pb.start_tick), "pitchwheel", pb_val, 0, pb_ch))

        # Sort by absolute time; note_off before note_on at the same time
        # to avoid hanging notes. Priority: note_off(0) < CC(1) < pitchwheel(1.5) < note_on(2).
        def _sort_key(e):
            if e[1] == "note_off":
                return (e[0], 0)
            elif e[1] == "control_change":
                return (e[0], 1)
            elif e[1] == "pitchwheel":
                return (e[0], 1.5)
            else:
                return (e[0], 2)
        midi_events.sort(key=_sort_key)

        # Convert to delta times
        current_time = 0
        for event in midi_events:
            abs_time = event[0]
            msg_type = event[1]
            delta = abs_time - current_time
            if delta < 0:
                delta = 0

            if msg_type == "note_on":
                _, _, pitch, velocity, ch = event
                track.append(mido.Message("note_on", note=pitch, velocity=velocity,
                                           channel=ch, time=delta))
            elif msg_type == "note_off":
                _, _, pitch, velocity, ch = event
                track.append(mido.Message("note_off", note=pitch, velocity=0,
                                           channel=ch, time=delta))
            elif msg_type == "control_change":
                _, _, cc_num, cc_val, ch = event
                track.append(mido.Message("control_change", control=cc_num,
                                           value=cc_val, channel=ch, time=delta))
            elif msg_type == "pitchwheel":
                _, _, pb_val, _, ch = event
                track.append(mido.Message("pitchwheel", pitch=pb_val,
                                           channel=ch, time=delta))
            current_time = abs_time

        total_notes += len(note_events)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    mid.save(output_path)
    return total_notes


# ---------------------------------------------------------------------------
# Chord chart writer
# ---------------------------------------------------------------------------


def write_chord_chart(chords, output_path: str, bars_per_line: int = 4) -> None:
    """Write a text-file chord chart from a list of ChordEvent objects.

    Each bar shows the chord symbol(s) active during that bar. Bars are grouped
    into lines of bars_per_line for readability.

    Args:
        chords: List of ChordEvent objects (must have .label, .start_beat,
                .duration_beats properties).
        output_path: File path for the .txt output.
        bars_per_line: Number of bars per line in the chart.
    """
    if not chords:
        return

    # Find total duration
    total_beats = max(c.start_beat + c.duration_beats for c in chords)
    total_bars = int(total_beats) // 4
    if total_beats % 4 > 0:
        total_bars += 1

    # Build bar contents
    bar_contents = []
    for bar_idx in range(total_bars):
        bar_start = bar_idx * 4.0
        bar_end = bar_start + 4.0

        # Find all chords active during this bar
        bar_chords = []
        for c in chords:
            if c.start_beat < bar_end and c.start_beat + c.duration_beats > bar_start:
                bar_chords.append(c)

        if not bar_chords:
            bar_contents.append("      ")
        elif len(bar_chords) == 1:
            label = bar_chords[0].label
            bar_contents.append(label.center(12))
        else:
            # Multiple chords in one bar
            labels = " ".join(c.label for c in bar_chords)
            bar_contents.append(labels[:12].center(12))

    # Format into lines
    col_width = 14
    lines = []
    lines.append("=" * (bars_per_line * col_width + 1))
    for i in range(0, len(bar_contents), bars_per_line):
        row = bar_contents[i : i + bars_per_line]
        line = "|" + "|".join(f"{b:^{col_width}}" for b in row) + "|"
        lines.append(line)
        lines.append("-" * len(line))

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
