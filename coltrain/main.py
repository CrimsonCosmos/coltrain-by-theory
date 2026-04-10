"""Coltrain By Theory -- CLI entry point.

Rule-based jazz MIDI generation using pure music theory.

Usage:
    python -m coltrain.main --form giantsteps --key B --tempo 286 --choruses 3 --coltrane
"""

import argparse
import os
import random
import sys

from coltrain.theory.pitch import NOTE_TO_PC, pc_to_name
from coltrain.theory.chord import chord_label
from coltrain.generation.arrangement import (
    FORM_TEMPLATES,
    ArrangementSection,
    build_arrangement,
    build_chord_progression,
    generate_arrangement,
)
from coltrain.midi.writer import write_midi, write_chord_chart, LEAD_PROGRAMS


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="coltrain",
        description="Coltrain By Theory -- Rule-based jazz MIDI generation",
    )

    parser.add_argument(
        "--form",
        choices=list(FORM_TEMPLATES.keys()),
        default="blues12",
        help="Form template (default: blues12)",
    )
    parser.add_argument(
        "--key",
        default="C",
        help="Key (C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B). Default: C",
    )
    parser.add_argument(
        "--tempo",
        type=int,
        default=140,
        help="Tempo in BPM (default: 140)",
    )
    parser.add_argument(
        "--choruses",
        type=int,
        default=2,
        help="Number of solo choruses (total arrangement is longer). Default: 2",
    )
    parser.add_argument(
        "--tension",
        choices=["arc", "build", "wave", "plateau", "catharsis"],
        default="arc",
        help="Tension curve shape for solos (default: arc)",
    )
    parser.add_argument(
        "--coltrane",
        action="store_true",
        help="Enable Coltrane features (multi-tonic, sheets of sound)",
    )
    parser.add_argument(
        "--instrument",
        choices=["sax", "trumpet", "piano"],
        default="sax",
        help="Lead instrument (default: sax). sax=65, trumpet=56, piano=0",
    )
    parser.add_argument(
        "--swing",
        type=float,
        default=0.667,
        help="Swing ratio (0.5=straight, 0.667=triplet swing). Default: 0.667",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible output",
    )
    parser.add_argument(
        "--output", "-o",
        default="output.mid",
        help="Output MIDI file path (default: output.mid)",
    )
    parser.add_argument(
        "--chart",
        action="store_true",
        help="Also write a chord chart text file alongside the MIDI",
    )
    parser.add_argument(
        "--no-humanize",
        action="store_true",
        help="Disable post-processing humanization (timing/velocity micro-variations)",
    )
    parser.add_argument(
        "--ghost-notes",
        action="store_true",
        help="Enable ghost notes between solo phrases (auto-enabled by --coltrane)",
    )
    parser.add_argument(
        "--reharmonize",
        choices=["off", "light", "medium", "heavy"],
        default="off",
        help="Coltrane reharmonization density for solo sections (default: off)",
    )
    parser.add_argument(
        "--bass-style",
        choices=["walking", "modal"],
        default="walking",
        help="Bass line style (default: walking)",
    )
    parser.add_argument(
        "--drum-style",
        choices=["swing", "modal"],
        default="swing",
        help="Drum pattern style (default: swing)",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def main(argv=None):
    """Main entry point for the Coltrain CLI."""
    args = parse_args(argv)

    # Validate key
    key_name = args.key
    if key_name not in NOTE_TO_PC:
        print(f"Error: Unknown key '{key_name}'. "
              f"Valid keys: C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B", file=sys.stderr)
        sys.exit(1)
    key_pc = NOTE_TO_PC[key_name]

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)

    # Get form info
    form_name = args.form
    form_info = FORM_TEMPLATES[form_name]
    bars_per_chorus = form_info["bars"]
    tempo = args.tempo
    num_choruses = max(1, args.choruses)

    # Build arrangement structure
    arrangement = build_arrangement(form_name, num_choruses, bars_per_chorus)

    # Calculate total beats from the arrangement
    total_beats = max(s.end_beat for s in arrangement)
    total_bars = total_beats // 4

    # Build chord progression covering the full arrangement
    # The intro uses the last 4 bars of changes, head uses full form, etc.
    # We need chords for the entire arrangement duration.
    all_chords = _build_full_chord_progression(
        form_name, key_pc, arrangement, bars_per_chorus,
        coltrane=args.coltrane,
    )

    # Coltrane defaults: auto-enable certain features
    reharmonize_density = args.reharmonize
    if args.coltrane and reharmonize_density == "off":
        reharmonize_density = "medium"

    do_humanize = not args.no_humanize

    # Print summary
    print(f"\nColtrain By Theory")
    print(f"{'=' * 50}")
    print(f"  Form:       {form_name} ({bars_per_chorus} bars)")
    print(f"  Key:        {key_name}")
    print(f"  Tempo:      {tempo} BPM")
    print(f"  Choruses:   {num_choruses} solo")
    print(f"  Tension:    {args.tension}")
    print(f"  Instrument: {args.instrument}")
    print(f"  Swing:      {args.swing:.3f}")
    print(f"  Bass:       {args.bass_style}")
    print(f"  Drums:      {args.drum_style}")
    print(f"  Humanize:   {'on' if do_humanize else 'off'}")
    if reharmonize_density != "off":
        print(f"  Reharmonize: {reharmonize_density}")
    if args.coltrane:
        print(f"  Coltrane:   enabled")
    if args.seed is not None:
        print(f"  Seed:       {args.seed}")
    print(f"\nArrangement ({total_bars} bars total):")
    for section in arrangement:
        print(repr(section))
    print()

    # Generate arrangement
    print("Generating...", flush=True)
    tracks = generate_arrangement(
        arrangement=arrangement,
        chords=all_chords,
        form_bars=bars_per_chorus,
        key_pc=key_pc,
        tension_curve=args.tension,
        coltrane=args.coltrane,
        swing=(args.swing >= 0.55),
        seed=args.seed,
        humanize=do_humanize,
        tempo=tempo,
        bass_style=args.bass_style,
        drum_style=args.drum_style,
        reharmonize_density=reharmonize_density,
    )

    # Count total notes
    total_notes = sum(len(notes) for notes in tracks.values())
    note_counts = {name: len(notes) for name, notes in tracks.items()}

    # Write MIDI
    notes_written = write_midi(
        tracks, args.output, tempo=tempo, lead_instrument=args.instrument,
    )

    print(f"Saved: {args.output} ({total_bars} bars, {total_notes} notes)")
    for name, count in note_counts.items():
        print(f"  {name:8s}: {count:5d} notes")

    # Write chord chart if requested
    if args.chart:
        chart_path = os.path.splitext(args.output)[0] + "_chart.txt"
        write_chord_chart(all_chords, chart_path)
        print(f"Chart: {chart_path}")

    print()


# ---------------------------------------------------------------------------
# Chord progression for full arrangement
# ---------------------------------------------------------------------------


def _build_full_chord_progression(
    form_name: str,
    key_pc: int,
    arrangement: list,
    bars_per_chorus: int,
    coltrane: bool = False,
) -> list:
    """Build a chord progression that covers the entire arrangement.

    The arrangement has multiple sections (intro, head, solos, etc.), each
    needing appropriate chords. The intro uses the last 4 bars of the form.
    All other sections use complete form choruses.

    Args:
        form_name: Form template name.
        key_pc: Key pitch class.
        arrangement: List of ArrangementSection objects.
        bars_per_chorus: Bars in one chorus.
        coltrane: Enable key-center analysis.

    Returns:
        List of ChordEvent objects covering all arrangement beats.
    """
    from coltrain.theory.chord import ChordEvent

    all_chords = []

    for section in arrangement:
        if section.name == "intro":
            # Intro uses the last 4 bars of the form's changes
            full_chorus = build_chord_progression(
                form_name, key_pc, num_choruses=1, start_beat=0, coltrane=coltrane,
            )
            # Take the last 4 bars (16 beats) worth of chords
            total_form_beats = bars_per_chorus * 4
            intro_start_in_form = max(0, total_form_beats - 16)

            for c in full_chorus:
                c_end = c.start_beat + c.duration_beats
                if c.start_beat < total_form_beats and c_end > intro_start_in_form:
                    clipped_start = max(c.start_beat, intro_start_in_form)
                    clipped_end = min(c_end, total_form_beats)
                    clipped_dur = clipped_end - clipped_start
                    if clipped_dur > 0:
                        # Re-offset to section position
                        new_start = section.start_beat + (clipped_start - intro_start_in_form)
                        all_chords.append(ChordEvent(
                            root_pc=c.root_pc,
                            quality=c.quality,
                            start_beat=new_start,
                            duration_beats=clipped_dur,
                            key_center_pc=c.key_center_pc,
                            function=c.function,
                        ))

        elif section.name == "coda":
            # Coda uses the last 4 bars of the form
            full_chorus = build_chord_progression(
                form_name, key_pc, num_choruses=1, start_beat=0, coltrane=coltrane,
            )
            total_form_beats = bars_per_chorus * 4
            coda_start_in_form = max(0, total_form_beats - 16)

            for c in full_chorus:
                c_end = c.start_beat + c.duration_beats
                if c.start_beat < total_form_beats and c_end > coda_start_in_form:
                    clipped_start = max(c.start_beat, coda_start_in_form)
                    clipped_end = min(c_end, total_form_beats)
                    clipped_dur = clipped_end - clipped_start
                    if clipped_dur > 0:
                        new_start = section.start_beat + (clipped_start - coda_start_in_form)
                        all_chords.append(ChordEvent(
                            root_pc=c.root_pc,
                            quality=c.quality,
                            start_beat=new_start,
                            duration_beats=clipped_dur,
                            key_center_pc=c.key_center_pc,
                            function=c.function,
                        ))

        else:
            # Head, solo, trading: use a full chorus of changes
            section_chords = build_chord_progression(
                form_name, key_pc, num_choruses=1,
                start_beat=float(section.start_beat),
                coltrane=coltrane,
            )
            all_chords.extend(section_chords)

    # Sort by start_beat
    all_chords.sort(key=lambda c: c.start_beat)

    return all_chords


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
