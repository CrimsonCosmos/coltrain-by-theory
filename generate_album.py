#!/usr/bin/env python3
"""Generate the Coltrain By Theory album -- MIDI only.

Produces 6 original compositions with varied forms, tempos,
instruments, and tension curves.

Usage:
    python generate_album.py
"""

import glob
import os
import sys

from coltrain.main import main as coltrain_main

ALBUM_DIR = os.path.expanduser("~/Desktop/coltrain-album")

TRACKLIST = [
    {
        "title": "Late Meridian",
        "filename": "01_late_meridian.mid",
        "args": [
            "--form", "aaba32",
            "--key", "Eb",
            "--tempo", "130",
            "--choruses", "4",
            "--tension", "arc",
            "--instrument", "sax",
            "--reharmonize", "light",
            "--seed", "1001",
        ],
    },
    {
        "title": "Copper Wire",
        "filename": "02_copper_wire.mid",
        "args": [
            "--form", "blues12",
            "--key", "Bb",
            "--tempo", "100",
            "--choruses", "5",
            "--tension", "wave",
            "--instrument", "trumpet",
            "--seed", "2002",
        ],
    },
    {
        "title": "Prismatic",
        "filename": "03_prismatic.mid",
        "args": [
            "--form", "giantsteps",
            "--key", "B",
            "--tempo", "150",
            "--choruses", "3",
            "--tension", "build",
            "--instrument", "sax",
            "--reharmonize", "medium",
            "--coltrane",
            "--seed", "3003",
        ],
    },
    {
        "title": "Ninth Street Station",
        "filename": "04_ninth_street_station.mid",
        "args": [
            "--form", "rhythm_changes",
            "--key", "F",
            "--tempo", "165",
            "--choruses", "4",
            "--tension", "catharsis",
            "--instrument", "trumpet",
            "--ghost-notes",
            "--seed", "4004",
        ],
    },
    {
        "title": "Still Pools",
        "filename": "05_still_pools.mid",
        "args": [
            "--form", "blues_bird",
            "--key", "Ab",
            "--tempo", "92",
            "--choruses", "4",
            "--tension", "plateau",
            "--instrument", "piano",
            "--reharmonize", "light",
            "--seed", "5005",
        ],
    },
    {
        "title": "Tessellation",
        "filename": "06_tessellation.mid",
        "args": [
            "--form", "aaba32",
            "--key", "Db",
            "--tempo", "145",
            "--choruses", "5",
            "--tension", "arc",
            "--instrument", "sax",
            "--reharmonize", "heavy",
            "--coltrane",
            "--seed", "6006",
        ],
    },
]


def main():
    os.makedirs(ALBUM_DIR, exist_ok=True)

    # Clean old files
    for ext in ("*.mid", "*.mp3"):
        for f in glob.glob(os.path.join(ALBUM_DIR, ext)):
            os.remove(f)

    print("=" * 60)
    print("  COLTRAIN BY THEORY -- Album Generation")
    print("=" * 60)

    for i, track in enumerate(TRACKLIST, 1):
        output = os.path.join(ALBUM_DIR, track["filename"])
        argv = track["args"] + ["-o", output]

        print(f"\n{'─' * 60}")
        print(f"  Track {i}/{len(TRACKLIST)}: {track['title']}")
        print(f"{'─' * 60}")

        coltrain_main(argv)

    print(f"\n{'=' * 60}")
    print(f"  Album complete! {len(TRACKLIST)} tracks written to:")
    print(f"  {ALBUM_DIR}")
    print(f"{'=' * 60}\n")

    for track in TRACKLIST:
        path = os.path.join(ALBUM_DIR, track["filename"])
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"  {track['filename']:40s} {size // 1024:4d} KB")
    print()


if __name__ == "__main__":
    main()
