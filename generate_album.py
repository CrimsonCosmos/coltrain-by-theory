#!/usr/bin/env python3
"""Generate the Coltrain By Theory album -- MIDI only.

Produces 6 original compositions with randomly varied forms, tempos,
instruments, keys, and tension curves. Each run generates a unique album.

Usage:
    python generate_album.py
    python generate_album.py --seed 42   # reproducible album
"""

import glob
import os
import random
import sys

from coltrain.main import main as coltrain_main

ALBUM_DIR = os.path.expanduser("~/Desktop/coltrain-album")
NUM_TRACKS = 12

def _next_track_number(album_dir):
    """Find the highest existing track number in album_dir and return the next one."""
    highest = 0
    for f in glob.glob(os.path.join(album_dir, "*.mid")):
        name = os.path.splitext(os.path.basename(f))[0]  # "007" or "02_slug"
        try:
            num = int(name.split("_")[0])
            highest = max(highest, num)
        except (ValueError, IndexError):
            pass
    return highest + 1


# --- Musical parameter pools ---

FORMS = ["coltrain"]
KEYS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
TENSIONS = ["arc", "build", "wave", "plateau", "catharsis"]
DRUM_STYLES = ["swing", "modal", "brushes"]
REHARMONIZE_LEVELS = ["off", "light", "medium", "heavy"]


def generate_tracklist(num_tracks, album_dir, album_seed=None, style=None):
    """Generate a randomized tracklist ensuring variety.

    Args:
        style: Optional style preset. "giantsteps" locks all tracks to
               giantsteps form with coltrane mode and swing drums.
    """
    if album_seed is not None:
        random.seed(album_seed)

    tracks = []
    start_num = _next_track_number(album_dir)

    # Ensure we use a variety of forms — shuffle and cycle if needed
    form_bag = list(FORMS)
    random.shuffle(form_bag)

    for i in range(num_tracks):
        # --- Meter selection: ~15% chance of odd meter ---
        meter = "4/4"
        if style not in ("giantsteps", "coltrain"):
            r_meter = random.random()
            if r_meter < 0.10:
                meter = "5/4"
            elif r_meter < 0.15:
                meter = "7/4"

        if meter == "5/4":
            form = "modal_5"
        elif meter == "7/4":
            form = "modal_7"
        elif style == "giantsteps":
            form = "giantsteps"
        elif style == "coltrain":
            form = "coltrain"
        else:
            # Pick form: cycle through shuffled bag to avoid repeats
            if not form_bag:
                form_bag = list(FORMS)
                random.shuffle(form_bag)
            form = form_bag.pop()

        # Track number (always incrementing)
        track_num = start_num + i

        # Key: random
        key = random.choice(KEYS)

        # Tempo: form-appropriate ranges
        if meter == "5/4":
            tempo = random.randint(130, 175)
        elif meter == "7/4":
            tempo = random.randint(100, 150)
        elif style == "giantsteps":
            tempo = random.randint(130, 170)
        elif form == "giantsteps":
            tempo = random.randint(130, 170)
        elif form == "coltrain":
            tempo = random.randint(100, 165)
        elif form in ("blues12", "blues_bird"):
            tempo = random.randint(88, 145)
        elif form == "rhythm_changes":
            tempo = random.randint(140, 180)
        else:  # aaba32
            tempo = random.randint(100, 160)

        # Choruses: 3-5
        choruses = random.randint(3, 5)

        # Tension curve: random
        tension = random.choice(TENSIONS)

        # Reharmonize
        if style == "giantsteps":
            reharm = random.choice(["medium", "medium", "heavy"])
        elif form == "coltrain":
            reharm = random.choice(["light", "medium", "medium", "heavy"])
        elif form in ("giantsteps", "rhythm_changes"):
            reharm = random.choice(["off", "light", "medium", "medium", "heavy"])
        elif form == "blues_bird":
            reharm = random.choice(["off", "light", "light", "medium"])
        else:
            reharm = random.choice(["off", "off", "light", "medium"])

        # Coltrane mode — disabled for odd meters (too complex)
        if meter != "4/4":
            coltrane = False
        elif style == "giantsteps":
            coltrane = True
        elif form == "coltrain":
            coltrane = random.random() < 0.6  # 60% chance — bridge has its own modulations
        else:
            coltrane = form == "giantsteps" or (reharm == "heavy" and random.random() < 0.5)

        # Drum style — piano trio defaults to brushes (stirring, intimate texture)
        # Odd meters: always brushes (intimate, Take Five feel)
        if meter != "4/4":
            drum_style = "brushes"
        elif tempo >= 170:
            # Very fast tempos: swing sticks sound better
            drum_style = "swing"
        elif tempo >= 150:
            # Fast tempos: mostly swing, occasional brushes
            drum_style = "brushes" if random.random() < 0.30 else "swing"
        elif form == "giantsteps":
            # Giant Steps: swing at fast tempos, brushes at moderate
            drum_style = "brushes" if tempo < 140 else "swing"
        else:
            # Under 150 BPM: heavily favor brushes (80%), occasional modal/swing
            r = random.random()
            if r < 0.80:
                drum_style = "brushes"
            elif r < 0.90:
                drum_style = "modal"
            else:
                drum_style = "swing"

        # Unique seed per track (derived from album seed or random)
        track_seed = random.randint(1, 99999)

        tracks.append({
            "number": track_num,
            "form": form,
            "key": key,
            "tempo": tempo,
            "choruses": choruses,
            "tension": tension,
            "reharmonize": reharm,
            "coltrane": coltrane,
            "drum_style": drum_style,
            "drum_solo": False,
            "bass_solo": False,
            "seed": track_seed,
            "meter": meter,
        })

    # Assign drum and bass solos to specific tracks for variety
    drum_solo_indices = {2, 9, 10}    # 0-indexed
    bass_solo_indices = {4, 5}         # 0-indexed
    for idx, t in enumerate(tracks):
        if idx in drum_solo_indices:
            t["drum_solo"] = True
        if idx in bass_solo_indices:
            t["bass_solo"] = True

    return tracks


def main():
    # Parse optional --seed and --style arguments
    album_seed = None
    style = None
    if "--seed" in sys.argv:
        idx = sys.argv.index("--seed")
        if idx + 1 < len(sys.argv):
            album_seed = int(sys.argv[idx + 1])
    if "--style" in sys.argv:
        idx = sys.argv.index("--style")
        if idx + 1 < len(sys.argv):
            style = sys.argv[idx + 1]

    os.makedirs(ALBUM_DIR, exist_ok=True)

    # Wipe existing tracks before generating
    for old_mid in glob.glob(os.path.join(ALBUM_DIR, "*.mid")):
        os.remove(old_mid)

    tracklist = generate_tracklist(NUM_TRACKS, ALBUM_DIR, album_seed, style=style)

    print("=" * 60)
    print("  COLTRAIN BY THEORY -- Album Generation")
    if album_seed is not None:
        print(f"  Album seed: {album_seed}")
    print("=" * 60)

    for i, track in enumerate(tracklist, 1):
        num = track["number"]
        filename = f"{num:03d}.mid"
        output = os.path.join(ALBUM_DIR, filename)

        args = [
            "--form", track["form"],
            "--key", track["key"],
            "--tempo", str(track["tempo"]),
            "--choruses", str(track["choruses"]),
            "--tension", track["tension"],
            "--seed", str(track["seed"]),
        ]
        if track["reharmonize"] != "off":
            args += ["--reharmonize", track["reharmonize"]]
        if track["coltrane"]:
            args += ["--coltrane"]
        if track.get("drum_style", "swing") != "swing":
            args += ["--drum-style", track["drum_style"]]
        if track.get("drum_solo"):
            args += ["--drum-solo"]
        if track.get("bass_solo"):
            args += ["--bass-solo"]
        if track.get("meter", "4/4") != "4/4":
            args += ["--meter", track["meter"]]
        args += ["-o", output]

        print(f"\n{'~'*60}")
        print(f"  Track {i}/{NUM_TRACKS}: #{num}")
        extras = []
        if track.get("coltrane"):
            extras.append("coltrane")
        if track.get("drum_solo"):
            extras.append("drum solo")
        if track.get("bass_solo"):
            extras.append("bass solo")
        meter_str = track.get("meter", "4/4")
        if meter_str != "4/4":
            extras.append(meter_str)
        extras_str = ", " + ", ".join(extras) if extras else ""
        print(f"  {track['form']} in {track['key']} @ {track['tempo']} BPM"
              f"  (piano, {track['tension']}, {track['drum_style']}{extras_str})")
        print(f"{'~'*60}")

        coltrain_main(args)

    print(f"\n{'=' * 60}")
    print(f"  Album complete! {NUM_TRACKS} tracks written to:")
    print(f"  {ALBUM_DIR}")
    print(f"{'=' * 60}\n")

    for track in tracklist:
        num = track["number"]
        filename = f"{num:03d}.mid"
        path = os.path.join(ALBUM_DIR, filename)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"  {filename:20s} {size // 1024:4d} KB")
    print()


if __name__ == "__main__":
    main()
