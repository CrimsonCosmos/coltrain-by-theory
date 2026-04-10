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
NUM_TRACKS = 6

# --- Title generation pools ---

ADJECTIVES = [
    "amber", "blue", "broken", "burning", "cold", "crimson", "dark",
    "distant", "drifting", "dusty", "electric", "fading", "floating",
    "foggy", "forgotten", "fractured", "frozen", "gilded", "glass",
    "golden", "hollow", "hushed", "ivory", "jagged", "late", "liquid",
    "lonesome", "lost", "luminous", "marble", "muted", "narrow",
    "neon", "obsidian", "open", "pale", "quiet", "restless", "rusted",
    "scattered", "shadowed", "sharp", "shifting", "silent", "silver",
    "sleepless", "slow", "smoked", "solitary", "southern", "spare",
    "steep", "still", "sunken", "tangled", "tarnished", "thin",
    "twilight", "unspoken", "vacant", "velvet", "warm", "winding",
    "worn",
]

NOUNS = [
    "alley", "arc", "avenue", "axis", "basin", "bloom", "boulevard",
    "bridge", "cadence", "canyon", "cathedral", "circuit", "clearing",
    "clockwork", "coastline", "contour", "corridor", "crossroads",
    "current", "dawn", "descent", "doorway", "dusk", "echo", "edge",
    "embers", "equation", "estuary", "evening", "fever", "field",
    "flight", "fog", "fracture", "garden", "gate", "geometry", "glass",
    "gravity", "harbor", "haze", "horizon", "inlet", "junction",
    "lantern", "lattice", "ledge", "light", "meridian", "midnight",
    "mirror", "monument", "mosaic", "motion", "nightfall", "orbit",
    "overpass", "passage", "pavilion", "pendulum", "pier", "plaza",
    "pool", "prism", "pulse", "rain", "reflection", "ridge", "river",
    "rooftop", "rotation", "ruins", "scaffold", "shadow", "signal",
    "skyline", "smoke", "solstice", "spiral", "station", "stone",
    "summit", "surface", "territory", "thread", "threshold", "tide",
    "tower", "transit", "tributary", "undertow", "viaduct", "voltage",
    "waterline", "window",
]

# Patterns: functions that combine words into a title
def _title_adj_noun():
    return f"{random.choice(ADJECTIVES).title()} {random.choice(NOUNS).title()}"

def _title_the_noun():
    return f"The {random.choice(NOUNS).title()}"

def _title_noun_of_noun():
    return f"{random.choice(NOUNS).title()} Of {random.choice(NOUNS).title()}"

def _title_single_noun():
    return random.choice(NOUNS).title()

def _title_adj_adj_noun():
    a1 = random.choice(ADJECTIVES)
    a2 = random.choice([a for a in ADJECTIVES if a != a1])
    return f"{a1.title()} {a2.title()} {random.choice(NOUNS).title()}"

_TITLE_GENERATORS = [
    _title_adj_noun, _title_adj_noun, _title_adj_noun,  # weighted toward this
    _title_the_noun,
    _title_noun_of_noun,
    _title_single_noun,
    _title_adj_adj_noun,
]


def generate_title(existing_titles):
    """Generate a unique random song title."""
    for _ in range(50):
        title = random.choice(_TITLE_GENERATORS)()
        if title not in existing_titles:
            return title
    return f"Track {len(existing_titles) + 1}"


def sanitize_filename(title):
    """Convert a title to a safe filename slug."""
    return title.lower().replace(" ", "_").replace("'", "")


# --- Musical parameter pools ---

FORMS = ["blues12", "blues_bird", "rhythm_changes", "aaba32", "giantsteps"]
KEYS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
INSTRUMENTS = ["trumpet", "piano"]
TENSIONS = ["arc", "build", "wave", "plateau", "catharsis"]
REHARMONIZE_LEVELS = ["off", "light", "medium", "heavy"]


def generate_tracklist(num_tracks, album_seed=None):
    """Generate a randomized tracklist ensuring variety."""
    if album_seed is not None:
        random.seed(album_seed)

    tracks = []
    used_titles = set()
    used_forms = []

    # Ensure we use a variety of forms — shuffle and cycle if needed
    form_bag = list(FORMS)
    random.shuffle(form_bag)

    for i in range(num_tracks):
        # Pick form: cycle through shuffled bag to avoid repeats
        if not form_bag:
            form_bag = list(FORMS)
            random.shuffle(form_bag)
        form = form_bag.pop()

        # Title
        title = generate_title(used_titles)
        used_titles.add(title)

        # Key: random
        key = random.choice(KEYS)

        # Tempo: form-appropriate ranges
        if form == "giantsteps":
            tempo = random.randint(130, 170)
        elif form in ("blues12", "blues_bird"):
            tempo = random.randint(88, 145)
        elif form == "rhythm_changes":
            tempo = random.randint(140, 180)
        else:  # aaba32
            tempo = random.randint(100, 160)

        # Choruses: 3-5
        choruses = random.randint(3, 5)

        # Instrument: random but avoid 3+ in a row
        if len(tracks) >= 2 and tracks[-1]["instrument"] == tracks[-2]["instrument"]:
            inst = random.choice([x for x in INSTRUMENTS if x != tracks[-1]["instrument"]])
        else:
            inst = random.choice(INSTRUMENTS)

        # Tension curve: random
        tension = random.choice(TENSIONS)

        # Reharmonize: more likely on complex forms
        if form in ("giantsteps", "rhythm_changes"):
            reharm = random.choice(["off", "light", "medium", "medium", "heavy"])
        elif form == "blues_bird":
            reharm = random.choice(["off", "light", "light", "medium"])
        else:
            reharm = random.choice(["off", "off", "light", "medium"])

        # Coltrane mode: only with giantsteps or complex reharm
        coltrane = form == "giantsteps" or (reharm == "heavy" and random.random() < 0.5)

        # Unique seed per track (derived from album seed or random)
        track_seed = random.randint(1, 99999)

        tracks.append({
            "title": title,
            "form": form,
            "key": key,
            "tempo": tempo,
            "choruses": choruses,
            "instrument": inst,
            "tension": tension,
            "reharmonize": reharm,
            "coltrane": coltrane,
            "seed": track_seed,
        })

    return tracks


def main():
    # Parse optional --seed argument
    album_seed = None
    if "--seed" in sys.argv:
        idx = sys.argv.index("--seed")
        if idx + 1 < len(sys.argv):
            album_seed = int(sys.argv[idx + 1])

    os.makedirs(ALBUM_DIR, exist_ok=True)

    # Clean old files
    for ext in ("*.mid", "*.mp3"):
        for f in glob.glob(os.path.join(ALBUM_DIR, ext)):
            os.remove(f)

    tracklist = generate_tracklist(NUM_TRACKS, album_seed)

    print("=" * 60)
    print("  COLTRAIN BY THEORY -- Album Generation")
    if album_seed is not None:
        print(f"  Album seed: {album_seed}")
    print("=" * 60)

    for i, track in enumerate(tracklist, 1):
        slug = sanitize_filename(track["title"])
        filename = f"{i:02d}_{slug}.mid"
        output = os.path.join(ALBUM_DIR, filename)

        args = [
            "--form", track["form"],
            "--key", track["key"],
            "--tempo", str(track["tempo"]),
            "--choruses", str(track["choruses"]),
            "--tension", track["tension"],
            "--instrument", track["instrument"],
            "--seed", str(track["seed"]),
        ]
        if track["reharmonize"] != "off":
            args += ["--reharmonize", track["reharmonize"]]
        if track["coltrane"]:
            args += ["--coltrane"]
        args += ["-o", output]

        print(f"\n{'~'*60}")
        print(f"  Track {i}/{NUM_TRACKS}: {track['title']}")
        print(f"  {track['form']} in {track['key']} @ {track['tempo']} BPM"
              f"  ({track['instrument']}, {track['tension']})")
        print(f"{'~'*60}")

        coltrain_main(args)

    print(f"\n{'=' * 60}")
    print(f"  Album complete! {NUM_TRACKS} tracks written to:")
    print(f"  {ALBUM_DIR}")
    print(f"{'=' * 60}\n")

    for i, track in enumerate(tracklist, 1):
        slug = sanitize_filename(track["title"])
        filename = f"{i:02d}_{slug}.mid"
        path = os.path.join(ALBUM_DIR, filename)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"  {filename:45s} {size // 1024:4d} KB")
    print()


if __name__ == "__main__":
    main()
