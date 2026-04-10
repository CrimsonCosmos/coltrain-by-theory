# Coltrain By Theory

Rule-based jazz MIDI generation using pure music theory.

## Quick Start

```bash
pip install -r requirements.txt
python -m coltrain.main --form giantsteps --key B --tempo 286 --coltrane
```

## Features

- **Walking bass** -- chord-tone targeting, chromatic approach notes, voice-leading continuity
- **Swing drums** -- ride cymbal pattern, hi-hat on 2 and 4, feathered kick, snare comping, fills
- **Piano comping** -- shell, rootless, and drop-2 voicings with smooth voice leading and syncopated rhythms
- **Head melody** -- thematic composed lines targeting chord tones on strong beats
- **Improvised solos** -- three-tier tension system (melodic, motivic, sheets of sound)
- **Trading fours** -- alternating 4-bar melody and drum exchanges
- **Full arrangement** -- intro, head in, solos, trading, head out, coda with intensity curves
- **Coltrane multi-tonic system** -- key-center awareness across three tonal centers (Giant Steps)
- **Motivic development** -- motif capture, inversion, retrograde, augmentation, diminution
- **Sheets of sound** -- rapid 16th-note digital patterns and arpeggio runs at high tension
- **Tension curves** -- arc, build, and wave curves driving note density, register, and vocabulary
- **Swing feel** -- configurable swing ratio applied to all eighth-note subdivisions
- **Form templates** -- blues (12-bar), Bird blues, rhythm changes, AABA, Giant Steps

## Architecture

```
coltrain/
  theory/          Music theory engine
    pitch.py         Pitch class representation and note/MIDI conversion
    chord.py         Chord types, parsing, tone classification
    scale.py         Scale/mode system with chord-scale mapping
    voice_leading.py Voicing construction and voice-leading optimization

  generation/      Note generation
    bass.py          Walking bass and two-feel generators
    drums.py         Swing drum pattern and fill generators
    piano.py         Piano comping with voicing and rhythm patterns
    melody.py        Head melody, solo, and trading-fours generators
    arrangement.py   Master orchestrator and form templates

  midi/            Output
    writer.py        Multi-track MIDI file writer

  main.py          CLI entry point
```

## CLI Reference

| Argument       | Values                                                | Default     |
|----------------|-------------------------------------------------------|-------------|
| `--form`       | `blues12`, `blues_bird`, `rhythm_changes`, `aaba32`, `giantsteps` | `blues12`   |
| `--key`        | `C`, `Db`, `D`, `Eb`, `E`, `F`, `Gb`, `G`, `Ab`, `A`, `Bb`, `B` | `C`         |
| `--tempo`      | Integer BPM                                            | `140`       |
| `--choruses`   | Integer (solo choruses)                                | `2`         |
| `--tension`    | `arc`, `build`, `wave`                                 | `arc`       |
| `--coltrane`   | Flag (enables multi-tonic features)                    | off         |
| `--instrument` | `sax`, `trumpet`, `piano`                              | `sax`       |
| `--swing`      | Float ratio (0.5=straight, 0.667=triplet)              | `0.667`     |
| `--seed`       | Integer (reproducible output)                          | random      |
| `--output`     | File path                                              | `output.mid`|
| `--chart`      | Flag (write chord chart text file)                     | off         |

## Examples

```bash
# 12-bar blues in Bb at medium tempo
python -m coltrain.main --form blues12 --key Bb --tempo 120 --choruses 3

# Giant Steps at Coltrane tempo with sheets of sound
python -m coltrain.main --form giantsteps --key B --tempo 286 --choruses 4 --coltrane --tension build

# Rhythm changes for trumpet
python -m coltrain.main --form rhythm_changes --key Bb --instrument trumpet --choruses 2

# Reproducible output with seed
python -m coltrain.main --form aaba32 --key C --seed 42 -o my_tune.mid --chart
```

## Credits

Draws inspiration from algorithmic composition research, jazz theory pedagogy,
and the musical language of John Coltrane, Charlie Parker, and the bebop tradition.
