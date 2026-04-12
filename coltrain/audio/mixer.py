"""Stereo mixer and mastering for Coltrain audio engine.

Handles panning, per-instrument reverb sends, normalization, and file output.
"""

import os
import subprocess

import numpy as np
import soundfile as sf

from . import SAMPLE_RATE
from .effects import SchroederReverb, warm_eq, bus_compress


# Stereo panning positions (-1.0 = hard left, +1.0 = hard right)
PAN_POSITIONS = {
    "melody": -0.2,   # Piano slightly left
    "bass": 0.0,      # Bass center
    "drums": 0.15,    # Drums slightly right
}

# Reverb send levels per instrument (0.0-1.0)
REVERB_SENDS = {
    "melody": 0.35,
    "bass": 0.10,
    "drums": 0.25,
}


def _pan_to_stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    """Convert mono to stereo with constant-power panning.

    Args:
        mono: Mono audio buffer.
        pan: Panning position (-1.0 left, 0.0 center, +1.0 right).

    Returns:
        Stereo buffer of shape (N, 2).
    """
    # Constant-power panning law
    angle = (pan + 1.0) / 2.0 * (np.pi / 2.0)
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)

    stereo = np.zeros((len(mono), 2), dtype=np.float64)
    stereo[:, 0] = mono * left_gain
    stereo[:, 1] = mono * right_gain
    return stereo


def mix_to_stereo(
    instrument_buffers: dict,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Mix individual instrument mono buffers into a mastered stereo output.

    Args:
        instrument_buffers: Dict mapping instrument name ('melody', 'bass', 'drums')
                           to mono numpy arrays.

    Returns:
        Stereo audio buffer of shape (N, 2), normalized.
    """
    # Find max length across all instruments
    max_len = max(len(buf) for buf in instrument_buffers.values())

    # Create reverb instance
    reverb = SchroederReverb(sample_rate)

    stereo_mix = np.zeros((max_len, 2), dtype=np.float64)

    for name, mono in instrument_buffers.items():
        if len(mono) == 0:
            continue

        # Pad to max length
        padded = np.zeros(max_len, dtype=np.float64)
        padded[:len(mono)] = mono

        # Apply reverb send
        send_level = REVERB_SENDS.get(name, 0.2)
        if send_level > 0:
            padded = reverb.process(padded, wet=send_level)

        # Apply warm EQ
        padded = warm_eq(padded, sample_rate)

        # Pan to stereo
        pan = PAN_POSITIONS.get(name, 0.0)
        stereo = _pan_to_stereo(padded, pan)

        stereo_mix += stereo

    # Bus compression on the mix (process L and R independently)
    stereo_mix[:, 0] = bus_compress(stereo_mix[:, 0], sample_rate=sample_rate)
    stereo_mix[:, 1] = bus_compress(stereo_mix[:, 1], sample_rate=sample_rate)

    # Normalize to -1dB peak
    peak = np.max(np.abs(stereo_mix))
    if peak > 0:
        target_peak = 10.0 ** (-1.0 / 20.0)  # -1dB
        stereo_mix *= target_peak / peak

    return stereo_mix


def write_wav(stereo: np.ndarray, output_path: str, sample_rate: int = SAMPLE_RATE):
    """Write stereo audio to WAV file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    sf.write(output_path, stereo, sample_rate, subtype="PCM_16")


def write_mp3(stereo: np.ndarray, output_path: str, sample_rate: int = SAMPLE_RATE):
    """Write stereo audio to MP3 file via ffmpeg."""
    # Write temporary WAV first
    wav_path = output_path + ".tmp.wav"
    write_wav(stereo, wav_path, sample_rate)

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-b:a", "192k", output_path],
            check=True, capture_output=True,
        )
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
