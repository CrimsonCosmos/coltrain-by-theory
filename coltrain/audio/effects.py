"""Audio effects: reverb, EQ, and compression.

Provides a jazz-club ambiance via Schroeder reverb, warm EQ rolloff,
and gentle bus compression.
"""

import numpy as np
from . import SAMPLE_RATE


# ---------------------------------------------------------------------------
# Schroeder reverb
# ---------------------------------------------------------------------------

class SchroederReverb:
    """Schroeder reverb: 4 parallel comb filters → 2 series allpass.

    Tuned for a small jazz club with RT60 ~1.2s.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, rt60: float = 1.2):
        self.sr = sample_rate
        self.rt60 = rt60

        # Comb filter delay lengths (in samples) — mutually prime
        self.comb_delays = [
            int(0.0297 * sr) for sr in [sample_rate]
        ]
        self.comb_delays = [
            int(0.0297 * sample_rate),
            int(0.0371 * sample_rate),
            int(0.0411 * sample_rate),
            int(0.0437 * sample_rate),
        ]

        # Comb feedback gains derived from RT60
        self.comb_gains = []
        for delay in self.comb_delays:
            delay_sec = delay / sample_rate
            g = 10.0 ** (-3.0 * delay_sec / rt60)
            self.comb_gains.append(min(g, 0.98))

        # Allpass delays
        self.ap_delays = [
            int(0.0050 * sample_rate),
            int(0.0017 * sample_rate),
        ]
        self.ap_gain = 0.7

    def process(self, signal: np.ndarray, wet: float = 0.3) -> np.ndarray:
        """Apply reverb to mono signal.

        Args:
            signal: Input mono audio.
            wet: Wet/dry mix (0.0 = dry, 1.0 = full wet).

        Returns:
            Processed mono audio (same length as input).
        """
        n = len(signal)

        # Parallel comb filters
        comb_sum = np.zeros(n, dtype=np.float64)
        for delay, gain in zip(self.comb_delays, self.comb_gains):
            buf = np.zeros(n, dtype=np.float64)
            for i in range(n):
                if i >= delay:
                    buf[i] = signal[i] + gain * buf[i - delay]
                else:
                    buf[i] = signal[i]
            comb_sum += buf
        comb_sum /= len(self.comb_delays)

        # Series allpass filters
        out = comb_sum
        for delay in self.ap_delays:
            ap_out = np.zeros(n, dtype=np.float64)
            g = self.ap_gain
            for i in range(n):
                if i >= delay:
                    ap_out[i] = -g * out[i] + out[i - delay] + g * ap_out[i - delay]
                else:
                    ap_out[i] = -g * out[i]
            out = ap_out

        return (1.0 - wet) * signal + wet * out


# ---------------------------------------------------------------------------
# Warm EQ
# ---------------------------------------------------------------------------

def warm_eq(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Butterworth lowpass at 10kHz for warmth."""
    from scipy.signal import butter, sosfilt

    nyq = sample_rate / 2.0
    cutoff = min(10000.0 / nyq, 0.99)
    sos = butter(2, cutoff, btype="low", output="sos")
    return sosfilt(sos, signal)


# ---------------------------------------------------------------------------
# Bus compression
# ---------------------------------------------------------------------------

def bus_compress(
    signal: np.ndarray,
    threshold_db: float = -12.0,
    ratio: float = 3.0,
    attack_ms: float = 30.0,
    release_ms: float = 200.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Simple bus compressor.

    Args:
        signal: Input audio.
        threshold_db: Compression threshold in dB.
        ratio: Compression ratio (e.g., 3.0 = 3:1).
        attack_ms: Attack time in milliseconds.
        release_ms: Release time in milliseconds.

    Returns:
        Compressed audio.
    """
    threshold = 10.0 ** (threshold_db / 20.0)
    attack_coeff = np.exp(-1.0 / (attack_ms / 1000.0 * sample_rate))
    release_coeff = np.exp(-1.0 / (release_ms / 1000.0 * sample_rate))

    n = len(signal)
    output = np.zeros(n, dtype=np.float64)
    envelope = 0.0

    for i in range(n):
        level = abs(signal[i])
        if level > envelope:
            envelope = attack_coeff * envelope + (1 - attack_coeff) * level
        else:
            envelope = release_coeff * envelope + (1 - release_coeff) * level

        if envelope > threshold:
            gain_reduction = threshold + (envelope - threshold) / ratio
            gain = gain_reduction / max(envelope, 1e-10)
        else:
            gain = 1.0

        output[i] = signal[i] * gain

    return output
