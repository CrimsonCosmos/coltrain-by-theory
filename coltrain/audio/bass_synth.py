"""Karplus-Strong upright bass synthesizer.

Physical-modeling synthesis using a delay-line feedback loop excited by
shaped noise, with body resonance filtering for warmth.
"""

import numpy as np
from . import SAMPLE_RATE


class UprightBassSynth:
    """Karplus-Strong bass with body resonance and expression support."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sr = sample_rate

    def render_note(
        self,
        pitch: int,
        duration_samples: int,
        velocity: int,
        pitch_bend_curve: np.ndarray = None,
        expression_curve: np.ndarray = None,
    ) -> np.ndarray:
        """Render a single bass note.

        Args:
            pitch: MIDI note number (28-55 typical bass range).
            duration_samples: Total output length in samples.
            velocity: MIDI velocity (1-127).
            pitch_bend_curve: Per-sample pitch bend in semitones (None = no bend).
            expression_curve: Per-sample amplitude multiplier 0-1 (None = flat).

        Returns:
            Mono audio buffer as float64 numpy array.
        """
        freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        amp = (velocity / 127.0) ** 1.5  # Slightly exponential velocity curve

        # Delay line length for Karplus-Strong
        delay_len = max(2, int(round(self.sr / freq)))

        # Excitation: shaped noise burst (~15ms) + low-freq thump
        excite_len = max(int(0.015 * self.sr), delay_len)
        excite = np.random.uniform(-1.0, 1.0, excite_len)

        # Low-pass the excitation for warmer tone
        for i in range(1, excite_len):
            excite[i] = 0.5 * excite[i] + 0.5 * excite[i - 1]

        # Add "thump" transient — a short sine burst at sub-fundamental
        thump_len = min(int(0.008 * self.sr), excite_len)
        thump_freq = freq * 0.5
        t_thump = np.arange(thump_len) / self.sr
        thump = 0.6 * np.sin(2 * np.pi * thump_freq * t_thump)
        thump *= np.linspace(1.0, 0.0, thump_len)  # Quick decay
        excite[:thump_len] += thump

        excite *= amp

        # Karplus-Strong synthesis
        output = np.zeros(duration_samples, dtype=np.float64)
        output[:excite_len] = excite[:min(excite_len, duration_samples)]

        # Feedback damping — lower notes ring longer
        damping = 0.993 + 0.003 * (pitch - 28) / (55 - 28)
        damping = min(0.998, max(0.990, damping))

        for i in range(delay_len, duration_samples):
            # Two-point averaging lowpass filter
            avg = 0.5 * (output[i - delay_len] + output[i - delay_len + 1])
            output[i] += damping * avg

            # Apply pitch bend by modulating delay length
            if pitch_bend_curve is not None and i < len(pitch_bend_curve):
                bend_semitones = pitch_bend_curve[i]
                if abs(bend_semitones) > 0.001:
                    bent_freq = freq * (2.0 ** (bend_semitones / 12.0))
                    bent_delay = max(2, int(round(self.sr / bent_freq)))
                    if i >= bent_delay + 1:
                        avg2 = 0.5 * (output[i - bent_delay] + output[i - bent_delay + 1])
                        output[i] = damping * avg2

        # Body resonance: bandpass 50-400Hz mixed at 30%
        output = self._body_resonance(output, freq)

        # Apply expression curve
        if expression_curve is not None:
            n = min(len(output), len(expression_curve))
            output[:n] *= expression_curve[:n]

        # Gentle amplitude envelope to avoid clicks
        attack = min(int(0.005 * self.sr), duration_samples // 4)
        release = min(int(0.010 * self.sr), duration_samples // 4)
        if attack > 0:
            output[:attack] *= np.linspace(0.0, 1.0, attack)
        if release > 0:
            output[-release:] *= np.linspace(1.0, 0.0, release)

        return output

    def _body_resonance(self, signal: np.ndarray, fund_freq: float) -> np.ndarray:
        """Apply bandpass body resonance (50-400Hz) at 30% mix."""
        from scipy.signal import butter, sosfilt

        nyq = self.sr / 2.0
        low = max(50.0, fund_freq * 0.4) / nyq
        high = min(400.0, fund_freq * 3.0) / nyq
        low = min(low, 0.99)
        high = min(high, 0.99)
        if low >= high:
            return signal

        sos = butter(2, [low, high], btype="band", output="sos")
        resonance = sosfilt(sos, signal)
        return 0.7 * signal + 0.3 * resonance
