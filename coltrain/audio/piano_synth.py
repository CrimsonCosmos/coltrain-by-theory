"""Additive synthesis jazz piano.

Multi-harmonic synthesis with inharmonicity, velocity-dependent brightness,
and sustain pedal support for a warm, realistic piano tone.
"""

import numpy as np
from . import SAMPLE_RATE


class PianoSynth:
    """Additive synthesis piano with 8 harmonics and inharmonicity."""

    # Inharmonicity coefficient (higher = more detuned upper partials)
    B = 0.0004

    # Relative amplitudes for 8 harmonics (fundamental strongest)
    HARMONIC_AMPS = np.array([1.0, 0.7, 0.4, 0.25, 0.15, 0.10, 0.06, 0.03])

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sr = sample_rate

    def render_note(
        self,
        pitch: int,
        duration_samples: int,
        velocity: int,
        sustain: bool = False,
    ) -> np.ndarray:
        """Render a single piano note.

        Args:
            pitch: MIDI note number.
            duration_samples: Total output length in samples.
            velocity: MIDI velocity (1-127).
            sustain: If True, extend decay 3x (sustain pedal held).

        Returns:
            Mono audio buffer as float64 numpy array.
        """
        freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        vel_norm = velocity / 127.0
        amp = vel_norm ** 1.3  # Slightly compressed dynamics

        # Velocity controls brightness: louder = more harmonics
        brightness = 0.3 + 0.7 * vel_norm

        t = np.arange(duration_samples, dtype=np.float64) / self.sr

        signal = np.zeros(duration_samples, dtype=np.float64)

        for h in range(1, 9):
            # Inharmonicity: h_freq = f * h * sqrt(1 + B * h^2)
            h_freq = freq * h * np.sqrt(1.0 + self.B * h * h)

            if h_freq >= self.sr / 2:
                break  # Skip harmonics above Nyquist

            # Amplitude rolls off for higher harmonics, modulated by brightness
            h_amp = self.HARMONIC_AMPS[h - 1] * (brightness ** (h - 1))

            # Each harmonic has its own decay rate (higher = faster decay)
            # Lower notes decay slower
            base_decay = 1.5 + 2.0 * (pitch - 21) / (108 - 21)
            h_decay = base_decay * (1.0 + 0.5 * (h - 1))
            if sustain:
                h_decay /= 3.0  # Sustain pedal extends decay 3x

            envelope = np.exp(-h_decay * t)
            signal += h_amp * envelope * np.sin(2 * np.pi * h_freq * t)

        signal *= amp

        # Fast attack (5ms)
        attack_samples = min(int(0.005 * self.sr), duration_samples // 4)
        if attack_samples > 0:
            signal[:attack_samples] *= np.linspace(0.0, 1.0, attack_samples)

        # Release (10ms anti-click at end)
        release_samples = min(int(0.010 * self.sr), duration_samples // 4)
        if release_samples > 0:
            signal[-release_samples:] *= np.linspace(1.0, 0.0, release_samples)

        return signal
