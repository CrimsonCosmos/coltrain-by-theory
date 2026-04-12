"""Hybrid synthesis drum kit.

Synthesizes kick, snare, hi-hat, ride, and crash using combinations of
sine sweeps, filtered noise, and metallic partials.
"""

import numpy as np
from . import SAMPLE_RATE

# GM drum map pitches
KICK = 36
SNARE = 38
SIDE_STICK = 37
CLOSED_HH = 42
OPEN_HH = 46
RIDE = 51
RIDE_BELL = 53
CRASH = 49
HI_TOM = 50
MID_TOM = 47
LO_TOM = 45
FLOOR_TOM = 43


class DrumSynth:
    """Hybrid synthesis drum kit with CC4 hi-hat control."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sr = sample_rate

    def render_hit(
        self,
        pitch: int,
        velocity: int,
        cc4_value: int = 0,
    ) -> np.ndarray:
        """Render a single drum hit.

        Args:
            pitch: GM drum map MIDI note number.
            velocity: MIDI velocity (1-127).
            cc4_value: Hi-hat foot controller (0=closed, 127=open).

        Returns:
            Mono audio buffer as float64 numpy array.
        """
        vel = velocity / 127.0

        if pitch == KICK:
            return self._kick(vel)
        elif pitch == SNARE:
            return self._snare(vel)
        elif pitch == SIDE_STICK:
            return self._side_stick(vel)
        elif pitch in (CLOSED_HH, OPEN_HH):
            return self._hihat(vel, cc4_value, is_open=(pitch == OPEN_HH))
        elif pitch == RIDE:
            return self._ride(vel, bell=False)
        elif pitch == RIDE_BELL:
            return self._ride(vel, bell=True)
        elif pitch == CRASH:
            return self._crash(vel)
        elif pitch in (HI_TOM, MID_TOM, LO_TOM, FLOOR_TOM):
            return self._tom(vel, pitch)
        else:
            # Unknown percussion — short noise burst
            n = int(0.05 * self.sr)
            out = np.random.uniform(-1, 1, n) * vel * 0.3
            out *= np.exp(-np.linspace(0, 10, n))
            return out

    def _kick(self, vel: float) -> np.ndarray:
        """Kick drum: sine sweep 150→50Hz + noise transient."""
        dur = int(0.25 * self.sr)
        t = np.arange(dur, dtype=np.float64) / self.sr

        # Pitch sweep: starts at 150Hz, drops to 50Hz
        freq = 50.0 + 100.0 * np.exp(-30.0 * t)
        phase = np.cumsum(2 * np.pi * freq / self.sr)
        body = np.sin(phase)

        # Amplitude envelope
        env = np.exp(-6.0 * t)
        body *= env

        # Noise transient for click
        click_len = int(0.003 * self.sr)
        click = np.random.uniform(-1, 1, click_len) * 0.4
        click *= np.linspace(1, 0, click_len)

        out = body * 0.8
        out[:click_len] += click

        return out * vel

    def _snare(self, vel: float) -> np.ndarray:
        """Snare: 200Hz body + filtered noise (snare wires)."""
        dur = int(0.18 * self.sr)
        t = np.arange(dur, dtype=np.float64) / self.sr

        # Body tone
        body = np.sin(2 * np.pi * 200 * t) * np.exp(-15.0 * t) * 0.5

        # Snare wires: bandpass noise
        noise = np.random.uniform(-1, 1, dur)
        # Simple lowpass for warmth
        for i in range(1, dur):
            noise[i] = 0.3 * noise[i] + 0.7 * noise[i - 1]
        wire_env = np.exp(-8.0 * t)
        wires = noise * wire_env * 0.6

        return (body + wires) * vel

    def _side_stick(self, vel: float) -> np.ndarray:
        """Side stick: short high click."""
        dur = int(0.04 * self.sr)
        t = np.arange(dur, dtype=np.float64) / self.sr
        click = np.sin(2 * np.pi * 800 * t) * np.exp(-40.0 * t) * 0.5
        noise = np.random.uniform(-1, 1, dur) * np.exp(-50.0 * t) * 0.3
        return (click + noise) * vel

    def _hihat(self, vel: float, cc4: int, is_open: bool) -> np.ndarray:
        """Hi-hat: bandpass noise with CC4-controlled decay."""
        # Decay time: closed 50ms → open 450ms, CC4 interpolates
        openness = max(cc4 / 127.0, 1.0 if is_open else 0.0)
        decay_time = 0.05 + 0.40 * openness
        dur = int(decay_time * 3 * self.sr)  # 3x decay for full ring-out
        t = np.arange(dur, dtype=np.float64) / self.sr

        # Metallic noise: bandpass 5-15kHz via spectral shaping
        noise = np.random.uniform(-1, 1, dur)

        # Highpass at ~5kHz (simple first-order)
        alpha_hp = 0.7
        hp = np.zeros(dur)
        hp[0] = noise[0]
        for i in range(1, dur):
            hp[i] = alpha_hp * (hp[i - 1] + noise[i] - noise[i - 1])

        # Lowpass at ~15kHz
        alpha_lp = 0.3
        for i in range(1, dur):
            hp[i] = alpha_lp * hp[i] + (1 - alpha_lp) * hp[i - 1]

        env = np.exp(-t / decay_time)
        return hp * env * vel * 0.4

    def _ride(self, vel: float, bell: bool = False) -> np.ndarray:
        """Ride cymbal: metallic partials + filtered noise, long decay."""
        dur = int(1.5 * self.sr)
        t = np.arange(dur, dtype=np.float64) / self.sr

        # Multiple metallic partials (non-harmonic)
        freqs = [340, 620, 880, 1200, 2400, 3800]
        amps = [0.3, 0.2, 0.15, 0.1, 0.08, 0.05]
        if bell:
            freqs = [f * 1.5 for f in freqs]
            amps = [a * 1.5 for a in amps]

        signal = np.zeros(dur)
        for f, a in zip(freqs, amps):
            if f < self.sr / 2:
                decay = 2.0 + 1.0 * (f / freqs[0])
                signal += a * np.sin(2 * np.pi * f * t) * np.exp(-decay * t)

        # Add filtered noise for shimmer
        noise = np.random.uniform(-1, 1, dur)
        alpha = 0.15
        for i in range(1, dur):
            noise[i] = alpha * noise[i] + (1 - alpha) * noise[i - 1]
        signal += noise * np.exp(-3.0 * t) * 0.1

        return signal * vel

    def _crash(self, vel: float) -> np.ndarray:
        """Crash cymbal: wide-band noise + partials, long decay."""
        dur = int(2.0 * self.sr)
        t = np.arange(dur, dtype=np.float64) / self.sr

        # Wide-band noise
        noise = np.random.uniform(-1, 1, dur) * np.exp(-2.0 * t) * 0.4

        # Metallic partials
        freqs = [260, 540, 940, 1600, 2800, 5000]
        signal = np.zeros(dur)
        for f in freqs:
            if f < self.sr / 2:
                signal += 0.1 * np.sin(2 * np.pi * f * t) * np.exp(-2.5 * t)

        return (noise + signal) * vel

    def _tom(self, vel: float, pitch: int) -> np.ndarray:
        """Tom: sine body with pitch-dependent frequency + noise."""
        # Map GM toms to frequencies
        tom_freqs = {HI_TOM: 200, MID_TOM: 160, LO_TOM: 120, FLOOR_TOM: 85}
        freq = tom_freqs.get(pitch, 140)
        dur = int(0.35 * self.sr)
        t = np.arange(dur, dtype=np.float64) / self.sr

        body = np.sin(2 * np.pi * freq * t) * np.exp(-8.0 * t) * 0.6
        noise = np.random.uniform(-1, 1, dur) * np.exp(-12.0 * t) * 0.2

        return (body + noise) * vel
