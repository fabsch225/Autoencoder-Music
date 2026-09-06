"""Rendering helpers: play a melody with real piano samples and write a WAV.

Reuses the `music` sound library for sample assembly / WAV writing and the
University of Iowa Musical Instrument Samples for the actual piano sound.
"""

import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import aifc

import numpy as np

import music

SAMPLE_RATE = 44100
NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def midi_to_name(midi):
    """MIDI note number -> pitch name used by the Iowa sample files (C4=60)."""
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def read_aiff(path):
    """Decode an AIFF file to mono float samples in [-1, 1]."""
    with aifc.open(path, "rb") as f:
        nch = f.getnchannels()
        sw = f.getsampwidth()
        raw = f.readframes(f.getnframes())
    if sw == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0
    elif sw == 2:
        data = np.frombuffer(raw, dtype=">i2").astype(np.float64)
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        data = ((b[:, 0].astype(np.int32) << 16)
                | (b[:, 1].astype(np.int32) << 8)
                | b[:, 2].astype(np.int32)).astype(np.float64)
        data[data >= 2 ** 23] -= 2 ** 24
    elif sw == 4:
        data = np.frombuffer(raw, dtype=">i4").astype(np.float64)
    else:
        raise ValueError(f"unsupported AIFF sample width {sw}")
    if nch == 2:
        data = data.reshape(-1, 2).mean(axis=1)
    return data / float(2 ** (8 * sw - 1))


def trim_onset(x, sample_rate=SAMPLE_RATE, keep_sec=3.0):
    """Cut leading silence and keep the attack plus `keep_sec` of decay."""
    thresh = 0.01 * np.abs(x).max()
    first = int(np.argmax(np.abs(x) > thresh))
    start = max(0, first - int(0.005 * sample_rate))
    return x[start:start + int(keep_sec * sample_rate)]


def load_samples(sample_dir, midi_min=60, midi_max=83):
    """Load and cache the real piano samples for [midi_min, midi_max]."""
    samples = {}
    for midi in range(midi_min, midi_max + 1):
        path = os.path.join(sample_dir, f"Piano.ff.{midi_to_name(midi)}.aiff")
        if os.path.exists(path):
            samples[midi] = trim_onset(read_aiff(path))
    if not samples:
        raise FileNotFoundError(
            f"No piano samples found in {sample_dir}; download them first."
        )
    return samples


def _nearest_sampled_octave(midi, samples):
    """Shift `midi` by whole octaves until it lands on a loaded sample.

    `piano_samples` only covers two octaves (MIDI 60-83); datasets with a
    wider range (e.g. jazz solos) fall outside that. Octave-shifting keeps
    the pitch class (and therefore melodic contour/scale) intact while
    landing on an available sample.
    """
    if midi in samples:
        return midi
    for shift in range(12, 120, 12):
        if midi - shift in samples:
            return midi - shift
        if midi + shift in samples:
            return midi + shift
    raise KeyError(f"no sample within any octave of MIDI note {midi}")


def render_melody(midi_notes, out_path, sample_dir, note_sec=0.45, gap_sec=0.05):
    """Render a list of MIDI pitches to a WAV using real piano samples.

    Notes outside the sampled range are octave-shifted to the nearest
    available sample (see `_nearest_sampled_octave`) rather than failing.

    Returns the absolute path of the written file.
    """
    samples = load_samples(sample_dir)
    sounds = []
    for midi in midi_notes:
        midi = _nearest_sampled_octave(midi, samples)
        x = samples[midi]
        n = int(note_sec * SAMPLE_RATE)
        note = x[:n].copy() if len(x) >= n else np.pad(x, (0, n - len(x)))
        fade = min(int(0.05 * SAMPLE_RATE), n)
        note[-fade:] *= np.linspace(1.0, 0.0, fade)
        sounds.append(note)
        sounds.append(music.silence(duration=gap_sec))

    sonic = music.horizontal_stack(*sounds)
    music.write_wav_mono(sonic, filename=out_path, sample_rate=SAMPLE_RATE,
                         fades=(10, 20))
    return os.path.abspath(out_path)
