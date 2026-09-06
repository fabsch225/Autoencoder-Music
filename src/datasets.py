"""Loaders for melody datasets used by music.ipynb and vae_music.py.

Every loader returns a list of 1D numpy arrays of raw MIDI pitch numbers (one
array per song/solo) -- label-encoding into contiguous symbol indices is a
separate, shared step (`encode_melodies`), so callers can train any
`SequenceModel` identically regardless of which dataset supplied the notes.

All default paths point into `data/` (a sibling of this file's `src/`
directory), regardless of the caller's current working directory.

Datasets
--------
jsb (Bach chorales)
    `data/jsb-chorales-16th.pkl` -- a pickled {"train"/"valid"/"test": [...]}
    dict of 16th-note piano rolls (Boulanger-Lewandowski et al. 2012). Each
    song is a list of chords (one per time step); we take the soprano (top)
    voice.

nottingham (folk tunes)
    `data/nottingham_midi/{train,valid,test}/*.mid` -- 1037 English/Irish jigs,
    reels and hornpipes (Boulanger-Lewandowski's Nottingham release: raw MIDI,
    not the same pre-pickled piano-roll format as JSB). Each file has a
    monophonic melody track plus a block-chord accompaniment track at a lower
    pitch; we quantize every track onto a 16th-note grid, merge them into a
    piano roll the same shape as the JSB chorales', and take the top voice --
    which recovers the melody even without knowing which MIDI track it's on,
    since the accompaniment always sits lower. Needs `pip install mido`.

weimar (jazz solos)
    `data/wjazzd.db` -- the Weimar Jazz Database: an SQLite database of 456
    monophonic solo transcriptions (Charlie Parker, Miles Davis, John
    Coltrane, etc.) from the Jazzomat Research Project
    (https://jazzomat.hfm-weimar.de, ODbL license). Its `melody` table has
    one row per transcribed note event (columns include `melid`, `onset`,
    `pitch`); we read one sequence per solo (`melid`), ordered by onset. If
    the schema has since changed, `load_weimar_jazz` raises with the
    tables/columns it actually found so the query above can be adjusted.
"""

import glob
import os
import pickle
import sqlite3

import numpy as np
from sklearn.preprocessing import LabelEncoder

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")


def _soprano_from_pianoroll(songs):
    """Top-voice (max-pitch) melody from a list of piano-roll chord lists."""
    melodies = []
    for song in songs:
        mel = [max(chord) for chord in song if len(chord) > 0]
        if mel:
            melodies.append(np.asarray(mel, dtype=int))
    return melodies


def load_jsb_chorales(path, split="train"):
    """JSB Chorales: soprano melody of each 16th-note piano-roll chorale."""
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return _soprano_from_pianoroll(data[split])


def _midi_to_pianoroll(path, steps_per_beat=4):
    """Quantize every track of a MIDI file onto a 16th-note grid.

    Returns a list of chords (one tuple of active MIDI pitches per grid
    step), merging all tracks -- the same shape as one JSB-chorales entry,
    so `_soprano_from_pianoroll` works on either. Melody and accompaniment
    tracks aren't distinguished; taking the max pitch per step recovers the
    melody anyway, since accompaniment chords sit in a lower register.
    """
    import mido

    mid = mido.MidiFile(path)
    ticks_per_step = mid.ticks_per_beat / steps_per_beat

    events = []  # (abs_tick, pitch, is_note_on)
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                events.append((abs_tick, msg.note, True))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                events.append((abs_tick, msg.note, False))
    if not events:
        return []

    events.sort(key=lambda e: (e[0], e[2]))  # note-offs before note-ons per tick
    n_steps = int(events[-1][0] / ticks_per_step) + 1

    active = set()
    pianoroll = []
    idx = 0
    for step in range(n_steps):
        step_tick = step * ticks_per_step
        while idx < len(events) and events[idx][0] <= step_tick:
            _, pitch, is_on = events[idx]
            active.add(pitch) if is_on else active.discard(pitch)
            idx += 1
        pianoroll.append(tuple(active))
    return pianoroll


def load_nottingham(path, split="train"):
    """Nottingham folk tunes: soprano melody of each MIDI file's piano roll."""
    midi_files = sorted(glob.glob(os.path.join(path, split, "*.mid")))
    if not midi_files:
        raise FileNotFoundError(f"no .mid files found under {path}/{split}")
    songs = [_midi_to_pianoroll(f) for f in midi_files]
    return _soprano_from_pianoroll(songs)


def load_weimar_jazz(path, min_len=16):
    """Weimar Jazz Database: one melody per transcribed solo.

    Parameters
    ----------
    path : str
        Path to the `wjazzd.db` SQLite file.
    min_len : int
        Discard solos shorter than this many notes.
    """
    conn = sqlite3.connect(path)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(melody)")}
        required = {"melid", "pitch", "onset"}
        if not required.issubset(cols):
            tables = [row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
            raise ValueError(
                f"'melody' table in {path} is missing expected columns "
                f"{sorted(required - cols)} (found: {sorted(cols)}). "
                f"Tables present: {tables}. The WJD schema may have changed "
                "-- update the query in load_weimar_jazz() to match it."
            )
        rows = conn.execute(
            "SELECT melid, pitch FROM melody ORDER BY melid, onset"
        ).fetchall()
    finally:
        conn.close()

    solos = {}
    for melid, pitch in rows:
        solos.setdefault(melid, []).append(int(pitch))

    return [np.asarray(notes, dtype=int)
            for notes in solos.values() if len(notes) >= min_len]


DATASETS = {
    "jsb": dict(
        loader=load_jsb_chorales,
        default_path=os.path.join(DATA_DIR, "jsb-chorales-16th.pkl"),
        name="JSB Chorales (Bach)",
    ),
    "nottingham": dict(
        loader=load_nottingham,
        default_path=os.path.join(DATA_DIR, "nottingham_midi"),
        name="Nottingham folk tunes",
    ),
    "weimar": dict(
        loader=load_weimar_jazz,
        default_path=os.path.join(DATA_DIR, "wjazzd.db"),
        name="Weimar Jazz Database (jazz solos)",
    ),
}


def load_melodies(dataset, path=None):
    """Load raw-MIDI melodies for one of the datasets registered above."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choices: {sorted(DATASETS)}")
    spec = DATASETS[dataset]
    return spec["loader"](path or spec["default_path"])


def encode_melodies(melodies):
    """Label-encode raw MIDI pitches into contiguous symbol indices.

    Returns (sequences, encoder); `encoder.inverse_transform` maps generated
    symbols back to MIDI pitches.
    """
    all_notes = np.concatenate(melodies)
    encoder = LabelEncoder()
    encoder.fit(all_notes)
    sequences = [encoder.transform(m) for m in melodies]
    return sequences, encoder
