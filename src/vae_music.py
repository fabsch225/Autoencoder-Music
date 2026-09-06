"""Standalone VAE-only melody generator, with a toggleable dataset.

Trains a `VAEModel` (see vae_model.py) on one of the melody datasets
registered in `datasets.py` and renders a generated melody to WAV. Kept
separate from music.ipynb so the dataset (and hyperparameters) can be
switched from the command line instead of editing notebook cells.

Usage
-----
    python vae_music.py --dataset nottingham
    python vae_music.py --dataset jsb --notes 256 --epochs 50
    python vae_music.py --dataset weimar --path ../data/wjazzd.db --temperature 0.8
"""

import argparse
import os

import datasets
import render
from vae_model import VAEModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "data", "piano_samples")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(datasets.DATASETS),
                         default="nottingham",
                         help="which melody dataset to train on (default: nottingham)")
    parser.add_argument("--path", default=None,
                         help="override the dataset's default file path")
    parser.add_argument("--notes", type=int, default=512,
                         help="number of notes to generate")
    parser.add_argument("--out", default=None,
                         help="output WAV path (default: <dataset>_vae_melody.wav)")
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    melodies = datasets.load_melodies(args.dataset, path=args.path)
    sequences, encoder = datasets.encode_melodies(melodies)
    print(f"{datasets.DATASETS[args.dataset]['name']}: {len(melodies)} melodies, "
          f"{sum(len(m) for m in melodies)} notes, {len(encoder.classes_)} symbols")

    model = VAEModel(embed_dim=args.embed_dim, hidden_size=args.hidden_size,
                      latent_dim=args.latent_dim, context_len=args.context_len,
                      epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      beta=args.beta, temperature=args.temperature,
                      random_state=args.seed)
    model.fit(sequences)
    symbols = model.generate(args.notes, seed=args.seed)
    midi = encoder.inverse_transform(symbols).astype(int)

    out_path = args.out or os.path.join(RESULTS_DIR, f"{args.dataset}_vae_melody.wav")
    written = render.render_melody(midi, out_path, SAMPLE_DIR)
    print(f"Wrote {written}")


if __name__ == "__main__":
    main()
