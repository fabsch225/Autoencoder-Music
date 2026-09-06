"""VAEModel: a recurrent VAE over musical symbols (PyTorch).

Same variational-autoencoder mechanics as the MNIST VAE (encode to a
Gaussian, reparameterize, decode, optimize a reconstruction + KL ELBO), but
with an LSTM encoder/decoder instead of MLPs so it can handle variable-length
symbol sequences. Implements the `SequenceModel` interface: it is trained on
fixed-length windows the same way `RNNModel` is, but every window is first
squeezed through a latent bottleneck, so generation samples a latent code
from the prior N(0, I) and decodes it autoregressively (rather than just
running the language model forward from a start token).

Beyond `fit`/`generate`, this exposes `encode`/`decode` so notebooks can
inspect and interpolate in latent space, mirroring `inspect_latent_point`
in the MNIST VAE notebook.
"""

import numpy as np
import torch
import torch.nn as nn

from sequence_model import SequenceModel


class _Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, latent_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x, lengths):
        # x: (B, T) long, right-padded; lengths: (B,) true window lengths.
        e = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        h = h[-1]  # (B, hidden_size), final layer's hidden state
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, latent_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.z_to_h0 = nn.Linear(latent_dim, hidden_size)
        self.z_to_c0 = nn.Linear(latent_dim, hidden_size)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, z):
        h0 = torch.tanh(self.z_to_h0(z)).unsqueeze(0)  # (1, B, hidden)
        c0 = torch.tanh(self.z_to_c0(z)).unsqueeze(0)
        return h0, c0

    def forward(self, z, x, hidden=None):
        # x: (B, T) long tokens fed in (teacher-forced during training,
        # one step at a time during sampling). hidden is (h, c) or None to
        # initialize it from z.
        if hidden is None:
            hidden = self.init_hidden(z)
        e = self.embed(x)
        out, hidden = self.lstm(e, hidden)
        return self.fc(out), hidden


class VAEModel(SequenceModel):
    """A recurrent variational autoencoder over symbol sequences.

    Trains like `RNNModel` on overlapping start-token-prefixed windows, but
    each window is encoded to a Gaussian latent code, reparameterized, and
    reconstructed by an LSTM decoder conditioned on that code (its initial
    hidden state). The loss is the standard VAE ELBO: reconstruction
    cross-entropy plus a (linearly annealed) KL-to-prior term.

    Parameters
    ----------
    embed_dim : int
        Embedding dimensionality (shared by encoder and decoder).
    hidden_size : int
        LSTM hidden-state size.
    latent_dim : int
        Dimensionality of the latent code z.
    context_len : int
        Window length used for training (and the block size decoded at a
        time when generating longer sequences).
    epochs : int
        Full passes over the training data.
    batch_size : int
        Number of windows per training step.
    lr : float
        Adam learning rate.
    beta : float
        Final KL weight (beta-VAE style). Annealed linearly from 0 over the
        first half of training to curb posterior collapse.
    temperature : float
        Sampling temperature (1.0 = model distribution, <1 sharper).
    random_state : int
        Seed for reproducible training and sampling.
    """

    def __init__(self, embed_dim=32, hidden_size=64, latent_dim=8,
                 context_len=32, epochs=10, batch_size=64, lr=1e-3,
                 beta=1.0, temperature=1.0, random_state=42):
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.context_len = context_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta
        self.temperature = temperature
        self.random_state = random_state

        self.n_symbols = None      # number of data symbols (set in fit)
        self._start = None         # start-token index (= n_symbols)
        self._encoder = None
        self._decoder = None
        self.device = torch.device("mps") if torch.backends.mps.is_available() \
            else torch.device("cpu")

    # -- helpers ---------------------------------------------------------
    def _build(self, vocab_size):
        encoder = _Encoder(
            vocab_size, self.embed_dim, self.hidden_size, self.latent_dim
        ).to(self.device)
        decoder = _Decoder(
            vocab_size, self.embed_dim, self.hidden_size, self.latent_dim
        ).to(self.device)
        return encoder, decoder

    @staticmethod
    def _reparameterize(mu, logvar, generator=None):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.shape, generator=generator) if generator is not None \
            else torch.randn_like(std)
        return mu + std * eps

    def _collate(self, batch):
        """Pad a batch of start-token-prefixed windows.

        Returns (enc_in, dec_in, dec_target, lengths):
        - enc_in / dec_in: the window without its final token (what the
          encoder sees, and what's teacher-forced into the decoder).
        - dec_target: the window shifted by one (what the decoder predicts).
        - lengths: true (unpadded) length of each enc_in/dec_in sequence,
          for pack_padded_sequence.
        """
        B = len(batch)
        L = max(len(w) for w in batch) - 1
        x = torch.zeros(B, L, dtype=torch.long)
        y = torch.full((B, L), -100, dtype=torch.long)
        lengths = torch.zeros(B, dtype=torch.long)
        for i, w in enumerate(batch):
            n = len(w) - 1
            x[i, :n] = torch.from_numpy(w[:-1])
            y[i, :n] = torch.from_numpy(w[1:])
            lengths[i] = n
        return x, y, lengths

    # -- interface -------------------------------------------------------
    def fit(self, sequences):
        self.n_symbols = max(max(s) for s in sequences) + 1
        self._start = self.n_symbols  # one extra embedding for the start token

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._encoder, self._decoder = self._build(self.n_symbols + 1)
        params = list(self._encoder.parameters()) + list(self._decoder.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

        # Build overlapping windows (each prefixed with the start token),
        # same scheme as RNNModel.
        windows = []
        for s in sequences:
            s = np.asarray(s, dtype=np.int64)
            if len(s) < 2:
                continue
            tokens = np.concatenate(([self._start], s))
            for start in range(0, len(tokens) - 1, self.context_len):
                windows.append(tokens[start:start + self.context_len + 1])

        n = len(windows)
        anneal_epochs = max(1, self.epochs // 2)
        self._encoder.train()
        self._decoder.train()
        for epoch in range(self.epochs):
            kl_weight = self.beta * min(1.0, (epoch + 1) / anneal_epochs)
            perm = np.random.permutation(n)
            total_recon, total_kl, count = 0.0, 0.0, 0
            for i in range(0, n, self.batch_size):
                batch = [windows[j] for j in perm[i:i + self.batch_size]]
                x, y, lengths = self._collate(batch)
                x, y = x.to(self.device), y.to(self.device)

                mu, logvar = self._encoder(x, lengths)
                z = self._reparameterize(mu, logvar)
                logits, _ = self._decoder(z, x)

                recon = loss_fn(logits.transpose(1, 2), y) / len(batch)
                kl = -0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp()
                ) / len(batch)
                loss = recon + kl_weight * kl

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_recon += recon.item() * len(batch)
                total_kl += kl.item() * len(batch)
                count += len(batch)
            if (epoch + 1) % max(1, self.epochs // 5) == 0 or epoch == 0:
                print(f"  VAE epoch {epoch + 1}/{self.epochs}  "
                      f"recon {total_recon / count:.4f}  "
                      f"kl {total_kl / count:.4f}  "
                      f"beta {kl_weight:.3f}")
        return self

    def generate(self, length, seed=None):
        """Sample z ~ N(0, I) and autoregressively decode `length` symbols."""
        if self._decoder is None:
            raise RuntimeError("fit() must be called before generate()")

        gen = torch.Generator().manual_seed(
            self.random_state if seed is None else seed
        )
        z = torch.randn(1, self.latent_dim, generator=gen)
        return self.decode(z, length, seed=seed)

    def encode(self, sequence):
        """Encode a single observed symbol sequence to (mu, logvar).

        Useful for inspecting or interpolating in latent space; not part of
        generation itself.
        """
        if self._encoder is None:
            raise RuntimeError("fit() must be called before encode()")
        s = np.asarray(sequence, dtype=np.int64)
        tokens = np.concatenate(([self._start], s))
        x = torch.from_numpy(tokens[:-1]).long().unsqueeze(0).to(self.device)
        lengths = torch.tensor([x.shape[1]])
        self._encoder.eval()
        with torch.no_grad():
            mu, logvar = self._encoder(x, lengths)
        return mu, logvar

    def decode(self, z, length, seed=None):
        """Autoregressively decode `length` symbols from a latent code z.

        Parameters
        ----------
        z : tensor of shape (1, latent_dim)
        length : int
        seed : int, optional
        """
        if self._decoder is None:
            raise RuntimeError("fit() must be called before decode()")

        z = z.to(self.device)
        gen = torch.Generator().manual_seed(
            self.random_state if seed is None else seed
        )
        self._decoder.eval()
        generated = []
        token = self._start
        hidden = None
        with torch.no_grad():
            for _ in range(length):
                x = torch.tensor([[token]], dtype=torch.long, device=self.device)
                logits, hidden = self._decoder(z, x, hidden)
                logits = logits[:, -1, :].cpu() / self.temperature
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                token = torch.multinomial(probs, 1, generator=gen).item()
                generated.append(token)
        return np.asarray(generated, dtype=int)
