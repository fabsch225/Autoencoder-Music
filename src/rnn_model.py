"""RNNModel: a basic LSTM language model over musical symbols (PyTorch).

Learns to predict the next symbol in a sequence and is then sampled
autoregressively, implementing the `SequenceModel` interface.
"""

import numpy as np
import torch
import torch.nn as nn

from sequence_model import SequenceModel


class _NextTokenLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_size=64, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x: (B, T) long -> (B, T, V) logits
        e = self.embed(x)
        out, hidden = self.lstm(e, hidden)
        return self.fc(out), hidden


class RNNModel(SequenceModel):
    """A basic next-symbol LSTM language model.

    Parameters
    ----------
    embed_dim : int
        Embedding dimensionality.
    hidden_size : int
        LSTM hidden-state size.
    num_layers : int
        Number of stacked LSTM layers.
    context_len : int
        Window length used for backpropagation-through-time during training.
    epochs : int
        Full passes over the training data.
    batch_size : int
        Number of windows per training step.
    lr : float
        Adam learning rate.
    temperature : float
        Sampling temperature (1.0 = model distribution, <1 sharper).
    random_state : int
        Seed for reproducible training and sampling.
    """

    def __init__(self, embed_dim=32, hidden_size=64, num_layers=1,
                 context_len=32, epochs=10, batch_size=64, lr=1e-3,
                 temperature=1.0, random_state=42):
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.context_len = context_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.temperature = temperature
        self.random_state = random_state

        self.n_symbols = None      # number of data symbols (set in fit)
        self._start = None         # start-token index (= n_symbols)
        self._model = None
        self.device = torch.device("mps") if torch.backends.mps.is_available() \
            else torch.device("cpu")

    # -- helpers ---------------------------------------------------------
    def _build(self, vocab_size):
        return _NextTokenLSTM(
            vocab_size, self.embed_dim, self.hidden_size, self.num_layers
        ).to(self.device)

    def _collate(self, batch):
        """Pad a batch of token windows into (x, y) tensors.

        Each window already begins with the start token. We predict each
        token from the one before it; padded targets are -100 (ignored).
        """
        B = len(batch)
        L = max(len(w) for w in batch) - 1
        x = torch.zeros(B, L, dtype=torch.long)
        y = torch.full((B, L), -100, dtype=torch.long)
        for i, w in enumerate(batch):
            x[i, :len(w) - 1] = torch.from_numpy(w[:-1])
            y[i, :len(w) - 1] = torch.from_numpy(w[1:])
        return x, y

    # -- interface -------------------------------------------------------
    def fit(self, sequences):
        self.n_symbols = max(max(s) for s in sequences) + 1
        self._start = self.n_symbols  # one extra embedding for the start token

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._model = self._build(self.n_symbols + 1)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # Build overlapping windows (each prefixed with the start token).
        windows = []
        for s in sequences:
            s = np.asarray(s, dtype=np.int64)
            if len(s) < 2:
                continue
            tokens = np.concatenate(([self._start], s))
            for start in range(0, len(tokens) - 1, self.context_len):
                windows.append(tokens[start:start + self.context_len + 1])

        n = len(windows)
        self._model.train()
        for epoch in range(self.epochs):
            perm = np.random.permutation(n)
            total, count = 0.0, 0
            for i in range(0, n, self.batch_size):
                batch = [windows[j] for j in perm[i:i + self.batch_size]]
                x, y = self._collate(batch)
                x, y = x.to(self.device), y.to(self.device)
                logits, _ = self._model(x)              # (B, T, V)
                loss = loss_fn(logits.transpose(1, 2), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
                count += len(batch)
            if (epoch + 1) % max(1, self.epochs // 5) == 0 or epoch == 0:
                print(f"  RNN epoch {epoch + 1}/{self.epochs}  loss "
                      f"{total / count:.4f}")
        return self

    def generate(self, length, seed=None):
        """Autoregressively sample `length` symbols from the trained RNN."""
        if self._model is None:
            raise RuntimeError("fit() must be called before generate()")

        gen = torch.Generator().manual_seed(
            self.random_state if seed is None else seed
        )
        self._model.eval()
        generated = []
        token = self._start
        hidden = None
        with torch.no_grad():
            for _ in range(length):
                x = torch.tensor([[token]], dtype=torch.long, device=self.device)
                logits, hidden = self._model(x, hidden)
                logits = logits[:, -1, :].cpu() / self.temperature
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                token = torch.multinomial(probs, 1, generator=gen).item()
                generated.append(token)
        return np.asarray(generated, dtype=int)
