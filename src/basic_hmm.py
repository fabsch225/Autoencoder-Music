"""BasicHMM: a small categorical-observation Hidden Markov Model.

Thin wrapper around `hmmlearn.hmm.CategoricalHMM` that implements the
`SequenceModel` interface.
"""

import numpy as np
from hmmlearn import hmm

from sequence_model import SequenceModel


class BasicHMM(SequenceModel):
    """A very basic HMM with categorical emissions.

    Parameters
    ----------
    n_components : int
        Number of hidden (latent) states.
    n_iter : int
        Number of Baum-Welch (EM) iterations.
    tol : float
        Convergence tolerance for the EM loop.
    random_state : int
        Seed for reproducible initialization and sampling.
    emission_weights : array_like, optional
        Per-symbol positive weights (length = number of symbols) applied to
        the emission probabilities at sampling time. This lets the caller
        bias generation toward "tonally correct" notes (e.g. notes in the
        chosen key) without retraining.
    """

    def __init__(self, n_components=8, n_iter=50, tol=1e-3, random_state=42,
                 emission_weights=None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.emission_weights = (None if emission_weights is None
                                 else np.asarray(emission_weights, dtype=float))
        self._model = None

    @property
    def n_symbols(self):
        """Number of emission symbols (set after fit())."""
        return None if self._model is None else self._model.n_features

    def fit(self, sequences):
        """Train via Baum-Welch on the (concatenated) sequences."""
        lengths = [len(s) for s in sequences]
        X = np.concatenate(sequences).reshape(-1, 1)  # hmmlearn wants 2D
        self._model = hmm.CategoricalHMM(
            n_components=self.n_components,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        self._model.fit(X, lengths)
        return self

    def generate(self, length, seed=None):
        """Sample a new sequence of `length` symbols.

        If `emission_weights` was provided, the learned emission probabilities
        are re-weighted by them before sampling, so favored symbols (e.g.
        notes in the target key) are more likely to be emitted.
        """
        if self._model is None:
            raise RuntimeError("fit() must be called before generate()")

        rng = np.random.default_rng(self.random_state if seed is None else seed)
        startprob = self._model.startprob_
        transmat = self._model.transmat_
        emission = self._model.emissionprob_

        # Optionally re-weight emissions (tonal bias), renormalizing per state.
        if self.emission_weights is not None:
            emission = emission * self.emission_weights
            emission /= emission.sum(axis=1, keepdims=True)

        n_symbols = emission.shape[1]
        state = int(rng.choice(self.n_components, p=startprob))
        out = []
        for _ in range(length):
            out.append(int(rng.choice(n_symbols, p=emission[state])))
            state = int(rng.choice(self.n_components, p=transmat[state]))
        return np.asarray(out, dtype=int)
