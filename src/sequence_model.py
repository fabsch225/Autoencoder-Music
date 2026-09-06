"""Abstract interface for sequence-generation models.

Both the HMM and the RNN implement this interface so they can be trained
and sampled interchangeably in the driver script.
"""

from abc import ABC, abstractmethod


class SequenceModel(ABC):
    """A generative model over integer symbol sequences.

    The model operates on integer "symbol" indices (e.g. MIDI pitches that
    have been label-encoded). Encoding/decoding back to musical notes is the
    caller's responsibility, so this interface stays model-agnostic.
    """

    @abstractmethod
    def fit(self, sequences):
        """Train the model on a collection of symbol sequences.

        Parameters
        ----------
        sequences : list of 1D numpy integer arrays
            Each array is a sequence of symbol indices.

        Returns
        -------
        self
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, length, seed=None):
        """Sample a new sequence of symbol indices.

        Parameters
        ----------
        length : int
            Number of symbols to generate.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        numpy integer array of shape (length,)
        """
        raise NotImplementedError
