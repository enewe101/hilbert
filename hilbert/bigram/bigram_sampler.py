import random
from abc import ABC, abstractmethod
import numpy as np
import time
import hilbert as h


def get_sampler(
    name, 
    bigram,
    window,
    min_count=None,
):
    """
    Create and return a new sampler instance by `name`, initializing it with
    `bigram`, `window`, and `min_count`.
    """
    samplers = {
        'flat': FlatSampler,
        'harmonic': HarmonicSampler,
        'dynamic': DynamicSampler
    }
    if name not in samplers:
        raise ValueError(
            "Unexpected sampler type {}.  Expected 'flat', 'harmonic', "
            "or 'dynamic'.".format(repr(name))
        )
    return samplers[name](bigram, window, min_count)



class BaseSampler(ABC):
    """
    Baseclass for cooccurrence samplers.  Override `sample()`.
    """
    def __init__(self, bigram, window, min_count=None):
        self.bigram = bigram
        self.window = window
        self.min_count = min_count

    @abstractmethod
    def sample(self, tokens):
        pass

    def filter_tokens(self, tokens):
        return [
            self.bigram.dictionary.get_id(t) for t in tokens 
            if t in self.bigram.dictionary 
            and (
                self.min_count is None or self.min_count is 1 or
                self.bigram.unigram.count(t) >= self.min_count
            )
        ]


# TODO: make samplerflat and samplerharmonic work the way that sampler dynamic
# works.


class FlatSampler(BaseSampler):
    """
    Sampler that extracts cooccurrence statistics from a corpus, where two
    words are considered to cooccur if they are separated by a distance of 
    ``window`` or less.  Cooccurrence statistics are recorded in
    ``bigram``, an instance of ``hilbert.bigram.BigramMutable``.  Words not
    in the ``bigram.dictionary``, and words occurring fewer than ``min_count``
    number of times will be removed before cooccurrence is evaluated.  
    All cooccurrences are given equal weight, regardless of how far appart the
    tokens are (but provided they are within ``window`` of one another).
    """
    def sample(self, tokens):
        tokens = self.filter_tokens(tokens)
        for i in range(len(tokens)):
            for j in range(i-self.window, i+self.window+1):
                if j == i or j < 0 or j >= len(tokens):
                    continue
                self.bigram.add(tokens[i], tokens[j], skip_unk=True)



class HarmonicSampler(BaseSampler):
    """
    Sampler that extracts cooccurrence statistics from a corpus, where two
    words are considered to cooccur if they are separated by a distance of 
    ``window`` or less.  Cooccurrence statistics are recorded in
    ``bigram``, an instance of ``hilbert.bigram.BigramMutable``.  Words not
    in the ``bigram.dictionary``, and words occurring fewer than ``min_count``
    number of times will be removed before cooccurrence is evaluated.  
    Cooccurrences are weighted based on the distance between tokens:
    ``weight = 1 / distance``.  Adjacent tokens have distance 1.
    """
    def sample(self, tokens):
        tokens = self.filter_tokens(tokens)
        for i in range(len(tokens)):
            for j in range(i-self.window, i+self.window+1):
                if j == i or j < 0 or j >= len(tokens):
                    continue
                self.bigram.add(
                    tokens[i], tokens[j], count=1.0/abs(i-j), skip_unk=True)



class DynamicSampler(BaseSampler):
    """
    Sampler that extracts cooccurrence statistics from a corpus, where two
    words are considered to cooccur if they are separated by a distance of 
    ``window`` or less.  Cooccurrence statistics are recorded in
    ``bigram``, an instance of ``hilbert.bigram.BigramMutable``.  Words not
    in the ``bigram.dictionary``, and words occurring fewer than ``min_count``
    number of times will be removed before cooccurrence is evaluated.  
    Cooccurrences are weighted based on the distance between tokens:
    ``weight = (distance+1)/window``.  Adjacent tokens have distance 1.
    """
    def sample(self, tokens):
        tokens = self.filter_tokens(tokens)
        # Cooccurrences are weighted based on distance.
        for offset in range(1, self.window+1):
            focal_ids = tokens[:-offset] + tokens[offset:]
            context_ids = tokens[offset:] + tokens[:-offset] 
            weight = (self.window - offset + 1) / self.window
            self.bigram.add_id(focal_ids, context_ids, weight)
        return




