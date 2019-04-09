import random
from abc import ABC, abstractmethod
import numpy as np
import time
import hilbert as h
import os

WEIGHTS_EXAMPLE_PATH = os.path.join(h.CONSTANTS.TEST_DIR,'example-weights.txt')

def get_extractor(
    extractor_str, 
    cooccurrence,
    window=None,
    weights=None,
    min_count=None,
):
    """
    Create and return a new extractor instance by using one of the preset
    weight kernels (`flat`, `harmonic`, or `dynamic`) or custom weights.
    """

    # Ensure a custom extractor gets weights and non-custom gets a window.
    validate_cooccurrence_extractor_args(extractor_str, window, weights)

    # Prepare a weighting scheme based on the `extractor_str`.
    if extractor_str == 'dynamic':
        weights_branch = [ (window - i) / window for i in range(window)]
        weights = (weights_branch, weights_branch)
    elif extractor_str == 'harmonic':
        weights_branch = [1/(i+1) for i in range(window)]
        weights = (weights_branch, weights_branch)
    elif extractor_str == 'flat':
        weights_branch = [1 for i in range(window)]
        weights = (weights_branch, weights_branch)
    elif extractor_str == 'custom':
        # Caller is responsible for custom weights
        pass
    else:
        raise ValueError(
            "Unexpected extractor type {}.  Expected 'flat', 'harmonic', "
            "or 'dynamic'.".format(repr(extractor_str))
        )

    return CooccurrenceExtractor(
        cooccurrence=cooccurrence, 
        weights=weights, min_count=min_count
    )


def read_weights_file(path):
    # Read the weights file.  Kill extra whitespace and any comments.
    with open(path) as weights_file:
        weight_strs = weights_file.read().strip().split('\n')

    # Screen out comments and blank lines
    weight_strs = [
        ws for ws in weight_strs 
        if not ws.startswith('#') and not ws.strip() == ''
    ]

    # Validate the weights file.
    if len(weight_strs) != 2:
        raise ValueError(
            "Weights file bad format: the weights file should have two "
            "lines: the first line should list weights to be applied to the "
            "right of focal words, the second line should list weights to be "
            "applied to the left of the focal words.  Note that left weights"
            "will be applied as if it is mirrorred, therefore, the first "
            "float on the second line is the weight applied adjacent "
            "and to the left of the focal word.  See {} for an example".format(
                WEIGHTS_EXAMPLE_PATH
        ))

    right_weights = [float(s) for s in weight_strs[0].split()]
    left_weights = [float(s) for s in weight_strs[1].split()]

    return right_weights, left_weights



def validate_cooccurrence_extractor_args(extractor_str, window, weights):

    if extractor_str == 'custom':
        if weights is None:
            raise ValueError(
                "Must provide an array of weights for custom cooccurrence "
                "extractors."
            )
        if window is not None:
            raise ValueError(
                "Window should not be provided for custom cooccurrence "
                "extractors."
            )
    else:
        if weights is not None:
            raise ValueError(
                "weights should only be provided for custom cooccurrence "
                "extractors, not `{}`.".format(extractor_str)
            )



class CooccurrenceExtractor():
    """
    Extracts weighted cooccurrence statistics based on the weights provided.
    """
    def __init__(self, cooccurrence, weights, min_count=None):
        self.cooccurrence = cooccurrence
        self.right_weights, self.left_weights = weights
        self.min_count = min_count


    def filter_tokens(self, tokens):
        return [
            self.cooccurrence.dictionary.get_id(t) for t in tokens 
            if t in self.cooccurrence.dictionary 
            and (
                self.min_count is None or self.min_count is 1 or
                self.cooccurrence.unigram.count(t) >= self.min_count
            )
        ]

    def extract(self, tokens):
        tokens = self.filter_tokens(tokens)
        # Cooccurrences are weighted based on distance.
        for i, weight in enumerate(self.right_weights):
            offset = i + 1
            focal_ids = tokens[:-offset] 
            context_ids = tokens[offset:] 
            self.cooccurrence.add_id(focal_ids, context_ids, weight)

        for i, weight in enumerate(self.left_weights):
            offset = i + 1
            focal_ids = tokens[offset:] 
            context_ids = tokens[:-offset] 
            self.cooccurrence.add_id(focal_ids, context_ids, weight)
        return





