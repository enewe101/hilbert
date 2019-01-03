import random
import numpy as np
import time
import hilbert as h

# TODOS:
#   Test the use of skip_unk in all of the samplers
#   For SamplerW2V, switch to using the dictionary to filter low frequency words

def get_sampler(
    name, 
    bigram,
    window,
    min_count=1,
    thresh=1,
):

    if name == 'flat':
        return SamplerFlat(bigram, window)
    elif name == 'harmonic':
        return SamplerHarmonic(bigram, window)
    elif name == 'w2v':
        return SamplerW2V(bigram, window, thresh)
    elif name == 'dynamic':
        return SamplerDynamic(bigram, window, min_count)
    else:
        raise ValueError(
            "Unexpected sampler type {}.  Expected 'flat', 'harmonic', "
            "'dynamic', or 'w2v'.".format(repr(name))
        )


class SamplerFlat:

    def __init__(self, bigram, window):
        self.bigram = bigram
        self.window = window

    def sample(self, tokens):
        for i in range(len(tokens)):
            for j in range(i-self.window, i+self.window+1):
                if j == i or j < 0 or j >= len(tokens):
                    continue
                self.bigram.add(tokens[i], tokens[j], skip_unk=True)


class SamplerHarmonic:

    def __init__(self, bigram, window):
        self.bigram = bigram
        self.window = window

    def sample(self, tokens):
        for i in range(len(tokens)):
            for j in range(i-self.window, i+self.window+1):
                if j == i or j < 0 or j >= len(tokens):
                    continue
                self.bigram.add(
                    tokens[i], tokens[j], count=1.0/abs(i-j), skip_unk=True)




class SamplerDynamic:

    def __init__(self, bigram, window, min_count=None):
        self.bigram = bigram
        self.window = window
        self.min_count = min_count


    def sample(self, tokens):

        tokens = [
            self.bigram.dictionary.get_id(t) for t in tokens 
            if t in self.bigram.dictionary 
            and (
                self.min_count is None 
                or self.bigram.unigram.count(t) >= self.min_count
            )
        ]

        # Cooccurrences are weighted based on distance.
        for offset in range(1, self.window+1):
            focal_ids = tokens[:-offset] + tokens[offset:]
            context_ids = tokens[offset:] + tokens[:-offset] 
            weight = (self.window - offset + 1) / self.window
            self.bigram.add_id(focal_ids, context_ids, weight)

        return




# TODO: min_count should no longer be handled here.  Instead, the unigram
# contained within the passed bigram instance should have already been pruned
# down to desired vocabulary.  Filtering out rare words should be done based
# on looking at whether the words are in the unigram's dictionary.
class SamplerW2V:

    def __init__(self, bigram, window, thresh, min_count=0):
        self.bigram = bigram
        self.window = window
        self.thresh = thresh
        self.min_count = min_count
        self.max_dist = max(2*window, window+5)


    def sample(self, tokens):

        # If needed, filter out rare words
        # TODO: change this to check for presence of tokens in dictionary
        if self.min_count > 1:
            tokens = [
                t for t in tokens 
                if self.bigram.unigram.count(t) >= self.min_count
            ]

        # If the threshold is high (1), the calculation is very simple:
        # cooccurrences are weighted based on distance.
        if self.thresh == 1:
            for i in range(len(tokens)):
                for j in range(i-self.window, i+self.window+1):
                    if j < 0 or j >= len(tokens) or j == i:
                        continue
                    distance = abs(i-j)
                    weight = (self.window - distance + 1) / self.window
                    self.bigram.add(tokens[i], tokens[j], weight)
            return


        # Otherwise, if threshold is not (1), then we need to consider the
        # possible relative locations of tokens due to dropping common words
        # to determine cooccurrence weights.
        drop_probs = self.drop_prob(tokens)
        expected_weights = self.get_count_prob(drop_probs)
        expected_weight_right, expected_weight_left = expected_weights

        for i in range(len(tokens)):
            for c in range(self.max_dist):
                j = i - c
                if j < 0: continue
                self.bigram.add(
                    tokens[i], tokens[j], expected_weight_right[i,c])

        for i in range(len(tokens)):
            for c in range(self.max_dist):
                j = i + c
                if j >= len(tokens): continue
                self.bigram.add(
                    tokens[i], tokens[j], expected_weight_left[i,c])


    def get_count_prob(self, drop_probs):
        weight = np.array(
            [0] + [(self.window-d)/self.window for d in range(self.window)] 
            + [0] * (self.max_dist - self.window - 1)
        ).reshape(1,-1,1)
        count_prob_right = self.get_count_prob_right(drop_probs)
        expected_weight_right = (count_prob_right * weight).sum(axis=1)
        count_prob_left = self.get_count_prob_left(drop_probs)
        expected_weight_left = (count_prob_left * weight).sum(axis=1)
        return expected_weight_right, expected_weight_left


    def get_count_prob_left(self, drop_prob):
        reverse_drop_prob = drop_prob[::-1]
        count_prob = self.get_count_prob_right(reverse_drop_prob)
        return count_prob[::-1]


    def get_count_prob_right(self, drop_prob):

        l = len(drop_prob)

        # Axis 1 indexes context words, axis 2 indexes relative distances, 
        # axis 3 indexes target words
        pos_prob = np.zeros((l,self.max_dist,self.max_dist))
        count_prob = np.zeros((l,self.max_dist,self.max_dist))
        pos_prob[range(l),0,0] = (1-drop_prob)

        for d in range(1,l):
            pos_prob[d,:,1:] += pos_prob[d-1,:,:-1] * drop_prob[d]
            pos_prob[d, 1:, 1:] += pos_prob[d-1, :-1, :-1] * (1-drop_prob[d])
            count_prob[d, 1:, 1:] += pos_prob[d-1, :-1, :-1] * (1-drop_prob[d])

        return count_prob


    def drop_prob(self, tokens):
        drop_probabilities = np.zeros(len(tokens))
        for i, token in enumerate(tokens):
            freq = self.bigram.unigram.freq(token)
            if freq == 0:
                prob = 0
            else:
                prob = (freq - self.thresh) / freq - (self.thresh / freq)**.5
            drop_probabilities[i] = clip(0,1,prob)
        return drop_probabilities


def clip(minimum, maximum, val):
    val = min(val, maximum)
    return max(val, minimum)


