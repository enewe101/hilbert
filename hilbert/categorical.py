import torch
import pdb


class Categorical:


    def __init__(self, probs, dtype=torch.float32, device=None):
        self.dtype = dtype
        self.device = device or probs.device
        self.probs = torch.tensor(probs, dtype=dtype, device=self.device)
        if len(self.probs.shape) != 1:
            raise ValueError("``probs`` should be a 1D-tensor.")
        self.probs /= self.probs.sum()
        self.alias_probs = None
        self.alias_outcomes = None
        self.setup()


    def __len__(self):
        return self.probs.shape[0]


    def setup(self):
        self.alias_probs = self.probs * len(self)
        self.alias_outcomes = torch.zeros(
            (len(self), 2), dtype=torch.long, device=self.device)
        self.alias_outcomes[:,0] = torch.arange(len(self), device=self.device)
        small_idxs, large_idxs = self.split_idxs(
            self.alias_probs, torch.arange(len(self), device=self.device))
        while small_idxs.shape[0] > 0 and large_idxs.shape[0] > 0:

            overlap_size = min(small_idxs.shape[0], large_idxs.shape[0])
            # Available large_idxs get allocated to the secondary 
            # outcomes for aliases holding small outcomes
            covered_small = small_idxs[:overlap_size]
            used_large = large_idxs[:overlap_size]
            self.alias_outcomes[covered_small, 1] = used_large

            remaining_small = small_idxs[overlap_size:]
            self.alias_probs[used_large] -= (1-self.alias_probs[covered_small])

            new_small, large_idxs = self.split_idxs(
                self.alias_probs[large_idxs], large_idxs)

            small_idxs = torch.cat((remaining_small, new_small))



    def split_idxs(self, vector, idxs=None):
        is_large = vector > 1
        num_large = is_large.sum()
        sorted_positions = torch.sort(is_large)[1]
        sorted_idxs = idxs[sorted_positions]
        large_idxs = sorted_idxs[-num_large:]
        small_idxs = sorted_idxs[:-num_large]

        return small_idxs, large_idxs






    def slow_setup(self):
        """
        For reference, this is an approach to assembling the alias probabilities
        and alias outcome map without taking advantage of easy parallelization
        using tensor-based operations.
        """
        self.alias_probs = torch.zeros(
            len(self), dtype=self.dtype, device=self.device)
        self.alias_outcomes = torch.zeros(
            (len(self), 2), dtype=torch.long, device=self.device)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger  = []
        for outcome in range(len(self)):
            self.alias_probs[outcome] = len(self) * self.probs[outcome]
            self.alias_outcomes[outcome,0] = outcome
            if self.alias_probs[outcome] < 1.0:
                smaller.append(outcome)
            else:
                larger.append(outcome)
        while len(smaller) > 0 and len(larger) > 0:
            small_outcome = smaller.pop()
            large_outcome = larger.pop()
            self.alias_outcomes[small_outcome, 1] = large_outcome
            self.alias_probs[large_outcome] = (
                self.alias_probs[large_outcome] -
                (1.0 - self.alias_probs[small_outcome])
            )
            if self.alias_probs[large_outcome] < 1.0:
                smaller.append(large_outcome)
            else:
                larger.append(large_outcome)


    def sample(self, shape=(1,), dtype='int64'):
        primary_outcomes = torch.randint(0, len(self), shape, dtype=torch.long)
        secondary_outcomes = (
            torch.rand(shape, device=self.device) > 
            self.alias_probs[primary_outcomes] 
        ).long()
        return self.alias_outcomes[primary_outcomes, secondary_outcomes]




