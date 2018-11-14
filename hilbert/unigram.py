import os
from copy import deepcopy
try:
    import numpy as np
except ImportError:
    np = None
import hilbert as h


#TODO: ensure the dtype is float32 not float64
class Unigram(object):
    """Represents unigram statistics."""

    def __init__(
        self,
        dictionary=None,
        Nx=None,
        device=None,
        verbose=True
    ):
        '''
        ``dictionary`` -- A hilbert.dictionary.Dictionary instance mapping token
            strings from/to integer IDs; 
        ``Nx`` -- A 1D array-like instance (e.g. numpy.ndarray,
            scipy.sparse.csr_matrix, or list), in which the ith element
            contains the number of cooccurrences for token with ID i.

        Unigram Keeps track of token occurrence counts, and saves/loads from
        disk.  Provide no arguments to create an empty instance, useful for 
        accumulating counts while reading through a corpus.
        '''

        self.validate_args(dictionary, Nx)
        self.dictionary = dictionary or h.dictionary.Dictionary()

        if Nx is None:
            self.Nx = list()
            self.N = 0
        else:
            self.Nx = list(Nx)
            self.N = sum(self.Nx)

        self.device = device
        self.verbose = verbose

        self.sorted = self.check_sorted()


    def check_sorted(self):
        """
        Check whether the dictionary tokens are sorted in decreasing order of 
        frequency.
        """
        last_count = np.inf
        for count in self.Nx:
            if count > last_count:
                return False
            last_count = count
        return True


    def apply_smoothing(self, alpha):
        if alpha == 1 or alpha is None:
            if self.verbose: print('unigram-smoothing:\t1')
            return
        self.Nx = [count**alpha for count in self.Nx]
        self.N = sum(self.Nx)
        if self.verbose: print('unigram-smoothing:\t{}'.format(alpha))


    #   CHECK
    def __getitem__(self, shard):
        return self.load_shard(shard)

        
    # TODO: test sharding
    #   CHECK
    def load_shard(self, shard=None, device=None):

        if shard is None:
            shard = h.shards.whole

        device = device or self.device or h.CONSTANTS.MATRIX_DEVICE

        loaded_Nx = h.utils.load_shard(
            self.Nx, shard[0], device=device).view(-1,1)
        loaded_Nxt = h.utils.load_shard(
            self.Nx, shard[1], device=device).view(1,-1)
        loaded_N = h.utils.load_shard(self.N, device=device)

        return loaded_Nx, loaded_Nxt, loaded_N

    def __len__(self):
        return len(self.Nx)

    #   CHECK
    def __copy__(self):
        return deepcopy(self)


    #   CHECK
    def __deepcopy__(self, memo):
        result = Unigram(
            dictionary=deepcopy(self.dictionary),
            Nx=self.Nx,
            verbose=self.verbose
        )
        memo[id(self)] = result
        return result


    #   CHECK
    def __iter__(self):
        """
        Returns (Nx, N). So that the Unigram instance easily unpacks
        into unigram counts, and total count.
        """
        return iter(self[h.shards.whole])

    
    #   CHECK
    def __add__(self, other):
        """
        Create a new Unigram that has counts from both operands.
        """

        if not isinstance(other, Unigram):
            return NotImplemented

        result = deepcopy(self)
        result.__iadd__(other)
        return result


    #   CHECK
    def __iadd__(self, other):
        """
        Add counts from `other` to `self`, in place.
        """

        if not isinstance(other, Unigram):
            return NotImplemented

        self.sort_by_tokens(other.dictionary.tokens)

        for i in range(len(other.dictionary)):
            other_count = other.Nx[i]
            self.Nx[i] += other_count

        self.N = sum(self.Nx)

        return self


    # Converted
    def validate_args(self, dictionary, Nx):
        # Dictionaries are mandatory.
        if Nx is not None and dictionary is None:
            raise ValueError(
                'A dictionary must be provided to create a non-empty '
                'Unigram object.'
            )


    # TODO: TEST
    def count(self, token):
        token_id = self.dictionary.get_id(token)
        return self.Nx[token_id]


    def freq(self, token):
        return self.freq_id(self.dictionary.get_id(token))


    def freq_id(self, token_idx):
        return self.Nx[token_idx] / self.N

        
    def add(self, token, count=1):
        idx = self.dictionary.add_token(token)
        if idx == len(self.Nx):
            self.Nx.append(count)
        elif idx < len(self.Nx):
            self.Nx[idx] += count
        else:
            raise ValueError(
                'Unigram out of sync with Dictionary: got ID %d for token %s, '
                'out of bounds for Nx of length %d' 
                % (idx, token, len(self.Nx))
            )
        self.N += count


    def sort_by_tokens(self, token_order):
        """
        Reassign indexes for tokens in token_order.  Tokens not found in 
        token order are assigned higher indexes.  If tokens in token order are
        not in vocabulary add them.  The token at position i in token order
        gets index i.
        """

        remaining_tokens = list(set(self.dictionary.tokens) - set(token_order))
        token_order = token_order + remaining_tokens
        idx_order = [self.dictionary.add_token(token) for token in token_order]

        self.Nx += [0] * (len(token_order) - len(self.Nx))
        self.sort_by_idxs(idx_order)


    def sort_by_idxs(self, idx_order):
        """
        Re-assign indexes.  Assumes that there is no change in vocabulary.  If
        the index at location i is j, then the word with old index j obtains
        new index i.  Re-order storage of Nx and dictionary accordingly.
        """
        self.Nx = [self.Nx[idx] for idx in idx_order]
        self.dictionary = h.dictionary.Dictionary(
            [self.dictionary.tokens[idx] for idx in idx_order])


    def sort(self):
        """
        Re-assign token indices.  The most common words get the lowest indices.
        This affects the dictionary mapping, and The indexing of Nx.
        """

        # momentarily convert into numpy, to take advantage of their easy 
        # sorting.
        self.Nx = np.array(self.Nx)
        top_indices = np.argsort(-self.Nx)
        self.Nx = list(self.Nx[top_indices])
        self.dictionary = h.dictionary.Dictionary([
            self.dictionary.tokens[i] for i in top_indices])


    def save(self, path, save_dictionary=True):
        """
        Save the count data and dictionary to disk.  A new directory will be
        created at `path`, and two files will be created within it to store the
        counts and the token-ID mapping.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, 'Nx.txt'), 'w') as f_counts:
            f_counts.write('\n'.join([str(count) for count in self.Nx]))
        if save_dictionary:
            self.dictionary.save(os.path.join(path, 'dictionary'))


    def truncate(self, k):
        """Drop all but the `k` most common words."""
        self.Nx = self.Nx[:k]
        self.N = sum(self.Nx)
        self.dictionary = h.dictionary.Dictionary(self.dictionary.tokens[:k])


    @staticmethod
    def load(path, verbose=True):
        """
        Load the token-ID mapping and cooccurrence data previously saved in
        the directory at `path`.
        """
        dictionary = h.dictionary.Dictionary.load(
            os.path.join(path, 'dictionary'))
        with open(os.path.join(path, 'Nx.txt')) as f_counts:
            Nx = [int(count) for count in f_counts]
        return Unigram(dictionary=dictionary, Nx=Nx, verbose=verbose)



