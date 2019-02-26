import os
import warnings

import hilbert as h

try:
    import torch
    import numpy as np
except ImportError:
    torch = None
    np = None

def random(
    vocab, d, include_covectors=True, include_biases=False, dictionary=None,
    seed=None, distribution='uniform', scale=0.2, device=None
):
    """
    Obtain ``Embeddings`` containing ``vocab`` number of ``d``-dimensional
    vectors, whose components are sampled randomly.  Optionaly provide a
    `dictionary` to associate.   

    If include_covectors is ``True``, both vectors and covectors are sampled.
    If include biases is ``True``, then the embeddings will be initialized 
    with biases for each vector and covector, which will all be zero.

    You may provide a random ``seed`` for replicability.  

    You can set the ``device`` to ``'cpu'`` or ``'cuda'``.

    Components are uniformly sampled in the range ``[-scale,scale]``, with
    ``scale`` defaulting to 0.2.  Optionally set ``distribution=normal``
    to sample a Gaussian with mean 0 and standard deviation equal to ``scale``.
    """

    if seed is not None:
        np.random.seed(seed)

    W = None
    v_bias, w_bias = None, None
    if distribution == 'uniform':
        V = np.random.uniform(-scale, scale, (vocab, d)).astype(np.float32) 
        if include_covectors:
            W = np.random.uniform(-scale, scale, (vocab, d)).astype(np.float32) 

    elif distribution == 'normal':
        V = np.random.normal(0, scale, (vocab, d)).astype(np.float32)
        if include_covectors:
            W = np.random.normal(0, scale, (vocab, d)).astype(np.float32)

    if include_biases:
        v_bias, w_bias = torch.zeros(vocab), torch.zeros(vocab)

    return Embeddings(
        V, W=W, v_bias=v_bias, w_bias=w_bias, dictionary=dictionary,
        device=device
    ) 
    


# TODO: include biases as part of saving and loading.
class Embeddings:
    """
    Creates new embeddings.  The vector embeddings based on the 2D torch
    tensor or numpy array ``V``.  Each row in ``V`` should correspond to
    a vector for one word (or item).

    Similarly, you can provide covectors ``W``, or leave ``W`` as None.  Each
    row in ``W`` should correspond to the covector for one word.

    If you provide a ``hilbert.dictionary.Dictionary``, then you will
    be able to access vectors and covectors by name.

    Optionally specify the ``device``.

    If ``normalize`` is True, then normalize the vectors if they are not 
    already normalized.
    """

    def __init__(
        self, V, W=None, v_bias=None, w_bias=None, dictionary=None,
        device=None, normalize=False
    ):

        # Store the choice of device, use (but don't save) default if needed.
        self.device = device
        dtype = h.CONSTANTS.DEFAULT_DTYPE

        # Own the provided tensors (note, this copies)
        self.V = torch.tensor(V, dtype=dtype, device=device)
        self.W = (None if W is None else torch.tensor(
            W, dtype=dtype, device=device))
        self.v_bias = (None if v_bias is None else torch.tensor(
            v_bias, dtype=dtype, device=device))
        self.w_bias = (None if w_bias is None else torch.tensor(
            w_bias, dtype=dtype, device=device))
        self.set_dictionary(dictionary)

        self.validate()

        self._unkV = None
        self._unkW = None
        self._unkv_bias = None
        self._unkw_bias = None
        self.check_normalized()
        if normalize:
            self.normalize()

    def validate(self):
        # Note, most validation is performed here, but validating that the
        # dictionary length is correct (if provided) is done within
        # `set_dictionary()`
        if self.W is None and self.w_bias is not None:
            raise ValueError(
                'Embeddings should not include covector biases (``w_bias``) '
                'if covectors (``W``) were not provided.'
            )

        if self.v_bias is not None and self.V.shape[0] != self.v_bias.shape[0]:
            raise ValueError(
                'The number of vector embeddings and vector biases do not '
                'match.  Got {} vectors and {} biases'.format(
                    self.V.shape[0], self.v_bias.shape[0]
                )
            )

        # The following validations only apply when covectors are provided
        if self.W is None:
            return

        if self.W.shape != self.V.shape:
            raise ValueError(
                'The number and length of vectors ({}, {}) does not match '
                'the number and length of cvectors ({}, {}).'.format(
                    self.V.shape[0], self.V.shape[1], self.W.shape[0], 
                    self.W.shape[1]
                )
            )

        one_bias_not_none = self.w_bias is not None or self.v_bias is not None 
        both_not_none = self.w_bias is not None and self.v_bias is not None 
        if one_bias_not_none and not both_not_none:
            raise ValueError(
                'Embeddings that have covectors should either '
                'have biases for both vectors and covectors, or neither.'
            )

        if self.w_bias is not None and self.W.shape[0] != self.w_bias.shape[0]:
            raise ValueError(
                'The number of covector embeddings and covector biases do not '
                'match.  Got {} covectors and {} biases'.format(
                    self.W.shape[0], self.w_bias.shape[0]
                )
            )
            

    @property
    def unk(self, for_W=False):
        """
        The vector for unkown (out-of-vocabulary) tokens.  Equal to the
        centroid of the vectors.
        """
        if self._unkV is None:
            self._unkV = self.V.mean(0)
        return self._unkV


    @property
    def unkV(self):
        """
        The vector for unkown (out-of-vocabulary) tokens.  Equal to the
        centroid of the vectors.
        """
        if self._unkV is None:
            self._unkV = self.V.mean(0)
        return self._unkV


    @property
    def unkW(self):
        """
        The covector for unkown (out-of-vocabulary) tokens.  Equal to the
        centroid of the covectors.
        """
        if self._unkW is None:
            self._unkW = self.W.mean(0)
        return self._unkW


    @property
    def unkv_bias(self, for_W=False):
        """
        The bias for unkown (out-of-vocabulary) tokens.  Equal to the mean
        vector bias.
        """
        if self._unkv_bias is None:
            self._unkv_bias = self.v_bias.mean(0)
        return self._unkv_bias


    @property
    def unkw_bias(self, for_W=False):
        """
        The bias for unkown (out-of-vocabulary) tokens.  Equal to the mean
        vector bias.
        """
        if self._unkw_bias is None:
            self._unkw_bias = self.w_bias.mean(0)
        return self._unkw_bias


    def sort_like(self, other, allow_mismatch=False):
        self.sort_by_tokens(other.dictionary.tokens, allow_mismatch)


    def sort_by_tokens(
        self, tokens_or_dictionary, allow_mismatch=False,
        allow_missed_tokens=False, allow_missed_embeddings=False
    ):
        """
        Re-orders vectors / covectors, by assigning new indices to 
        tokens according to their position in ``tokens``.  ``tokens`` should
        be an iterable of strings, and it should have exactly the same 
        elements as self.dictionary.tokens, otherwise it's an error.
        """

        # The input can a dictionary or a list of tokens.
        if isinstance(tokens_or_dictionary, h.dictionary.Dictionary):
            dictionary = tokens_or_dictionary
            tokens = dictionary.tokens
        else:
            tokens = tokens_or_dictionary
            dictionary = h.dictionary.Dictionary(tokens)

        self_set = set(self.dictionary.tokens)
        other_set = set(tokens)

        # Check for extraneous tokens.
        self_coverage = other_set - self_set
        if len(self_coverage) > 0:

            if allow_mismatch or allow_missed_embeddings:
                # Drop the extraneous tokens
                tokens = [token for token in tokens if token in self.dictionary]

            else:
                print('missed:"{}"'.format(list(self_coverage)[0]))
                raise ValueError(
                    'The new dictionary has {} tokens that do not have a '
                    'corresponding embedding.\n{}"'.format(
                        len(self_coverage), '"\n"'.join(self_coverage))
                )

        # Check for missing tokens.
        dictionary_coverage = self_set - other_set
        if len(dictionary_coverage) > 0 :
            if not allow_mismatch and not allow_missed_tokens:
                raise ValueError(
                    "The new dictionary is missing {} entries for some "
                    "embedded tokens.\n{}".format(
                        len(dictionary_coverage),
                        '\n'.join(dictionary_coverage)
                    )
                )
            else:
                print(
                    'Warning, some embeddings were dropped because they have '
                    'no corresponding token in the provided dictoinary.'
                )

        sort_ids = [self.dictionary.get_id(token) for token in tokens]

        # This implicitly drops missing tokens
        self.V = self.V[sort_ids]
        if self.W is not None:
            self.W = self.W[sort_ids]
        if self.v_bias is not None:
            self.v_bias = self.v_bias[sort_ids]
        if self.w_bias is not None:
            self.w_bias = self.w_bias[sort_ids]

        self.dictionary = h.dictionary.Dictionary(tokens)


    def set_dictionary(self, dictionary):
        if dictionary is None:
            self.dictionary = dictionary

        elif len(dictionary) == self.V.shape[0]:
            self.dictionary = dictionary

        else:
            raise ValueError(
                "Dictionary provided does not have the correct number of "
                "tokens.  Got {num_tokens}, expected {num_vecs}".format(
                    num_tokens=len(dictionary.tokens), num_vecs=self.V.shape[0]
                )
            )


    def check_normalized(self):
        """
        Returns ``True`` if vectors and covectors have unit norm.  
        Sets ``self.normed``
        """
        device = self.device
        ones = torch.ones(self.V.shape[0], device=device)
        V_normed = torch.allclose(h.utils.norm(self.V, axis=1), ones)
        if self.W is None:
            self.normed = V_normed
            return V_normed

        W_normed = torch.allclose(h.utils.norm(self.W, axis=1), ones)
        self.normed = V_normed and W_normed
        return self.normed


    def normalize(self):
        """
        Normalize the vectors if they aren't already normed.
        """
        if self.normed:
            return
        self._normalize()


    def _normalize(self):
        """
        Normalize the vectors.
        """
        self.V = h.utils.normalize(self.V, axis=1)
        if self.W is not None:
            self.W = h.utils.normalize(self.W, axis=1)
        self.normed = True


    def __iter__(self):
        """
        Allows the embeddings to easily be unpacked into their underlying
        tensors and dictionary, using ``V, W, dictionary = embeddings``.
        """
        warnings.warn(
            "Implicit unpacking of embeddings is deprecated. ",
            DeprecationWarning
        )
        return iter((self.V, self.W, self.dictionary))


    def greatest_product(self, key):
        """
        Given an index or word, list all other indices or words from 
        highest to lowest inner product with the given one.
        """
        query_vec = self[key]
        query_id = self._as_id(key)
        inner_products = self.V @ query_vec
        top_indices = np.argsort(-inner_products)
        if isinstance(key, str):
            return [self.dictionary.get_token(idx) for idx in top_indices
                if idx != query_id
            ]
        else:
            return torch.tensor([idx for idx in top_indices if idx != query_id])


    def greatest_product_one(self, key):
        """
        Given an index or word, return the index or word whose corresponding
        embedding has the highest inner product with that of the given index
        or word.  There is no performance gain over calling
        ``self.greatest_product``.
        """
        return self.greatest_product(key)[0]


    def greatest_cosine(self, key):
        """
        Given an index or word, list all other indices or words from 
        highest to lowest cosine similarity with the given one.
        """
        if not self.normed:
            normed_V = h.utils.normalize(self.V, axis=1)
        else:
            normed_V = self.V

        normed_query_vec = h.utils.normalize(self[key], axis=0)
        query_id = self._as_id(key)
        inner_products = normed_V @ normed_query_vec
        top_indices = np.argsort(-inner_products)
        if isinstance(key, str):
            return [
                self.dictionary.get_token(idx) for idx in top_indices
                if idx != query_id
            ]
        else:
            return torch.tensor([idx for idx in top_indices if idx != query_id])


    def greatest_cosine_one(self, key):
        """
        Given an index or word, return the index or word whose corresponding
        embedding has the highest inner product with that of the given index
        or word.  There is no performance gain over calling
        ``self.greatest_cosine``.
        """
        return self.greatest_cosine(key)[0]

        
    def save(self, path):
        """
        Save the Vectors, Covectors, and dictionary in a new directory at
        ``path``.  If Covectors or dictionary are None, no file will be written
        for them.

        Vectors are always stored using numpy's format, but are held in memory
        as torch tensors.
        """

        if not os.path.exists(path):
            os.makedirs(path)

        np.save(os.path.join(path, 'V.npy'), self.V.cpu().numpy())

        if self.W is not None:
            np.save(os.path.join(path, 'W.npy'), self.W.cpu().numpy())

        if self.v_bias is not None:
            np.save(os.path.join(path, 'v_bias.npy'), self.v_bias.cpu().numpy())

        if self.w_bias is not None:
            np.save(os.path.join(path, 'w_bias.npy'), self.w_bias.cpu().numpy())

        if self.dictionary is not None:
            self.dictionary.save(os.path.join(path, 'dictionary'))


    def _as_slice(self, key):
        if isinstance(key, str):
            if self.dictionary is None:
                raise ValueError(
                    "Can't access vectors by token: these embeddings carry no "
                    "dictionary!"
                )
            return self.dictionary.get_id(key)
        return key


    def _as_id(self, id_or_token):
        if isinstance(id_or_token, str):
            if self.dictionary is None:
                raise ValueError(
                    "Can't access vectors by token: these embeddings carry no "
                    "dictionary!"
                )
            return self.dictionary.get_id(id_or_token)
        return id_or_token


    def get_vec(self, key, oov_policy='err'):
        """
        Gets the embedding for vector ``key``.  Key can either be an ``int``
        representing the index of the embedding in V, or it can be the
        name of the embedded word.  The embeddings have to have an associated 
        dictionary to access embeddings by name.

        If key is a ``str`` but is not found in ``self.dictionary``, then
        KeyError is raised.  But if ``oov_policy`` is ``'unk'``, then return 
        the centroid of the vectors.
        """
        if self.handle_out_of_vocab(key, oov_policy):
            return self.unk
        slice_obj = self._as_slice(key)
        return self.V[slice_obj]


    def get_covec(self, key, oov_policy='err'):
        """
        Gets the embedding for covector ``key``.  Key can either be an ``int``
        representing the index of the embedding in V, or it can be the
        name of the embedded word.  The embeddins have to have an associated 
        dictionary to access embeddings by name.

        If key is a ``str`` but is not found in ``self.dictionary``, then
        KeyError is raised.  But if ``oov_policy`` is ``'unk'``, then return 
        ``self.unk``.
        """
        if self.W is None:
            raise ValueError("This one-sided embedding has no co-vectors.")
        if self.handle_out_of_vocab(key, oov_policy):
            return self.unkW
        slice_obj = self._as_slice(key)
        return self.W[slice_obj]


    def get_vec_bias(self, key, oov_policy='err'):
        """
        Gets the bias for covector ``key``.  Key can either be an ``int``
        representing the index of the embedding in V, or it can be the
        name of the embedded word.  The embeddins have to have an associated 
        dictionary to access embeddings by name.

        If key is a ``str`` but is not found in ``self.dictionary``, then
        KeyError is raised.  But if ``oov_policy`` is ``'unk'``, then return 
        ``self.unk``.
        """
        if self.v_bias is None:
            raise ValueError("This embedding has biases.")
        if self.handle_out_of_vocab(key, oov_policy):
            return self.unkv_bias
        slice_obj = self._as_slice(key)
        return self.v_bias[slice_obj]



    def get_covec_bias(self, key, oov_policy='err'):
        """
        Gets the bias for covector ``key``.  Key can either be an ``int``
        representing the index of the embedding in V, or it can be the
        name of the embedded word.  The embeddins have to have an associated 
        dictionary to access embeddings by name.

        If key is a ``str`` but is not found in ``self.dictionary``, then
        KeyError is raised.  But if ``oov_policy`` is ``'unk'``, then return 
        ``self.unk``.
        """
        if self.W is None:
            raise ValueError("This embedding has no co-vectors.")
        if self.w_bias is None:
            raise ValueError("This embedding has biases.")
        if self.handle_out_of_vocab(key, oov_policy):
            return self.unkw_bias
        slice_obj = self._as_slice(key)
        return self.w_bias[slice_obj]


    def handle_out_of_vocab(self, key, policy):
        """
        If key is not in vocabulary, then
        - if policy=='err', raise ValueError.
        - if policy=='unk', return the ``self.unk`` embedding.
        """

        # Check that a meaninful policy was given
        if policy != 'err' and policy != 'unk':
            raise ValueError(
                "Unexpected out-of-vocabulary policy'.  Should be "
                "'err' or 'unk' (got %s)." % repr(handle_oov)
            )

        # Indices or slices are never considered out of vocabulary.  
        # TODO: test
        if isinstance(key, (int, tuple, slice)):
            return False

        # Handle missing dictionary
        if self.dictionary is None:
            raise ValueError(
                'Because embeddings have no dictionary , you can only select '
                'vectors using indices or slices, got %s.' % repr(key)
            )

        # Check if out of vocabulary.  If so, raise if policy says so.
        is_out_of_vocabulary = key not in self.dictionary
        if is_out_of_vocabulary and policy=='err':
            raise KeyError(
                'Token %s in out of vocabulary.  Pass ``policy="unk"`` to '
                'return the centroid embedder for '
                'out-of-vocaulary tokens.' % repr(key)
            )

        # Return whether the key is out of vocabulary
        return is_out_of_vocabulary


    def __getitem__(self, key):
        return self.get_vec(key, oov_policy='err')


    @staticmethod
    def load(path, device=None):
        """
        Static method for loading embeddings stored at ``path``.
        """

        V, W, v_bias, w_bias, dictionary = None, None, None, None, None

        dictionary_path = os.path.join(path, 'dictionary')
        if os.path.exists(dictionary_path):
            dictionary = h.dictionary.Dictionary.load(dictionary_path)
        V = np.load(os.path.join(path, 'V.npy'))
        if os.path.exists(os.path.join(path, 'W.npy')):
            W = np.load(os.path.join(path, 'W.npy'))
        if os.path.exists(os.path.join(path, 'v_bias.npy')):
            v_bias = np.load(os.path.join(path, 'v_bias.npy'))
        if os.path.exists(os.path.join(path, 'w_bias.npy')):
            w_bias = np.load(os.path.join(path, 'w_bias.npy'))

        return Embeddings(
            V, W=W, v_bias=v_bias, w_bias=w_bias, dictionary=dictionary,
            device=device
        )



