import os

import hilbert as h

try:
    import torch
    import numpy as np
except ImportError:
    torch = None
    np = None

def random(
    vocab, d, dictionary=None, shared=False, seed=None,
    distribution='uniform', scale=0.2,
    implementation='torch', device='cuda', 
    
):
    """
    Obtain ``Embeddings`` containing ``vocab`` number of ``d``-dimensional
    vectors, whose components are sampled randomly.  Optionaly provide a
    `dictionary` to associate.   

    By default, both vectors and covectors are sampled, independently.  If
    ``shared`` is True, then only vectors are sampled, and the covectors simply
    point to the same memory as vectors.
    
    You may provide a random ``seed`` for replicability.  ``implementation``
    can be ``'torch'`` or ``'numpy'``, and if ``'torch'`` is used, you can set
    the ``device``.

    Components are uniformly sampled in the range ``[-scale,scale]``, with
    ``scale`` defaulting to 0.2.  Optionally set ``distribution=normal``
    to sample a Gaussian with mean 0 and standard deviation equal to ``scale``.
    """

    if seed is not None:
        np.random.seed(seed)

    if distribution == 'uniform':
        V = np.random.uniform(-scale, scale, (vocab, d)).astype(np.float32) 
        if shared:
            W = V 
        else:
            W = np.random.uniform(-scale, scale, (vocab, d)).astype(np.float32) 

    elif distribution == 'normal':
        V = np.random.normal(0, scale, (vocab, d)).astype(np.float32)
        if shared:
            W = V 
        else:
            np.random.normal(0, scale, (vocab, d)).astype(np.float32)

    return Embeddings(V, W, dictionary, shared, implementation, device) 
    


class Embeddings:
    """
    Creates new embeddings.  The vector embeddings based on the 2D torch
    tensor or numpy array ``V``.  Each row in ``V`` should correspond to
    a vector for one word (or item).

    Similarly, you can provide covectors ``W``, or leave ``W`` as None.  
    Another option is to have ``V`` and ``W`` be identical pointers to the
    same memory by passing ``shared=True``.  Like ``V``, each row in ``W`` 
    should correspond to the covector for one word.

    If you provide a ``hilbert.dictionary.Dictionary``, then you will
    be able to access vectors and covectors by name.

    If ``normalize`` is True, then normalize the vectors if they are not 
    already normalized.

    Specify to store ``V`` and ``W`` either as ``numpy.ndarray``s or 
    ``torch.Tensor``\ s, by setting ``implementation`` to ``'torch'`` or
    ``'numpy'`, and specify the ``device``, in the case of torch tensors.
    """

    def __init__(
        self, V, W=None, dictionary=None, shared=False, 
        implementation='torch', device='cuda', normalize=False
    ):
        self.dictionary = dictionary
        self.shared = shared
        self.implementation = implementation
        self.device = device

        self.V = (
            np.array(V, dtype=np.float32) if implementation == 'numpy' else
            torch.tensor(V, dtype=torch.float32, device=device)
        )

        if shared:
            self.W = self.V
        else:
            self.W = (
                None if W is None else np.array(W, dtype=np.float32)
                if implementation == "numpy" else 
                torch.tensor(W, dtype=torch.float32, device=device)
            )

        self._unkV = None
        self._unkW = None
        self.normed = self.check_normalized()
        if normalize:
            self.normalize()


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


    def check_normalized(self):
        """
        Returns true if vectors and covectors have unit norm.  
        Sets ``self.normed``
        """
        V_normed = np.allclose(h.utils.norm(self.V, axis=1), 1.0)
        if self.shared or self.W is None:
            self.normed = V_normed
            return V_normed

        W_normed = np.allclose(h.utils.norm(self.W, axis=1), 1.0)
        self.normed = V_normed and W_normed
        return V_normed and W_normed


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
        if self.shared:
            self.W = self.V
        else:
            self.W = h.utils.normalize(self.W, axis=1)
        self.normed = True


    def __iter__(self):
        """
        Allows the embeddings to easily be unpacked into their underlying
        tensors and dictionary, using ``V, W, dictionary = embeddings``.
        """
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
            normed_embeddings = h.utils.normalize(self.V, axis=0)
        else:
            normed_embeddings = self.V

        query_vec = self[key]
        query_id = self._as_id(key)
        inner_products = normed_embeddings @ query_vec
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

        Vectors are alwasy stored using numpy's format.  Their implementation
        upon reading is decided only by the ``implementation`` argument in 
        the ``load`` method, see below.``
        """

        if not os.path.exists(path):
            os.makedirs(path)

        save_V = (
            self.V if isinstance(self.W, np.ndarray) else 
            np.array(self.V, dtype=np.float32)
        )
        np.save(os.path.join(path, 'V.npy'), self.V)

        if self.W is not None:
            save_W = (
                self.W if isinstance(self.W, np.ndarray) else 
                np.array(self.W, dtype=np.float32)
            )
            np.save(os.path.join(path, 'W.npy'), save_W)

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
        name of the embedded word.  The embeddins have to have an associated 
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


    def handle_out_of_vocab(self, key, policy):
        """
        If key is not in vocabulary, then
        - if policy=='err', raise ValueError.
        - if policy=='unk', return the ``self.unk`` embedding.
        """

        if policy != 'err' and policy != 'unk':
            raise ValueError(
                "Unexpected out-of-vocabulary policy'.  Should be "
                "'err' or 'unk' (got %s)." % repr(handle_oov)
            )

        # Only tokens are possibly out of vocabulary.  Indices or slices are
        # never considered out of vocabulary.
        # TODO: test
        if isinstance(key, (int, tuple, slice)):
            return False

        # tokens that are in the dictionary are not out of vocabulary
        is_out_of_vocabulary = key not in self.dictionary
        if is_out_of_vocabulary and policy=='err':
            raise KeyError(
                'Token %s in out of vocabulary.  '
                'Pass ``policy="unk"`` to return the centroid embedder for '
                'out-of-vocaulary tokens' % repr(key),
            )

        return is_out_of_vocabulary


    def __getitem__(self, key):
        return self.get_vec(key, oov_policy='err')


    @staticmethod
    def load(path, shared=False, implementation="torch", device="cuda"):
        """
        Static method for loading embeddings stored at ``path``.
        The arguments ``shared``, ``implementation``, and ``device`` have the
        same effect as in the constructor.
        """

        V, W, dictionary = None, None, None

        dictionary_path = os.path.join(path, 'dictionary')
        if os.path.exists(dictionary_path):
            dictionary = h.dictionary.Dictionary.load(dictionary_path)

        V_path = os.path.join(path, 'V.npy')
        if os.path.exists(V_path):
            V = np.load(V_path)

        W_path = os.path.join(path, 'W.npy')
        if os.path.exists(os.path.join(path, 'W.npy')):
            W = np.load(W_path)

        return Embeddings(V, W, dictionary, shared, implementation, device)



