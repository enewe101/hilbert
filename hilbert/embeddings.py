import os

import hilbert as h

try:
    import torch
    import numpy as np
except ImportError:
    torch = None
    np = None

def random(
    vocab, d, dictionary=None, shared=False, implementation='torch',
    device='cpu', seed=None
):

    if seed is not None:
        np.random.seed(seed)

    V = np.random.random((vocab, d)).astype(np.float32)
    W = (
        h.utils.transpose(V) if shared else 
        np.random.random((vocab, d)).astype(np.float32)
    )

    return Embeddings(V, W, dictionary, shared, implementation, device) 
    


class Embeddings:

    def __init__(
        self, V, W=None, dictionary=None, shared=False, 
        implementation='torch', device='cuda', normalize=False
    ):
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

        If ``normalize`` is True, then normalize the vectors if they don't 
        already have norm.

        Specify to store ``V`` and ``W`` either as ``numpy.ndarray``s or 
        ``torch.Tensor``s, by setting ``implementation`` to ``'torch'`` or
        ``'numpy'`, and specify the ``device``, in the case of torch tensors.
        """
        self.dictionary = dictionary
        self.shared = shared
        self.implementation = implementation
        self.device = device

        self.V = (
            np.array(V, dtype=np.float32) if implementation == 'numpy' else
            torch.tensor(V, dtype=torch.float32, device=device)
        )

        if shared:
            self.W = h.utils.transpose(self.V)
        else:
            self.W = (
                None if W is None else np.array(W, dtype=np.float32)
                if implementation == "numpy" else 
                torch.tensor(W, dtype=torch.float32, device=device)
            )

        self.normed = self.check_normalized()
        if normalize:
            self.normalize()


    def check_normalized(self):
        """
        Checks if vectors and covectors all have unit norm.  
        Sets ``self.normed``
        """
        V_normed = np.allclose(h.utils.norm(self.V, axis=0), 1.0)
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
        self.V = h.utils.normalize(self.V, axis=0)
        if self.shared:
            self.W = h.utils.transpose(self.V)
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
        inner_products = h.utils.transpose(normed_embeddings) @ query_vec
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


    def get_vec(self, key):
        """
        Gets the embedding for a single vector, either using an ``int``
        or the name of the word as a ``str``.  The embeddings must have a
        dictionary to be albe to access embeddings by name.
        """
        slice_obj = self._as_slice(key)
        print(slice_obj)
        print(self.V.shape)
        return self.V[slice_obj]


    def get_covec(self, key):
        """
        Gets the embedding for a single covector, either using an ``int``
        or the name of the word as a ``str``.  The embeddings must have a
        dictionary to be albe to access embeddings by name.
        """
        if self.W is None:
            raise ValueError("This one-sided embedding has no co-vectors.")
        slice_obj = self._as_slice(key)
        return self.W[slice_obj]


    def __getitem__(self, key):
        return self.get_vec(key)


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
