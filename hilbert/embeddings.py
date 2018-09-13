import os
import torch
import numpy as np
import hilbert as h


def random(
    d, vocab, dictionary=None, shared=False, implementation='torch',
    device='cpu', seed=None
):

    if seed is not None:
        np.random.seed(seed)

    V = np.random.random((d, vocab)).astype(np.float32)
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
        V_normed = np.allclose(h.utils.norm(self.V, axis=0), 1.0)
        if self.shared or self.W is None:
            return V_normed

        W_normed = np.allclose(h.utils.norm(self.W, axis=1), 1.0)
        return V_normed and W_normed


    def normalize(self):
        if self.normed:
            return
        self.V = h.utils.normalize(self.V, axis=0)
        if self.shared:
            self.W = h.utils.transpose(self.V)
        else:
            self.W = h.utils.normalize(self.W, axis=1)
        self.normed = True


    def __iter__(self):
        return iter((self.V, self.W, self.dictionary))


    def greatest_product(self, key):
        query_vec = self[key]
        query_id = self.as_id(key)
        inner_products = h.utils.transpose(self.V) @ query_vec
        top_indices = np.argsort(-inner_products)
        if isinstance(key, str):
            return [self.dictionary.get_token(idx) for idx in top_indices
                if idx != query_id
            ]
        else:
            return torch.tensor([idx for idx in top_indices if idx != query_id])


    def greatest_product_one(self, key):
        return self.greatest_product(key)[0]


    def greatest_cosine(self, key):
        if not self.normed:
            normed_embeddings = h.utils.normalize(self.V, axis=0)
        else:
            normed_embeddings = self.V

        query_vec = self[key]
        query_id = self.as_id(key)
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
        return self.greatest_cosine(key)[0]

        
    def save(self, path):
        """
        Save the vectors, and, subject to them not being None, save the 
        covectors and dictionary, to disk, as files under a new directory 
        called path.  Saving is done using th numpy format, regardless of the
        in-memory implementation.  The particular implementation when read
        back into memory depends on options to the load method, not on the 
        files written.
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


    def as_id(self, id_or_token):
        if isinstance(id_or_token, str):
            if self.dictionary is None:
                raise ValueError(
                    "Can't access vectors by token: these embeddings carry no "
                    "dictionary!"
                )
            return self.dictionary.get_id(id_or_token)
        return id_or_token


    def get_vec(self, id_or_token):
        idx = self.as_id(id_or_token)
        return self.V[:,idx]


    def get_covec(self, id_or_token):
        if self.W is None:
            raise ValueError("This one-sided embedding has no co-vectors.")
        idx = self.as_id(id_or_token)
        return self.W[idx]


    def __getitem__(self, id_or_token):
        return self.get_vec(id_or_token)


    @staticmethod
    def load(path, shared=False, implementation="torch", device="cuda"):

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
