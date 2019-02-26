from unittest import TestCase
import hilbert as h
import random

try:
    import numpy as np
    from scipy import sparse
    import torch
except ImportError:
    np = None
    sparse = None
    torch = None


class TestUtils(TestCase):

    def test_normalize(self):

        dtype=h.CONSTANTS.DEFAULT_DTYPE
        device=h.CONSTANTS.MATRIX_DEVICE
        d = 300
        vocab = 5000

        V_numpy = np.random.random((vocab, d))

        norm = np.linalg.norm(V_numpy, ord=2, axis=1, keepdims=True)
        expected = V_numpy / norm
        found = h.utils.normalize(V_numpy, ord=2, axis=1)
        self.assertTrue(np.allclose(found, expected))
        self.assertTrue(V_numpy.shape, found.shape)
        self.assertTrue(np.allclose(
            np.linalg.norm(found, ord=2, axis=1), np.ones(vocab)))

        V_numpy = np.random.random((vocab, d))

        norm = np.linalg.norm(V_numpy, ord=2, axis=0, keepdims=True)
        expected = V_numpy / norm
        found = h.utils.normalize(V_numpy, ord=2, axis=0)
        self.assertTrue(np.allclose(found, expected))
        self.assertTrue(V_numpy.shape, found.shape)
        self.assertTrue(np.allclose(
            np.linalg.norm(found, ord=2, axis=0), np.ones(d)))

        V_torch = torch.rand((vocab, d), device=device, dtype=dtype)

        norm = torch.norm(V_torch, p=2, dim=1, keepdim=True)
        expected = V_torch / norm
        found = h.utils.normalize(V_torch, ord=2, axis=1)
        self.assertTrue(torch.allclose(found, expected))
        self.assertTrue(V_torch.shape, found.shape)
        self.assertTrue(torch.allclose(
            torch.norm(found, p=2, dim=1),
            torch.ones(vocab, device=device, dtype=dtype)
        ))


        norm = torch.norm(V_torch, p=2, dim=0, keepdim=True)
        expected = V_torch / norm
        found = h.utils.normalize(V_torch, ord=2, axis=0)
        self.assertTrue(torch.allclose(found, expected))
        self.assertTrue(V_torch.shape, found.shape)
        self.assertTrue(torch.allclose(
            torch.norm(found, p=2, dim=0),
            torch.ones(d, dtype=dtype, device=device)
        ))


    def test_norm(self):

        dtype=h.CONSTANTS.DEFAULT_DTYPE
        device=h.CONSTANTS.MATRIX_DEVICE
        d = 300
        vocab = 5000

        V_numpy = np.random.random((vocab, d))

        expected = np.linalg.norm(V_numpy, ord=2, axis=0, keepdims=True)
        found = h.utils.norm(V_numpy, ord=2, axis=0, keepdims=True)
        self.assertTrue(np.allclose(found, expected))

        expected = np.linalg.norm(V_numpy, ord=3, axis=1, keepdims=False)
        found = h.utils.norm(V_numpy, ord=3, axis=1, keepdims=False)
        self.assertTrue(np.allclose(found, expected))

        V_torch = torch.rand((vocab, d), device=device, dtype=dtype)

        expected = torch.norm(V_torch, p=2, dim=0, keepdim=True)
        found = h.utils.norm(V_torch, ord=2, axis=0, keepdims=True)
        self.assertTrue(np.allclose(found, expected))

        expected = torch.norm(V_torch, p=3, dim=1, keepdim=False)
        found = h.utils.norm(V_torch, ord=3, axis=1, keepdims=False)
        self.assertTrue(np.allclose(found, expected))


    def test_load_shard(self):

        shards = h.shards.Shards(5)
        dtype=h.CONSTANTS.DEFAULT_DTYPE
        device=h.CONSTANTS.MATRIX_DEVICE

        # Handles Numpy arrays properly.
        source = np.arange(100).reshape(10,10)
        expected = torch.tensor(
            [[0,5],[50,55]], device=device, dtype=dtype)
        found = h.utils.load_shard(source, shards[0], device=device)
        self.assertTrue(torch.allclose(found, expected))

        # Handles Scipy CSR sparse matrices properly.
        source = sparse.random(10,10,0.3).tocsr()
        expected = torch.tensor(
            source.toarray()[shards[0]], device=device, dtype=dtype)
        found = h.utils.load_shard(source, shards[0], device=device)
        self.assertTrue(torch.allclose(found, expected))

        # Handles Numpy matrices properly.
        source = np.matrix(range(100)).reshape(10,10)
        expected = torch.tensor(
            np.asarray(source)[shards[0]], device=device, dtype=dtype)
        found = h.utils.load_shard(source, shards[0], device=device)
        self.assertTrue(torch.allclose(found, expected))


