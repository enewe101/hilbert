import os
import shutil
from unittest import TestCase
from copy import copy
from collections import Counter
import hilbert as h

try:
    import torch
except ImportError:
    torch = None


def load_test_tokens():
    return load_tokens(h.CONSTANTS.TEST_TOKEN_PATH)
def load_tokens(path):
    with open(path) as f:
        return f.read().split()






class TestUnigram(TestCase):

    def test_unigram_creation_from_corpus(self):
        # Make a unigram and fill it with tokens and counts.
        unigram = h.unigram.Unigram()

        # An empty unigram is by definition sorted.
        self.assertTrue(unigram.sorted)

        # Add counts
        for token in load_test_tokens():
            unigram.add(token)

        # Adding counts disrupts the sorting
        self.assertFalse(unigram.sorted)

        # The correct number of counts are registered for each token
        counts = Counter(load_test_tokens())
        for token in counts:
            token_id = unigram.dictionary.get_id(token)
            self.assertEqual(unigram.Nx[token_id], counts[token])

        # Test sorting.
        unigram.sort()
        for i in range(len(unigram.Nx)-1):
            self.assertTrue(unigram.Nx[i] >= unigram.Nx[i+1])


    def test_apply_smoothing(self):

        alpha = 0.6

        # Make a unigram and fill it with tokens and counts.
        unigram = h.unigram.Unigram(verbose=False)
        for token in load_test_tokens():
            unigram.add(token)

        counts = list(unigram.Nx)
        expected_smoothed_Nx = [c**alpha for c in counts]
        expected_smoothed_N = sum(expected_smoothed_Nx)

        unigram.apply_smoothing(alpha)
        self.assertEqual(expected_smoothed_Nx, unigram.Nx)
        self.assertEqual(expected_smoothed_N, unigram.N)

        # Attempting to apply smoothing twice is an error
        with self.assertRaises(ValueError):
            unigram.apply_smoothing(alpha)


    def test_unigram_creation_from_Nx(self):
        tokens = load_test_tokens()
        dictionary = h.dictionary.Dictionary(tokens)
        Nx = [0] * len(dictionary)
        for token in tokens:
            Nx[dictionary.get_id(token)] += 1

        # Must supply a dictionary to create a non-empty Unigram.
        with self.assertRaises(ValueError):
            unigram = h.unigram.Unigram(Nx=Nx)

        unigram = h.unigram.Unigram(dictionary=dictionary, Nx=Nx)

        # Check that unigram knows it is not yet sorted
        self.assertFalse(unigram.sorted)

        # The correct number of counts are registered for each token
        counts = Counter(load_test_tokens())
        for token in counts:
            token_id = unigram.dictionary.get_id(token)
            self.assertEqual(unigram.Nx[token_id], counts[token])

        # Ensure that if adding counts undoes sorting, then the sorting flag
        # of unigram becomes false.
        unigram.sort()
        self.assertTrue(unigram.sorted)
        for i in range(5):
            unigram.add('Eat')
        self.assertFalse(unigram.sorted)


    def test_load_shard(self):

        device = h.CONSTANTS.MATRIX_DEVICE
        dtype = h.CONSTANTS.DEFAULT_DTYPE

        # Make a unigram and fill it with tokens and counts.
        unigram = h.unigram.Unigram()
        for token in load_test_tokens():
            unigram.add(token)

        # To simplify the test, sor the unigram
        unigram.sort()

        # Ensure that all shards are correct.
        shards = h.shards.Shards(3)
        for i, shard in enumerate(shards):
            expected_Nxs = [
                torch.tensor([24,8,8,4], dtype=dtype, device=device),
                torch.tensor([12,8,8,4], dtype=dtype, device=device),
                torch.tensor([12,8,8], dtype=dtype, device=device),
            ]
            expected_N = torch.tensor(104, dtype=dtype, device=device)
            Nx, Nxt, N = unigram.load_shard(shard)

            if i // 3 ==0:
                self.assertTrue(torch.allclose(Nx, expected_Nxs[0].view(-1,1)))
            elif i // 3 == 1:
                self.assertTrue(torch.allclose(Nx, expected_Nxs[1].view(-1,1)))
            elif i // 3 == 2:
                self.assertTrue(torch.allclose(Nx, expected_Nxs[2].view(-1,1)))

            if i % 3 == 0:
                self.assertTrue(torch.allclose(Nxt,expected_Nxs[0].view(1,-1)))
            elif i % 3 == 1:
                self.assertTrue(torch.allclose(Nxt,expected_Nxs[1].view(1,-1)))
            elif i % 3 == 2:
                self.assertTrue(torch.allclose(Nxt,expected_Nxs[2].view(1,-1)))

            self.assertEqual(Nxt.dtype, h.CONSTANTS.DEFAULT_DTYPE)
            self.assertEqual(Nx.dtype, h.CONSTANTS.DEFAULT_DTYPE)
            self.assertTrue(
                str(Nxt.device).startswith(h.CONSTANTS.MATRIX_DEVICE))
            self.assertTrue(
                str(Nx.device).startswith(h.CONSTANTS.MATRIX_DEVICE))
            self.assertTrue(isinstance(Nxt, torch.Tensor))
            self.assertTrue(isinstance(Nx, torch.Tensor))

        self.assertTrue(torch.allclose(N, expected_N))
        self.assertTrue(str(N.device).startswith(h.CONSTANTS.MATRIX_DEVICE))
        self.assertEqual(N.dtype, h.CONSTANTS.DEFAULT_DTYPE)


    def test_copy(self):

        # Make a unigram and fill it with tokens and counts.
        unigram1 = h.unigram.Unigram()
        for token in load_test_tokens():
            unigram1.add(token)

        unigram2 = copy(unigram1)

        # Objects are distinct.
        self.assertFalse(unigram2 is unigram1)
        self.assertFalse(unigram2.Nx is unigram1.Nx)
        self.assertFalse(unigram2.dictionary is unigram1.dictionary)

        # Objects are equal.
        self.assertEqual(unigram2.N, unigram1.N)
        self.assertEqual(unigram2.Nx, unigram1.Nx)
        self.assertEqual(
            unigram2.dictionary.tokens, unigram2.dictionary.tokens)
        self.assertEqual(
            unigram2.dictionary.token_ids, unigram1.dictionary.token_ids)


    def test_unpacking(self):

        device = h.CONSTANTS.MATRIX_DEVICE
        dtype = h.CONSTANTS.DEFAULT_DTYPE

        # Make a unigram and fill it with tokens and counts.
        unigram = h.unigram.Unigram()
        for token in load_test_tokens():
            unigram.add(token)

        # To simplify the test, sor the unigram
        unigram.sort()

        expected_Nx = torch.tensor(
            [24, 12, 12, 8, 8, 8, 8, 8, 8, 4, 4], dtype=dtype, device=device
        ).view(-1,1)
        expected_Nxt = torch.tensor(
            [24, 12, 12, 8, 8, 8, 8, 8, 8, 4, 4], dtype=dtype, device=device
        ).view(1,-1)
        expected_N = torch.tensor(104, dtype=dtype, device=device)
        Nx, Nxt, N = unigram
        self.assertTrue(torch.allclose(Nx, expected_Nx))
        self.assertTrue(torch.allclose(Nxt, expected_Nxt))
        self.assertTrue(torch.allclose(N, expected_N))


    def test_sort_by_tokens(self):

        # Get tokens in alphabetical order
        tokens = list(set(load_test_tokens()))
        tokens.sort()

        # Make a unigram and fill it with tokens and counts.
        unigram = h.unigram.Unigram()
        for token in load_test_tokens():
            unigram.add(token)

        # Unigram has same tokens, but is not in alphabetical order.
        self.assertCountEqual(unigram.dictionary.tokens, tokens)
        self.assertNotEqual(unigram.dictionary.tokens, tokens)

        # Sort by provided token list.  Now token lists have same order.
        unigram.sort_by_tokens(tokens)
        self.assertEqual(unigram.dictionary.tokens, tokens)

        # Sort by unigram order, to verify that calling sort_by_tokens resets
        # the sorted flag
        self.assertFalse(unigram.sorted)
        unigram.sort()
        self.assertTrue(unigram.sorted)
        unigram.sort_by_tokens(tokens)
        self.assertFalse(unigram.sorted)



    def test_add(self):
        # Make a unigram and fill it with tokens and counts.
        unigram1 = h.unigram.Unigram()
        for token in load_test_tokens():
            unigram1.add(token)

        unigram2 = h.unigram.Unigram()
        additional_tokens = 'the car is green .'.split()
        for token in additional_tokens:
            unigram2.add(token)

        expected_counts = Counter(
            load_test_tokens()+additional_tokens)

        # Orginary addition
        unigram4 = unigram1 + unigram2
        for token in expected_counts:
            token_idx = unigram4.dictionary.get_id(token)
            self.assertEqual(unigram4.Nx[token_idx], expected_counts[token])

        # In-place addition
        unigram3 = h.unigram.Unigram()
        unigram3 += unigram1    # adds larger into smaller.
        unigram3 += unigram2    # adds smaller into larger.
        for token in expected_counts:
            token_idx = unigram3.dictionary.get_id(token)
            self.assertEqual(unigram3.Nx[token_idx], expected_counts[token])


    def test_save_load(self):

        # Work out the path, and clear away anything that is currently there.
        write_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-save-load-unigram')
        if os.path.exists(write_path):
            shutil.rmtree(write_path)

        # Make a unigram and fill it with tokens and counts.
        unigram1 = h.unigram.Unigram()
        for token in load_test_tokens():
            unigram1.add(token)

        # Do a save and load cycle
        unigram1.save(write_path)
        unigram2 = h.unigram.Unigram.load(write_path)

        # Objects are distinct.
        self.assertFalse(unigram2 is unigram1)
        self.assertFalse(unigram2.Nx is unigram1.Nx)
        self.assertFalse(unigram2.dictionary is unigram1.dictionary)

        # Objects are equal.
        self.assertEqual(unigram2.N, unigram1.N)
        self.assertEqual(unigram2.Nx, unigram1.Nx)
        self.assertEqual(
            unigram2.dictionary.tokens, unigram2.dictionary.tokens)
        self.assertEqual(
            unigram2.dictionary.token_ids, unigram1.dictionary.token_ids)

        # Cleanup.
        shutil.rmtree(write_path)


    def test_truncate(self):

        # Make a unigram and fill it with tokens and counts.
        unigram = h.unigram.Unigram()
        for token in load_test_tokens():
            unigram.add(token)

        expected_tokens = ['.', 'the', 'The', 'Eat', 'sandwich']
        expected_Nx = [24, 12, 12, 8, 8]
        expected_N = sum(expected_Nx)
        unigram.truncate(5)

        self.assertEqual(unigram.Nx, expected_Nx)
        self.assertEqual(unigram.N, expected_N)
        self.assertEqual(unigram.dictionary.tokens, expected_tokens)
        self.assertEqual(
            set(unigram.dictionary.token_ids.keys()), set(expected_tokens))



