import os
import shutil
import hilbert as h
import random
import numpy as np
import torch
from unittest import TestCase

def get_test_dictionary():
    return h.dictionary.Dictionary.load(
        os.path.join(h.CONSTANTS.TEST_DIR, 'dictionary'))

# TODO: include biases in tests


class TestEmbeddings(TestCase):

    def test_creating_embeddings(self):
        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()
        device = h.CONSTANTS.MATRIX_DEVICE
        V = torch.rand(vocab, d, device=device)
        v_bias = torch.rand(vocab, device=device)
        W = torch.rand(vocab, d, device=device)
        w_bias = torch.rand(vocab, device=device)


        # Create embeddings with V, W, v_bias, and w_bias
        embeddings = h.embeddings.Embeddings(
            V, W=W, v_bias=v_bias, w_bias=w_bias, dictionary=dictionary,
            device=device
        )
        self.assertFalse(embeddings.V is V)
        self.assertTrue(torch.allclose(embeddings.V, V))
        self.assertFalse(embeddings.W is W)
        self.assertTrue(torch.allclose(embeddings.W, W))
        self.assertFalse(embeddings.v_bias is v_bias)
        self.assertTrue(torch.allclose(embeddings.v_bias, v_bias))
        self.assertFalse(embeddings.w_bias is w_bias)
        self.assertTrue(torch.allclose(embeddings.w_bias, w_bias))
        self.assertTrue(embeddings.dictionary is dictionary)

        # Create embeddings with V, W, no bias
        embeddings = h.embeddings.Embeddings(
            V, W=W, dictionary=dictionary, device=device)
        self.assertFalse(embeddings.V is V)
        self.assertTrue(torch.allclose(embeddings.V, V))
        self.assertFalse(embeddings.W is W)
        self.assertTrue(torch.allclose(embeddings.W, W))
        self.assertTrue(embeddings.v_bias is None)
        self.assertTrue(embeddings.w_bias is None)
        self.assertTrue(embeddings.dictionary is dictionary)

        # Create embeddings with V, v_bias, no W or w_bias
        embeddings = h.embeddings.Embeddings(
            V, v_bias=v_bias, dictionary=dictionary, device=device)
        self.assertFalse(embeddings.V is V)
        self.assertTrue(torch.allclose(embeddings.V, V))
        self.assertTrue(embeddings.W is None)
        self.assertFalse(embeddings.v_bias is v_bias)
        self.assertTrue(torch.allclose(embeddings.v_bias, v_bias))
        self.assertTrue(embeddings.w_bias is None)
        self.assertTrue(embeddings.dictionary is dictionary)

        # Create embeddings with V, v_bias, no W or w_bias
        embeddings = h.embeddings.Embeddings(
            V, v_bias=v_bias, dictionary=dictionary, device=device)
        self.assertFalse(embeddings.V is V)
        self.assertTrue(torch.allclose(embeddings.V, V))
        self.assertTrue(embeddings.W is None)
        self.assertFalse(embeddings.v_bias is v_bias)
        self.assertTrue(torch.allclose(embeddings.v_bias, v_bias))
        self.assertTrue(embeddings.w_bias is None)
        self.assertTrue(embeddings.dictionary is dictionary)

        # Cannot create embeddings with only one bias but not the other
        with self.assertRaises(ValueError):
            embeddings = h.embeddings.Embeddings(
                V, W=W, v_bias=v_bias, dictionary=dictionary, device=device)
        with self.assertRaises(ValueError):
            embeddings = h.embeddings.Embeddings(
                V, W=W, w_bias=w_bias, dictionary=dictionary, device=device)

        # Cannot create embeddings with w_bias if there is no W
        with self.assertRaises(ValueError):
            embeddings = h.embeddings.Embeddings(
                V, v_bias=v_bias, w_bias=w_bias, dictionary=dictionary,
                device=device
            )

        # Cannot create embeddings if embeddings/biases don't have shape match.
        with self.assertRaises(ValueError):
            embeddings = h.embeddings.Embeddings(V, W=W[:-1], device=device)
        with self.assertRaises(ValueError):
            embeddings = h.embeddings.Embeddings(V, W=W[:,:-1], device=device)
        with self.assertRaises(ValueError):
            embeddings = h.embeddings.Embeddings(
                V, W=W, v_bias=v_bias, w_bias=w_bias[:-1], device=device)
        with self.assertRaises(ValueError):
            embeddings = h.embeddings.Embeddings(
                V, W=W, v_bias=v_bias[:-1], w_bias=w_bias, device=device)

        # Try making embeddings using an incorrectly-lengthed dictionary.
        V = V[:-5]
        with self.assertRaises(ValueError):
            embeddings = h.embeddings.Embeddings(
                V, dictionary=dictionary, device=device)


    def test_random(self):
        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()

        # Can make random embeddings and provide a dictionary to use.
        embeddings = h.embeddings.random(vocab, d, dictionary=dictionary)
        self.assertEqual(embeddings.V.shape, (vocab, d))
        self.assertEqual(embeddings.W.shape, (vocab, d))
        self.assertTrue(embeddings.dictionary is dictionary)

        # Can have random embeddings without covectors.
        embeddings = h.embeddings.random(
            vocab, d, dictionary=dictionary, include_covectors=False)
        self.assertEqual(embeddings.V.shape, (vocab, d))
        self.assertTrue(embeddings.W is None)
        self.assertTrue(embeddings.dictionary is dictionary)

        # Can have random embeddings with biases.
        zeros = torch.zeros(vocab)
        embeddings = h.embeddings.random(
            vocab, d, dictionary=dictionary, include_biases=True)
        self.assertEqual(embeddings.V.shape, (vocab, d))
        self.assertTrue(embeddings.W.shape, (vocab, d))
        self.assertTrue(torch.allclose(embeddings.v_bias, zeros))
        self.assertTrue(torch.allclose(embeddings.w_bias, zeros))
        self.assertTrue(embeddings.dictionary is dictionary)

        # Can omit the dictionary
        embeddings = h.embeddings.random(vocab, d, dictionary=None)
        self.assertEqual(embeddings.V.shape, (vocab, d))
        self.assertEqual(embeddings.W.shape, (vocab, d))
        self.assertTrue(embeddings.dictionary is None)

        # Uses torch.
        embeddings = h.embeddings.random(vocab, d, dictionary=dictionary)
        self.assertTrue(isinstance(embeddings.V, torch.Tensor))
        self.assertTrue(isinstance(embeddings.W, torch.Tensor))


    def test_random_distribution(self):
        d = 300
        vocab = 5000

        dictionary = get_test_dictionary()

        # Can make numpy random embeddings with uniform distribution
        embeddings = h.embeddings.random(
            vocab, d, dictionary=dictionary, distribution='uniform', scale=0.2,
            seed=0
        )

        np.random.seed(0)
        expected_uniform_V = np.random.uniform(-0.2, 0.2, (vocab, d))
        expected_uniform_W = np.random.uniform(-0.2, 0.2, (vocab, d))

        self.assertTrue(isinstance(embeddings.V, torch.Tensor))
        self.assertTrue(isinstance(embeddings.W, torch.Tensor))
        self.assertTrue(np.allclose(embeddings.V, expected_uniform_V))
        self.assertTrue(np.allclose(embeddings.W, expected_uniform_W))


        # Can make numpy random embeddings with normal distribution
        embeddings = h.embeddings.random(
            vocab, d, dictionary=dictionary, distribution='normal', scale=0.2,
            seed=0
        )

        np.random.seed(0)
        expected_normal_V = np.random.normal(0, 0.2, (vocab, d))
        expected_normal_W = np.random.normal(0, 0.2, (vocab, d))

        self.assertTrue(isinstance(embeddings.V, torch.Tensor))
        self.assertTrue(isinstance(embeddings.W, torch.Tensor))
        self.assertTrue(np.allclose(embeddings.V, expected_normal_V))
        self.assertTrue(np.allclose(embeddings.W, expected_normal_W))

        # Scale matters.
        embeddings = h.embeddings.random(
            vocab, d, dictionary=dictionary, distribution='uniform', scale=1,
            seed=0
        )

        np.random.seed(0)
        expected_uniform_scale_V = np.random.uniform(-1, 1, (vocab, d))
        expected_uniform_scale_W = np.random.uniform(-1, 1, (vocab, d))

        self.assertTrue(isinstance(embeddings.V, torch.Tensor))
        self.assertTrue(isinstance(embeddings.W, torch.Tensor))
        self.assertTrue(np.allclose(embeddings.V, expected_uniform_scale_V))
        self.assertTrue(np.allclose(embeddings.W, expected_uniform_scale_W))

        # Scale matters.
        embeddings = h.embeddings.random(
            vocab, d, dictionary=dictionary, distribution='normal', scale=1,
            seed=0
        )

        np.random.seed(0)
        expected_normal_scale_V = np.random.normal(0, 1, (vocab, d))
        expected_normal_scale_W = np.random.normal(0, 1, (vocab, d))

        self.assertTrue(isinstance(embeddings.V, torch.Tensor))
        self.assertTrue(isinstance(embeddings.W, torch.Tensor))
        self.assertTrue(np.allclose(embeddings.V, expected_normal_scale_V))
        self.assertTrue(np.allclose(embeddings.W, expected_normal_scale_W))


    def test_unk(self):

        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()
        device = h.CONSTANTS.MATRIX_DEVICE
        V = torch.rand(vocab, d, device=device)
        v_bias = torch.rand(vocab, device=device)
        W = torch.rand(vocab, d, device=device)
        w_bias = torch.rand(vocab, device=device)

        # Create embeddings with V, W, v_bias, and w_bias
        embeddings = h.embeddings.Embeddings(
            V, W=W, v_bias=v_bias, w_bias=w_bias, dictionary=dictionary)

        self.assertTrue(torch.allclose(embeddings.unk, embeddings.V.mean(0)))
        self.assertTrue(torch.allclose(embeddings.unkV, embeddings.V.mean(0)))
        self.assertTrue(torch.allclose(embeddings.unkW, embeddings.W.mean(0)))
        self.assertTrue(torch.allclose(
            embeddings.unkv_bias, embeddings.v_bias.mean(0)))
        self.assertTrue(torch.allclose(
            embeddings.unkw_bias, embeddings.w_bias.mean(0)))

        with self.assertRaises(KeyError):
            embeddings.get_vec('archaeopteryx')
        with self.assertRaises(KeyError):
            embeddings.get_covec('archaeopteryx')
        with self.assertRaises(KeyError):
            embeddings['archaeopteryx']
        with self.assertRaises(KeyError):
            embeddings.get_vec_bias('archaeopteryx')
        with self.assertRaises(KeyError):
            embeddings.get_covec_bias('archaeopteryx')

        self.assertTrue(torch.allclose(
            embeddings.get_vec('archaeopteryx', 'unk'),
            embeddings.V.mean(0)
        ))
        self.assertTrue(torch.allclose(
            embeddings.get_covec('archaeopteryx', 'unk'),
            embeddings.W.mean(0)
        ))
        self.assertTrue(torch.allclose(
            embeddings.get_vec_bias('archaeopteryx', 'unk'),
            embeddings.v_bias.mean(0)
        ))
        self.assertTrue(torch.allclose(
            embeddings.get_covec_bias('archaeopteryx', 'unk'),
            embeddings.w_bias.mean(0)
        ))


    def test_embedding_access(self):

        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()
        V = np.random.random((vocab, d))
        W = np.random.random((vocab, d))
        v_bias = np.random.random((vocab,))
        w_bias = np.random.random((vocab,))

        embeddings = h.embeddings.Embeddings(
            V, W=W, v_bias=v_bias, w_bias=w_bias, dictionary=dictionary)

        self.assertTrue(np.allclose(embeddings.get_vec(1000), V[1000]))
        self.assertTrue(np.allclose(
            embeddings.get_vec('apple'),
            V[dictionary.tokens.index('apple')]
        ))

        self.assertTrue(np.allclose(embeddings.get_covec(1000), W[1000]))
        self.assertTrue(np.allclose(
            embeddings.get_covec('apple'),
            W[dictionary.tokens.index('apple')]
        ))

        self.assertTrue(np.allclose(embeddings[1000], V[1000]))
        self.assertTrue(np.allclose(
            embeddings['apple'],
            V[dictionary.tokens.index('apple')]
        ))

        self.assertTrue(np.allclose(embeddings.v_bias[1000], v_bias[1000]))
        self.assertTrue(np.allclose(
            embeddings.get_vec_bias('apple'),
            v_bias[dictionary.tokens.index('apple')]
        ))

        self.assertTrue(np.allclose(embeddings.w_bias[1000], w_bias[1000]))
        self.assertTrue(np.allclose(
            embeddings.get_covec_bias('apple'),
            w_bias[dictionary.tokens.index('apple')]
        ))

        # KeyErrors are trigerred when trying to access embeddings that are
        # out-of-vocabulary.
        with self.assertRaises(KeyError):
            embeddings.get_vec('archaeopteryx')

        with self.assertRaises(KeyError):
            embeddings.get_covec('archaeopteryx')

        with self.assertRaises(KeyError):
            embeddings['archaeopteryx']

        with self.assertRaises(KeyError):
            embeddings.get_vec_bias('archaeopteryx')

        with self.assertRaises(KeyError):
            embeddings.get_covec_bias('archaeopteryx')

        # IndexErrors are raised for trying to access non-existent embedding
        # indices
        with self.assertRaises(IndexError):
            embeddings.get_vec(5000)

        with self.assertRaises(IndexError):
            embeddings.get_vec((0,300))

        with self.assertRaises(IndexError):
            embeddings.get_covec(5000)

        with self.assertRaises(IndexError):
            embeddings.get_covec((0,300))

        with self.assertRaises(IndexError):
            embeddings[5000]

        with self.assertRaises(IndexError):
            embeddings[0,300]

        with self.assertRaises(IndexError):
            embeddings.get_vec_bias(5000)

        with self.assertRaises(IndexError):
            embeddings.get_covec_bias(5000)

        embeddings = h.embeddings.Embeddings(V, W, dictionary=None)
        with self.assertRaises(ValueError):
            embeddings['apple']

        embeddings = h.embeddings.Embeddings(V, W=None, dictionary=dictionary)
        self.assertTrue(np.allclose(embeddings.V, V))
        self.assertTrue(embeddings.W is None)
        self.assertTrue(embeddings.dictionary is dictionary)

        self.assertTrue(np.allclose(embeddings.get_vec(1000), V[1000]))
        self.assertTrue(np.allclose(
            embeddings.get_vec('apple'),
            V[dictionary.tokens.index('apple')]
        ))

        self.assertTrue(np.allclose(embeddings[1000], V[1000]))
        self.assertTrue(np.allclose(
            embeddings['apple'],
            V[dictionary.tokens.index('apple')]
        ))

        with self.assertRaises(ValueError):
            embeddings.get_covec(1000)

        with self.assertRaises(ValueError):
            embeddings.get_covec('apple'),


    def test_save_load(self):

        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()
        V = np.random.random((vocab, d))
        W = np.random.random((vocab, d))
        out_path = os.path.join(h.CONSTANTS.TEST_DIR, 'test-embeddings')

        if os.path.exists(out_path):
            shutil.rmtree(out_path)


        # Create vectors using the numpy implementation, save them, then
        # reload them alternately using either numpy or torch implementation.
        embeddings1 = h.embeddings.Embeddings(V, W, dictionary=dictionary)
        embeddings1.save(out_path)

        embeddings2 = h.embeddings.Embeddings.load(out_path)
        self.assertTrue(isinstance(embeddings2.V, torch.Tensor))

        self.assertTrue(embeddings1.V is not embeddings2.V)
        self.assertTrue(embeddings1.W is not embeddings2.W)
        self.assertTrue(torch.allclose(embeddings1.V, embeddings2.V))
        self.assertTrue(torch.allclose(embeddings1.W, embeddings2.W))

        embeddings2 = h.embeddings.Embeddings.load(out_path)
        self.assertTrue(isinstance(embeddings2.V, torch.Tensor))

        self.assertTrue(embeddings1.V is not embeddings2.V)
        self.assertTrue(embeddings1.W is not embeddings2.W)
        self.assertTrue(np.allclose(embeddings1.V, embeddings2.V))
        self.assertTrue(np.allclose(embeddings1.W, embeddings2.W))

        shutil.rmtree(out_path)

        # We can do the same save and load cycle, this time starting from
        # torch embeddings.
        if os.path.exists(out_path):
            shutil.rmtree(out_path)

        embeddings1 = h.embeddings.Embeddings(V, W, dictionary=dictionary)
        embeddings1.save(out_path)

        embeddings2 = h.embeddings.Embeddings.load(out_path)
        self.assertTrue(isinstance(embeddings2.V, torch.Tensor))

        self.assertTrue(embeddings1.V is not embeddings2.V)
        self.assertTrue(embeddings1.W is not embeddings2.W)
        self.assertTrue(torch.allclose(embeddings1.V, embeddings2.V))
        self.assertTrue(torch.allclose(embeddings1.W, embeddings2.W))

        embeddings2 = h.embeddings.Embeddings.load(out_path)
        self.assertTrue(isinstance(embeddings2.V, torch.Tensor))

        self.assertTrue(embeddings1.V is not embeddings2.V)
        self.assertTrue(embeddings1.W is not embeddings2.W)
        self.assertTrue(np.allclose(embeddings1.V, embeddings2.V))
        self.assertTrue(np.allclose(embeddings1.W, embeddings2.W))

        shutil.rmtree(out_path)


    def test_embeddings_recognize_loading_normalized(self):

        in_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'normalized-test-embeddings')
        embeddings = h.embeddings.Embeddings.load(in_path)
        self.assertTrue(embeddings.normed)
        self.assertTrue(embeddings.check_normalized())


    def test_normalize_embeddings(self):

        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()
        V = np.random.random((vocab, d))
        W = np.random.random((vocab, d))

        embeddings = h.embeddings.Embeddings(V, W, dictionary=dictionary)

        self.assertFalse(embeddings.normed)
        self.assertFalse(embeddings.check_normalized())
        self.assertFalse(
            np.allclose(h.utils.norm(embeddings.V, axis=1), 1.0))
        self.assertFalse(
            np.allclose(h.utils.norm(embeddings.W, axis=1), 1.0))

        embeddings.normalize()

        self.assertTrue(embeddings.normed)
        self.assertTrue(embeddings.check_normalized())
        self.assertTrue(
            np.allclose(h.utils.norm(embeddings.V, axis=1), 1.0))
        self.assertTrue(
            np.allclose(h.utils.norm(embeddings.W, axis=1), 1.0))


    def test_greatest_product(self):
        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()

        # Some products are tied, and their sorting isn't stable.  But, when
        # we set fix the seed, the top and bottom ten are stably ranked.
        embeddings = h.embeddings.random(
            vocab, d, dictionary=dictionary, seed=0)

        # Given a query vector, verify that we can find the other vector having
        # the greatest dot product.
        query = embeddings['dog']
        products = embeddings.V @ query
        ranks = sorted(
            [(p, idx) for idx, p in enumerate(products)], reverse=True)
        expected_ranked_tokens = [
            dictionary.get_token(idx) for p, idx in ranks
            if dictionary.get_token(idx) != 'dog'
        ]
        expected_ranked_ids = [
            idx for p, idx in ranks if dictionary.get_token(idx) != 'dog']

        found_ranked_tokens = embeddings.greatest_product('dog')
        self.assertEqual(found_ranked_tokens[:10], expected_ranked_tokens[:10])
        self.assertEqual(
            found_ranked_tokens[-10:], expected_ranked_tokens[-10:])

        # If we provide an id as a query, the matches are returned as ids.
        found_ranked_ids = embeddings.greatest_product(
            dictionary.get_id('dog'))
        self.assertEqual(
            list(found_ranked_ids[:10]), expected_ranked_ids[:10])
        self.assertEqual(
            list(found_ranked_ids[-10:]), expected_ranked_ids[-10:])

        # Verify that we can get the single best match:
        found_best_match = embeddings.greatest_product_one('dog')
        self.assertEqual(found_best_match, expected_ranked_tokens[0])

        # Again, if we provide an id as the query, the best match is returned
        # as an id.
        found_best_match = embeddings.greatest_product_one(
            dictionary.get_id('dog'))
        self.assertEqual(found_best_match, expected_ranked_ids[0])


    def test_greatest_cosine(self):
        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()

        # Some products are tied, and their sorting isn't stable.  But, when
        # we set fix the seed, the top and bottom ten are stably ranked.
        embeddings = h.embeddings.random(
            vocab, d, dictionary=dictionary, seed=0)

        # Given a query vector, verify that we can find the other vector having
        # the greatest dot product.  We want to test cosine similarity, so we
        # should take the dot product of normalized vectors.

        normed_query = h.utils.normalize(embeddings['dog'], axis=0)
        normed_V = h.utils.normalize(embeddings.V, axis=1)
        products = normed_V @ normed_query
        ranks = sorted(
            [(p, idx) for idx, p in enumerate(products)], reverse=True)
        expected_ranked_tokens = [
            dictionary.get_token(idx) for p, idx in ranks
            if dictionary.get_token(idx) != 'dog'
        ]
        expected_ranked_ids = [
            idx for p, idx in ranks if dictionary.get_token(idx) != 'dog']


        found_ranked_tokens = embeddings.greatest_cosine('dog')

        self.assertEqual(found_ranked_tokens[:10], expected_ranked_tokens[:10])
        self.assertEqual(
            found_ranked_tokens[-10:], expected_ranked_tokens[-10:])

        # If we provide an id as a query, the matches are returned as ids.
        found_ranked_ids = embeddings.greatest_cosine(
            dictionary.get_id('dog'))
        self.assertEqual(
            list(found_ranked_ids[:10]), expected_ranked_ids[:10])
        self.assertEqual(
            list(found_ranked_ids[-10:]), expected_ranked_ids[-10:])

        # Verify that we can get the single best match:
        found_best_match = embeddings.greatest_cosine_one('dog')
        self.assertEqual(found_best_match, expected_ranked_tokens[0])

        # Again, if we provide an id as the query, the best match is returned
        # as an id.
        found_best_match = embeddings.greatest_cosine_one(
            dictionary.get_id('dog'))
        self.assertEqual(found_best_match, expected_ranked_ids[0])


    def test_slicing(self):

        dtype=h.CONSTANTS.DEFAULT_DTYPE
        device=h.CONSTANTS.MATRIX_DEVICE
        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()
        V = torch.rand((vocab, d), device=device, dtype=dtype)
        W = torch.rand((vocab, d), device=device, dtype=dtype)

        embeddings = h.embeddings.Embeddings(
            V, W, dictionary=dictionary, device=device)

        self.assertTrue(torch.allclose(embeddings[0:5000:1,0:300:1], V))
        self.assertTrue(torch.allclose(
            embeddings.get_vec((slice(0,5000,1),slice(0,300,1))), V))
        self.assertTrue(torch.allclose(
            embeddings.get_covec((slice(0,5000,1),slice(0,300,1))),
            W
        ))

        V = np.random.random((vocab, d))
        W = np.random.random((vocab, d))

        embeddings = h.embeddings.Embeddings(
            V, W, dictionary=dictionary, device=device)

        self.assertTrue(
            np.allclose(embeddings[0:5000:1,0:300:1], V))
        self.assertTrue(np.allclose(
                embeddings.get_vec((slice(0,5000,1),slice(0,300,1))), V))
        self.assertTrue(np.allclose(
                embeddings.get_covec((slice(0,5000,1),slice(0,300,1))), W))


    def test_sort_like(self):

        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()
        V = np.random.random((vocab, d))
        W = np.random.random((vocab, d))
        v_bias = np.random.random((vocab,))
        w_bias = np.random.random((vocab,))

        embeddings_pristine = h.embeddings.Embeddings(
            V, W=W, v_bias=v_bias, w_bias=w_bias,
            dictionary=get_test_dictionary()
        )
        embeddings_to_be_sorted = h.embeddings.Embeddings(
            V, W=W, v_bias=v_bias, w_bias=w_bias,
            dictionary=get_test_dictionary(), verbose=False
        )
        embeddings_to_sort_by = h.embeddings.Embeddings(
            V, W=W, v_bias=v_bias, w_bias=w_bias, 
            dictionary=get_test_dictionary()
        )
        sort_tokens = embeddings_to_sort_by.dictionary.tokens
        random.shuffle(sort_tokens)

        # Alter some of the embeddings so that we can test the handling of
        # sorting against embeddings that don't have the same set of tokens.
        extraneous_tokens = [
            'archeaopteryx', 'Calabi-Yau', 'snails-pace-maker',
            'xxxxxxx', 'yyyyyyy'
        ]
        ommited_tokens = sort_tokens[5:10]
        sort_tokens[5:10] = extraneous_tokens

        # Sort the embeddings according to a new shuffled token order.
        # Because the tokens don't all match, we will get an error unless we
        # stipulate to allow mismatches.
        with self.assertRaises(ValueError):
            embeddings_to_be_sorted.sort_like(embeddings_to_sort_by)
        embeddings_to_be_sorted.sort_like(
            embeddings_to_sort_by, allow_mismatch=True
        )

        # The number of embeddings is reduced, because we lost ommitted tokens
        # and because extraneous tokens are ignored.
        self.assertEqual(
            embeddings_to_be_sorted.V.shape[0],
            embeddings_pristine.V.shape[0] - 5
        )
        self.assertEqual(
            embeddings_to_be_sorted.W.shape[0],
            embeddings_pristine.W.shape[0] - 5
        )
        self.assertEqual(
            embeddings_to_be_sorted.v_bias.shape[0],
            embeddings_pristine.v_bias.shape[0] - 5
        )
        self.assertEqual(
            embeddings_to_be_sorted.w_bias.shape[0],
            embeddings_pristine.w_bias.shape[0] - 5
        )

        # The embeddings are reordered but still bound to the same tokens.
        for i, token in enumerate(sort_tokens):

            # Extraneous tokens are left out though.
            if i >= 5 and i < 10:
                with self.assertRaises(KeyError):
                    embeddings_to_be_sorted.get_vec(token)
                continue

            # Adjust indices after extraneous tokens, which were dropped
            if i >= 10:
                i = i - 5

            self.assertTrue(torch.allclose(
                embeddings_to_be_sorted.V[i],
                embeddings_to_be_sorted.get_vec(token)
            ))
            self.assertTrue(torch.allclose(
                embeddings_to_be_sorted.W[i],
                embeddings_to_be_sorted.get_covec(token)
            ))
            self.assertTrue(torch.allclose(
                embeddings_pristine.get_vec(token),
                embeddings_to_be_sorted.get_vec(token)
            ))
            self.assertTrue(torch.allclose(
                embeddings_pristine.get_covec(token),
                embeddings_to_be_sorted.get_covec(token)
            ))
            self.assertTrue(torch.allclose(
                embeddings_to_be_sorted.v_bias[i],
                embeddings_to_be_sorted.get_vec_bias(token)
            ))
            self.assertTrue(torch.allclose(
                embeddings_to_be_sorted.w_bias[i],
                embeddings_to_be_sorted.get_covec_bias(token)
            ))

