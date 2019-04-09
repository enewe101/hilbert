import os
import shutil
from unittest import TestCase
from copy import copy, deepcopy
import hilbert as h
import random

try:
    import numpy as np
    import torch
    from scipy import sparse
except ImportError:
    np = None
    torch = None
    sparse = None


def approx_equal(a, b):
    """ True if a and b differ by at most 1"""
    return abs(a - b) <= 1


class TestCooccurrenceSector(TestCase):

    def get_test_cooccurrence_sector(self):
        cooccurrence = h.cooccurrence.Cooccurrence.load(
            os.path.join(h.CONSTANTS.TEST_DIR, 'cooccurrence'))

        # Cooccurrences should generally be made by passing a unigram and Nxx
        sector_factor = 3
        sectors = h.shards.Shards(sector_factor)
        arbitrary_sector = sectors[5]
        args = {
            'unigram':cooccurrence.unigram,
            'Nxx':cooccurrence.Nxx[arbitrary_sector],
            'Nx':cooccurrence.Nx,
            'Nxt':cooccurrence.Nxt,
            'sector':arbitrary_sector
        }
        cooccurrence_sector = h.cooccurrence.CooccurrenceSector(**args)
        return cooccurrence_sector, cooccurrence


    def get_test_cooccurrence_stats(self):
        dictionary = h.dictionary.Dictionary(['banana','socks','car','field'])
        array = np.array([[0,3,1,1],[3,0,1,0],[1,1,0,0],[1,0,0,0]])
        unigram = h.unigram.Unigram(dictionary, array.sum(axis=1))
        return dictionary, array, unigram


    #def get_bigger_test_cooccurrence(self):
    #    read_path = os.path.join(h.CONSTANTS.TEST_DIR, 'cooccurrence')
    #    write_path = os.path.join(h.CONSTANTS.TEST_DIR, 'cooccurrence-sectors')
    #    cooccurrence_mutable = h.cooccurrence.CooccurrenceMutable.load(read_path)
    #    for i, sector in enumerate(h.shards.Shards(3)):
    #        if i == 0:
    #            cooccurrence_mutable.save_sector(write_path, sector)
    #        else:
    #            cooccurrence_mutable.save_sector(write_path, sector, False, False)


    def test_cooccurrence_sector(self):
        cooccurrence = h.cooccurrence.Cooccurrence.load(
            os.path.join(h.CONSTANTS.TEST_DIR, 'cooccurrence'))

        shard_factor = 3
        for sector in h.shards.Shards(shard_factor):

            cooccurrence_sector = h.cooccurrence.CooccurrenceSector(
                cooccurrence.unigram, cooccurrence.Nxx[sector],
                cooccurrence.Nx, cooccurrence.Nxt, sector
            )

            # CooccurrenceSector's length and shape are correct.
            # They should be equal up to rounding off true division 
            #self.assertTrue(approx_equal(
            #    len(cooccurrence_sector), len(cooccurrence)//shard_factor
            #))
            self.assertTrue(approx_equal(
                cooccurrence_sector.shape[0], cooccurrence.Nxx[sector].shape[0]))
            self.assertTrue(approx_equal(
                cooccurrence_sector.shape[1], cooccurrence.Nxx[sector].shape[1]))
            self.assertEqual(
                len(cooccurrence_sector.shape), len(cooccurrence.Nxx[sector].shape))

            # Except for the cooccurrence matrix Nxx, which is in sparse
            # matrix form, the other statistics are `torch.Tensor`s.
            self.assertTrue(isinstance(cooccurrence_sector.Nxx, sparse.lil_matrix))
            self.assertTrue(isinstance(cooccurrence_sector._Nx, torch.Tensor))
            self.assertTrue(isinstance(cooccurrence_sector._Nxt, torch.Tensor))
            self.assertTrue(isinstance(cooccurrence_sector._uNx, torch.Tensor))
            self.assertTrue(isinstance(cooccurrence_sector._uNxt, torch.Tensor))
            self.assertTrue(isinstance(cooccurrence_sector.Nx, torch.Tensor))
            self.assertTrue(isinstance(cooccurrence_sector.Nxt, torch.Tensor))
            self.assertTrue(isinstance(cooccurrence_sector.uNx, torch.Tensor))
            self.assertTrue(isinstance(cooccurrence_sector.uNxt, torch.Tensor))
            self.assertTrue(isinstance(cooccurrence_sector.uN, torch.Tensor))
            self.assertTrue(isinstance(cooccurrence_sector.N, torch.Tensor))

            # CooccurrenceSector still posesses the full unigram data, some of which
            # needs to be accessed through an underscored name.
            self.assertEqual(cooccurrence_sector.unigram, cooccurrence.unigram)
            self.assertEqual(cooccurrence_sector.dictionary, cooccurrence.dictionary)
            self.assertTrue(torch.allclose(cooccurrence_sector._uNx, cooccurrence.uNx))
            self.assertTrue(torch.allclose(
                cooccurrence_sector._uNxt, cooccurrence.uNxt))

            # When accessing the same data, but without the underscore, we 
            # see only what pertains to the sector.
            self.assertTrue(torch.allclose(
                cooccurrence_sector.uNx, cooccurrence.uNx[sector[0]]))
            self.assertTrue(torch.allclose(
                cooccurrence_sector.uNxt, cooccurrence.uNxt[:,sector[1]]))
            self.assertTrue(torch.allclose(cooccurrence_sector.uN, cooccurrence.uN))

            # CooccurrenceSector still possesses the full marginalized count data,
            # but it needs to be accessed using an underscored name.
            self.assertTrue(torch.allclose(cooccurrence_sector._Nx, cooccurrence.Nx))
            self.assertTrue(torch.allclose(cooccurrence_sector._Nxt, cooccurrence.Nxt))

            # When accessing the same data, but without the underscore, we 
            # see only what pertains to the sector.
            self.assertTrue(np.allclose(
                cooccurrence_sector.Nxx.toarray(), cooccurrence.Nxx[sector].toarray()))
            self.assertTrue(torch.allclose(
                cooccurrence_sector.Nx, cooccurrence.Nx[sector[0]]))
            self.assertTrue(torch.allclose(
                cooccurrence_sector.Nxt, cooccurrence.Nxt[:,sector[1]]))
            self.assertTrue(torch.allclose(
                cooccurrence_sector.N, cooccurrence.N))


    def test_invalid_arguments(self):

        cooccurrence = h.cooccurrence.Cooccurrence.load(
            os.path.join(h.CONSTANTS.TEST_DIR, 'cooccurrence'))

        # `Cooccurrence`s should generally be made by passing a unigram, Nxx
        # Nx, Nxt, and the sector defining the portion of the dataset held.
        sector_factor = 3
        sectors = h.shards.Shards(sector_factor)
        arbitrary_sector = sectors[5]
        args = {
            'unigram':cooccurrence.unigram,
            'Nxx':cooccurrence.Nxx[arbitrary_sector],
            'Nx':cooccurrence.Nx,
            'Nxt':cooccurrence.Nxt,
            'sector':arbitrary_sector
        }

        # If any of the arguments is missing, a TypeError is raised
        for key in args:
            temp_args = dict(args)
            del temp_args[key]
            with self.assertRaises(TypeError):
                h.cooccurrence.CooccurrenceSector(**temp_args)

        # `CooccurrenceSector`s need a sorted unigram instance
        unsorted_unigram = deepcopy(cooccurrence.unigram)
        random.shuffle(unsorted_unigram.Nx)
        self.assertFalse(unsorted_unigram.check_sorted())
        with self.assertRaises(ValueError):
            temp_args = dict(args)
            temp_args['unigram'] = unsorted_unigram
            h.cooccurrence.CooccurrenceSector(**temp_args)

        # Truncated unigram leads to ValueError
        truncated_unigram = deepcopy(cooccurrence.unigram)
        truncated_unigram.Nx = truncated_unigram.Nx[:-1]
        with self.assertRaises(ValueError):
            temp_args = dict(args)
            temp_args['unigram'] = truncated_unigram
            h.cooccurrence.CooccurrenceSector(**temp_args)

        # Truncated unigram dictionary leads to ValueError
        truncated_unigram = deepcopy(cooccurrence.unigram)
        truncated_unigram.dictionary = h.dictionary.Dictionary(
            cooccurrence.unigram.dictionary.tokens[:-1])
        with self.assertRaises(ValueError):
            temp_args = dict(args)
            temp_args['unigram'] = truncated_unigram
            h.cooccurrence.CooccurrenceSector(**temp_args)

        # Truncated Nx leads to ValueError
        temp_args = dict(args)
        temp_args['Nx'] = args['Nx'][:-1]
        with self.assertRaises(ValueError):
            h.cooccurrence.CooccurrenceSector(**temp_args)

        # Truncated Nxt leads to ValueError
        temp_args = dict(args)
        temp_args['Nxt'] = args['Nxt'][:,:-1]
        with self.assertRaises(ValueError):
            h.cooccurrence.CooccurrenceSector(**temp_args)


    def test_load(self):

        cooccurrence, unigram, Nxx = h.corpus_stats.get_test_cooccurrence()

        sector_factor = 3
        for sector in h.shards.Shards(sector_factor):

            # Build a `CooccurrenceSector` from the cooccurrence.
            args = {
                'unigram':cooccurrence.unigram,
                'Nxx':cooccurrence.Nxx[sector],
                'Nx':cooccurrence.Nx,
                'Nxt':cooccurrence.Nxt,
                'sector':sector
            }
            expected_sector = h.cooccurrence.CooccurrenceSector(**args)

            # Load the corresponding sector directly from disk.
            found_sector = h.cooccurrence.CooccurrenceSector.load(
                os.path.join(h.CONSTANTS.TEST_DIR, 'cooccurrence-sectors'), sector)

            # Cooccurrence arrays should be equal
            self.assertTrue(np.allclose(
                found_sector.Nxx.toarray(), expected_sector.Nxx.toarray()))

            # Marginal stats should be equal, viewed through the sector or not.
            self.assertTrue(torch.allclose(
                found_sector._Nx, expected_sector._Nx))
            self.assertTrue(torch.allclose(
                found_sector._Nxt, expected_sector._Nxt))
            self.assertTrue(torch.allclose(
                found_sector.Nx, expected_sector.Nx))
            self.assertTrue(torch.allclose(
                found_sector.Nxt, expected_sector.Nxt))
            self.assertTrue(torch.allclose(found_sector.N, expected_sector.N))

            # Unigram stats should be equal, viewed through the sector or not.
            self.assertTrue(torch.allclose(
                found_sector._uNx, expected_sector._uNx))
            self.assertTrue(torch.allclose(
                found_sector._uNxt, expected_sector._uNxt))
            self.assertTrue(torch.allclose(
                found_sector.uNx, expected_sector.uNx))
            self.assertTrue(torch.allclose(
                found_sector.uNxt, expected_sector.uNxt))
            self.assertTrue(torch.allclose(found_sector.uN, expected_sector.uN))

            # Dictionaries should be equal
            self.assertEqual(
                found_sector.dictionary.token_ids, 
                expected_sector.dictionary.token_ids, 
            )
            self.assertEqual(
                found_sector.row_dictionary.token_ids, 
                expected_sector.row_dictionary.token_ids, 
            )
            self.assertEqual(
                found_sector.column_dictionary.token_ids, 
                expected_sector.column_dictionary.token_ids, 
            )


    def test_load_shard(self):

        device = h.CONSTANTS.MATRIX_DEVICE
        dtype = h.CONSTANTS.DEFAULT_DTYPE

        cooccurrence = h.cooccurrence.Cooccurrence.load(
            os.path.join(h.CONSTANTS.TEST_DIR, 'cooccurrence'))

        for sector in h.shards.Shards(3):

            cooccurrence_sector = h.cooccurrence.CooccurrenceSector.load(
                os.path.join(h.CONSTANTS.TEST_DIR, 'cooccurrence-sectors'), sector)

            # Test loading cooccurrence data without specifying a shard. Should load
            # the full sector.
            xNxx, xNx, xNxt, xN = cooccurrence.load_shard(sector)
            fNxx, fNx, fNxt, fN = cooccurrence_sector.load_shard()
            frNxx, frNx, frNxt, frN = cooccurrence_sector.load_relative_shard()

            self.assertTrue(torch.allclose(fNxx, xNxx))
            self.assertTrue(torch.allclose(frNxx, xNxx))

            self.assertTrue(torch.allclose(fNx, xNx))
            self.assertTrue(torch.allclose(frNx, xNx))

            self.assertTrue(torch.allclose(fNxt, xNxt))
            self.assertTrue(torch.allclose(frNxt, xNxt))

            self.assertTrue(torch.allclose(fN, xN))
            self.assertTrue(torch.allclose(frN, xN))

            # Test loading unigram data without specifying a shard. Should load
            # the full sector.
            xuNx, xuNxt, xuN = cooccurrence.load_unigram_shard(sector)
            fuNx, fuNxt, fuN = cooccurrence_sector.load_unigram_shard()
            fruNx, fruNxt, fruN = cooccurrence_sector.load_relative_unigram_shard()

            self.assertTrue(torch.allclose(fuNx, xuNx))
            self.assertTrue(torch.allclose(fruNx, xuNx))

            self.assertTrue(torch.allclose(fuNxt, xuNxt))
            self.assertTrue(torch.allclose(fruNxt, xuNxt))

            self.assertTrue(torch.allclose(fuN, xuN))
            self.assertTrue(torch.allclose(fruN, xuN))

            # Test loading a few shards, both relatively and aboslutely.
            for rel_shard in h.shards.Shards(4):

                abs_shard = rel_shard * sector

                # Test that absolute and relative loading from a sector matches
                # the absolute loading from the full dataset
                xNxx, xNx, xNxt, xN = cooccurrence.load_shard(abs_shard)
                fNxx, fNx, fNxt, fN = cooccurrence_sector.load_shard(abs_shard)
                frNxx, frNx, frNxt, frN = cooccurrence_sector.load_relative_shard(
                    rel_shard)

                self.assertTrue(torch.allclose(fNxx, xNxx))
                self.assertTrue(torch.allclose(frNxx, xNxx))

                self.assertTrue(torch.allclose(fNx, xNx))
                self.assertTrue(torch.allclose(frNx, xNx))

                self.assertTrue(torch.allclose(fNxt, xNxt))
                self.assertTrue(torch.allclose(frNxt, xNxt))

                self.assertTrue(torch.allclose(fN, xN))
                self.assertTrue(torch.allclose(frN, xN))

                # Test that *unigram* absolute and relative loading from a 
                # sector matches the absolute loading from the full dataset
                xuNx, xuNxt, xuN = cooccurrence.load_unigram_shard(abs_shard)
                fuNx, fuNxt, fuN = cooccurrence_sector.load_unigram_shard(abs_shard)
                fruNx, fruNxt, fruN = cooccurrence_sector.load_relative_unigram_shard(
                    rel_shard)

                self.assertTrue(torch.allclose(fuNx, xuNx))
                self.assertTrue(torch.allclose(fruNx, xuNx))

                self.assertTrue(torch.allclose(fuNxt, xuNxt))
                self.assertTrue(torch.allclose(fruNxt, xuNxt))

                self.assertTrue(torch.allclose(fuN, xuN))
                self.assertTrue(torch.allclose(fruN, xuN))


    def test_merge(self):
        cooccurrence_sector, cooccurrence = self.get_test_cooccurrence_sector()

        # Make a similar cooccurrence, but change some of the cooccurrence statistics
        decremented_Nxx = cooccurrence.Nxx.toarray() - 1
        decremented_Nxx[decremented_Nxx<0] = 0
        decremented_cooccurrence_sector = h.cooccurrence.CooccurrenceSector(
            cooccurrence.unigram, 
            decremented_Nxx[cooccurrence_sector.sector],
            np.sum(decremented_Nxx, axis=1, keepdims=True),
            np.sum(decremented_Nxx, axis=0, keepdims=True),
            cooccurrence_sector.sector
        )

        # Merge the two cooccurrence instances
        cooccurrence_sector.merge(decremented_cooccurrence_sector)

        # The merged cooccurrence should have the sum of the individual cooccurrences'
        # statistics.
        self.assertTrue(np.allclose(
            cooccurrence_sector.Nxx.toarray(), 
            (cooccurrence.Nxx.toarray() + decremented_Nxx)[cooccurrence_sector.sector]
        ))


    def test_apply_unigram_smoothing(self):
        alpha = 0.6
        cooccurrence_sector, cooccurrence = self.get_test_cooccurrence_sector()

        expected_uNx = cooccurrence_sector._uNx**alpha
        expected_uNxt = cooccurrence_sector._uNxt**alpha
        expected_uN = torch.sum(expected_uNx)

        cooccurrence_sector.apply_unigram_smoothing(alpha)
        self.assertTrue(torch.allclose(expected_uNx, cooccurrence_sector._uNx))
        self.assertTrue(torch.allclose(expected_uNxt, cooccurrence_sector._uNxt))
        self.assertTrue(torch.allclose(expected_uN, cooccurrence_sector.uN))


    def test_apply_w2v_undersampling(self):

        t = 1e-5
        cooccurrence, unigram, _Nxx = h.corpus_stats.get_test_cooccurrence()

        sector_factor = 3
        for sector in h.shards.Shards(sector_factor):
            args = {
                'unigram':unigram,
                'Nxx':_Nxx[sector],
                'Nx':cooccurrence.Nx,
                'Nxt':cooccurrence.Nxt,
                'sector':sector
            }

            cooccurrence_sector = h.cooccurrence.CooccurrenceSector(**args)

            # Initially the counts reflect the provided cooccurrence matrix
            sNxx, sNx, sNxt, sN = cooccurrence_sector.load_shard()
            suNx, suNxt, suN = cooccurrence_sector.load_unigram_shard()
            Nxx, Nx, Nxt, N = cooccurrence.load_shard()
            uNx, uNxt, uN = cooccurrence.load_unigram_shard()
            self.assertTrue(np.allclose(Nxx[sector], sNxx))
            self.assertTrue(np.allclose(Nx[sector[0]], sNx))
            self.assertTrue(np.allclose(Nxt[:,sector[1]], sNxt))

            # Now apply undersampling
            p_i = h.corpus_stats.w2v_prob_keep(uNx, uN, t)
            p_j = h.corpus_stats.w2v_prob_keep(uNxt, uN, t)
            expected_Nx = Nx * p_i * torch.sum(uNxt/uN * p_j)
            expected_Nxt = Nxt * p_j * torch.sum(uNx/uN*p_i)
            expected_Nxx = Nxx * p_i * p_j
            expected_N = torch.sum(expected_Nx)

            cooccurrence_sector.apply_w2v_undersampling(t)
            found_Nxx, found_Nx, found_Nxt, found_N = cooccurrence_sector.load_shard()

            self.assertTrue(torch.allclose(found_Nxx, expected_Nxx[sector]))
            self.assertTrue(torch.allclose(found_Nx, expected_Nx[sector[0]]))
            self.assertTrue(torch.allclose(
                found_Nxt, expected_Nxt[:,sector[1]]))
            self.assertTrue(torch.allclose(found_N, expected_N))

            # attempting to call apply_undersampling twice is an error
            with self.assertRaises(ValueError):
                cooccurrence_sector.apply_w2v_undersampling(t)

        # Attempting to call apply_undersampling when in posession of a 
        # smoothed unigram would produce incorrect results, and is an error.
        cooccurrence_sector, cooccurrence = self.get_test_cooccurrence_sector()
        alpha = 0.6
        cooccurrence_sector.unigram.apply_smoothing(alpha)
        with self.assertRaises(ValueError):
            cooccurrence_sector.apply_w2v_undersampling(t)



    def test_get_sector(self):
        # This method of `Cooccurrence` should raise a NotImplementedError in 
        # `CooccurrenceSector`.
        cooccurrence_sector, cooccurrence = self.get_test_cooccurrence_sector()
        with self.assertRaises(NotImplementedError):
            cooccurrence_sector.get_sector(cooccurrence_sector.sector)


    def test_count(self):
        cooccurrence, unigram, Nxx = h.corpus_stats.get_test_cooccurrence()
        sector_factor = 3
        num_to_sample = 5
        for sector in h.shards.Shards(sector_factor):
            args = {
                'unigram':unigram,
                'Nxx':Nxx[sector],
                'Nx':cooccurrence.Nx,
                'Nxt':cooccurrence.Nxt,
                'sector':sector
            }
            cooccurrence_sector = h.cooccurrence.CooccurrenceSector(**args)
            row_tokens = random.sample(
                unigram.dictionary.tokens[sector[0]], num_to_sample)
            col_tokens = random.sample(
                unigram.dictionary.tokens[sector[1]], num_to_sample)
            for row_token, col_token in zip(row_tokens, col_tokens):
                self.assertEqual(
                    cooccurrence.count(row_token, col_token),
                    cooccurrence_sector.count(row_token, col_token)
                )


#    def test_density(self):
#        cooccurrence, unigram, Nxx = h.corpus_stats.get_test_cooccurrence()
#        sector_factor = 3
#        num_to_sample = 5
#        for sector in h.shards.Shards(sector_factor):
#            args = {
#                'unigram':unigram,
#                'Nxx':Nxx[sector],
#                'Nx':cooccurrence.Nx,
#                'Nxt':cooccurrence.Nxt,
#                'sector':sector
#            }
#            cooccurrence_sector = h.cooccurrence.CooccurrenceSector(**args)
#            expected_sector = cooccurrence.Nxx[sector]
#            size = np.prod(expected_sector.shape)
#            for thresh in [0,1,25]:
#                num_above_thresh = np.sum(expected_sector > thresh)
#                expected_density = num_above_thresh / size
#                self.assertEqual(
#                    cooccurrence_sector.density(thresh), expected_density)


