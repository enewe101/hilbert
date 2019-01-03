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


class TestBigramSector(TestCase):

    def get_test_bigram_sector(self):
        bigram_base = h.bigram.BigramBase.load(
            os.path.join(h.CONSTANTS.TEST_DIR, 'bigram'))

        # BigramBases should generally be made by passing a unigram and Nxx
        sector_factor = 3
        sectors = h.shards.Shards(sector_factor)
        arbitrary_sector = sectors[5]
        args = {
            'unigram':bigram_base.unigram,
            'Nxx':bigram_base.Nxx[arbitrary_sector],
            'Nx':bigram_base.Nx,
            'Nxt':bigram_base.Nxt,
            'sector':arbitrary_sector
        }
        bigram_sector = h.bigram.BigramSector(**args)
        return bigram_sector, bigram_base


    def get_test_cooccurrence_stats(self):
        dictionary = h.dictionary.Dictionary(['banana','socks','car','field'])
        array = np.array([[0,3,1,1],[3,0,1,0],[1,1,0,0],[1,0,0,0]])
        unigram = h.unigram.Unigram(dictionary, array.sum(axis=1))
        return dictionary, array, unigram


    #def get_bigger_test_bigram(self):
    #    read_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram')
    #    write_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram-sectors')
    #    bigram_mutable = h.bigram.BigramMutable.load(read_path)
    #    for i, sector in enumerate(h.shards.Shards(3)):
    #        if i == 0:
    #            bigram_mutable.save_sector(write_path, sector)
    #        else:
    #            bigram_mutable.save_sector(write_path, sector, False, False)


    def test_bigram_sector(self):
        bigram_base = h.bigram.BigramBase.load(
            os.path.join(h.CONSTANTS.TEST_DIR, 'bigram'))

        shard_factor = 3
        for sector in h.shards.Shards(shard_factor):

            bigram_sector = h.bigram.BigramSector(
                bigram_base.unigram, bigram_base.Nxx[sector],
                bigram_base.Nx, bigram_base.Nxt, sector
            )

            # BigramSector's length and shape are correct.
            # They should be equal up to rounding off true division 
            self.assertTrue(approx_equal(
                len(bigram_sector), len(bigram_base)//shard_factor
            ))
            self.assertTrue(approx_equal(
                bigram_sector.shape[0], bigram_base.Nxx[sector].shape[0]))
            self.assertTrue(approx_equal(
                bigram_sector.shape[1], bigram_base.Nxx[sector].shape[1]))
            self.assertEqual(
                len(bigram_sector.shape), len(bigram_base.Nxx[sector].shape))

            # Except for the cooccurrence matrix Nxx, which is in sparse
            # matrix form, the other statistics are `torch.Tensor`s.
            self.assertTrue(isinstance(bigram_sector.Nxx, sparse.lil_matrix))
            self.assertTrue(isinstance(bigram_sector._Nx, torch.Tensor))
            self.assertTrue(isinstance(bigram_sector._Nxt, torch.Tensor))
            self.assertTrue(isinstance(bigram_sector._uNx, torch.Tensor))
            self.assertTrue(isinstance(bigram_sector._uNxt, torch.Tensor))
            self.assertTrue(isinstance(bigram_sector.Nx, torch.Tensor))
            self.assertTrue(isinstance(bigram_sector.Nxt, torch.Tensor))
            self.assertTrue(isinstance(bigram_sector.uNx, torch.Tensor))
            self.assertTrue(isinstance(bigram_sector.uNxt, torch.Tensor))
            self.assertTrue(isinstance(bigram_sector.uN, torch.Tensor))
            self.assertTrue(isinstance(bigram_sector.N, torch.Tensor))

            # BigramSector still posesses the full unigram data, some of which
            # needs to be accessed through an underscored name.
            self.assertEqual(bigram_sector.unigram, bigram_base.unigram)
            self.assertEqual(bigram_sector.dictionary, bigram_base.dictionary)
            self.assertTrue(torch.allclose(bigram_sector._uNx, bigram_base.uNx))
            self.assertTrue(torch.allclose(
                bigram_sector._uNxt, bigram_base.uNxt))

            # When accessing the same data, but without the underscore, we 
            # see only what pertains to the sector.
            self.assertTrue(torch.allclose(
                bigram_sector.uNx, bigram_base.uNx[sector[0]]))
            self.assertTrue(torch.allclose(
                bigram_sector.uNxt, bigram_base.uNxt[:,sector[1]]))
            self.assertTrue(torch.allclose(bigram_sector.uN, bigram_base.uN))

            # BigramSector still possesses the full marginalized count data,
            # but it needs to be accessed using an underscored name.
            self.assertTrue(torch.allclose(bigram_sector._Nx, bigram_base.Nx))
            self.assertTrue(torch.allclose(bigram_sector._Nxt, bigram_base.Nxt))

            # When accessing the same data, but without the underscore, we 
            # see only what pertains to the sector.
            self.assertTrue(np.allclose(
                bigram_sector.Nxx.toarray(), bigram_base.Nxx[sector].toarray()))
            self.assertTrue(torch.allclose(
                bigram_sector.Nx, bigram_base.Nx[sector[0]]))
            self.assertTrue(torch.allclose(
                bigram_sector.Nxt, bigram_base.Nxt[:,sector[1]]))
            self.assertTrue(torch.allclose(
                bigram_sector.N, bigram_base.N))


    def test_invalid_arguments(self):

        bigram_base = h.bigram.BigramBase.load(
            os.path.join(h.CONSTANTS.TEST_DIR, 'bigram'))

        # `BigramBases`s should generally be made by passing a unigram, Nxx
        # Nx, Nxt, and the sector defining the portion of the dataset held.
        sector_factor = 3
        sectors = h.shards.Shards(sector_factor)
        arbitrary_sector = sectors[5]
        args = {
            'unigram':bigram_base.unigram,
            'Nxx':bigram_base.Nxx[arbitrary_sector],
            'Nx':bigram_base.Nx,
            'Nxt':bigram_base.Nxt,
            'sector':arbitrary_sector
        }

        # If any of the arguments is missing, a TypeError is raised
        for key in args:
            temp_args = dict(args)
            del temp_args[key]
            with self.assertRaises(TypeError):
                h.bigram.BigramSector(**temp_args)

        # `BigramSector`s need a sorted unigram instance
        unsorted_unigram = deepcopy(bigram_base.unigram)
        random.shuffle(unsorted_unigram.Nx)
        self.assertFalse(unsorted_unigram.check_sorted())
        with self.assertRaises(ValueError):
            temp_args = dict(args)
            temp_args['unigram'] = unsorted_unigram
            h.bigram.BigramSector(**temp_args)

        # Truncated unigram leads to ValueError
        truncated_unigram = deepcopy(bigram_base.unigram)
        truncated_unigram.Nx = truncated_unigram.Nx[:-1]
        with self.assertRaises(ValueError):
            temp_args = dict(args)
            temp_args['unigram'] = truncated_unigram
            h.bigram.BigramSector(**temp_args)

        # Truncated unigram dictionary leads to ValueError
        truncated_unigram = deepcopy(bigram_base.unigram)
        truncated_unigram.dictionary = h.dictionary.Dictionary(
            bigram_base.unigram.dictionary.tokens[:-1])
        with self.assertRaises(ValueError):
            temp_args = dict(args)
            temp_args['unigram'] = truncated_unigram
            h.bigram.BigramSector(**temp_args)

        # Truncated Nx leads to ValueError
        temp_args = dict(args)
        temp_args['Nx'] = args['Nx'][:-1]
        with self.assertRaises(ValueError):
            h.bigram.BigramSector(**temp_args)

        # Truncated Nxt leads to ValueError
        temp_args = dict(args)
        temp_args['Nxt'] = args['Nxt'][:,:-1]
        with self.assertRaises(ValueError):
            h.bigram.BigramSector(**temp_args)


    def test_load(self):

        bigram_base, unigram, Nxx = h.corpus_stats.get_test_bigram_base()

        sector_factor = 3
        for sector in h.shards.Shards(sector_factor):

            # Build a `BigramSector` from the bigram_base.
            args = {
                'unigram':bigram_base.unigram,
                'Nxx':bigram_base.Nxx[sector],
                'Nx':bigram_base.Nx,
                'Nxt':bigram_base.Nxt,
                'sector':sector
            }
            expected_sector = h.bigram.BigramSector(**args)

            # Load the corresponding sector directly from disk.
            found_sector = h.bigram.BigramSector.load(
                os.path.join(h.CONSTANTS.TEST_DIR, 'bigram-sectors'), sector)

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

        bigram_base = h.bigram.BigramBase.load(
            os.path.join(h.CONSTANTS.TEST_DIR, 'bigram'))

        for sector in h.shards.Shards(3):

            bigram_sector = h.bigram.BigramSector.load(
                os.path.join(h.CONSTANTS.TEST_DIR, 'bigram-sectors'), sector)

            # Test loading bigram data without specifying a shard. Should load
            # the full sector.
            xNxx, xNx, xNxt, xN = bigram_base.load_shard(sector)
            fNxx, fNx, fNxt, fN = bigram_sector.load_shard()
            frNxx, frNx, frNxt, frN = bigram_sector.load_relative_shard()

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
            xuNx, xuNxt, xuN = bigram_base.load_unigram_shard(sector)
            fuNx, fuNxt, fuN = bigram_sector.load_unigram_shard()
            fruNx, fruNxt, fruN = bigram_sector.load_relative_unigram_shard()

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
                xNxx, xNx, xNxt, xN = bigram_base.load_shard(abs_shard)
                fNxx, fNx, fNxt, fN = bigram_sector.load_shard(abs_shard)
                frNxx, frNx, frNxt, frN = bigram_sector.load_relative_shard(
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
                xuNx, xuNxt, xuN = bigram_base.load_unigram_shard(abs_shard)
                fuNx, fuNxt, fuN = bigram_sector.load_unigram_shard(abs_shard)
                fruNx, fruNxt, fruN = bigram_sector.load_relative_unigram_shard(
                    rel_shard)

                self.assertTrue(torch.allclose(fuNx, xuNx))
                self.assertTrue(torch.allclose(fruNx, xuNx))

                self.assertTrue(torch.allclose(fuNxt, xuNxt))
                self.assertTrue(torch.allclose(fruNxt, xuNxt))

                self.assertTrue(torch.allclose(fuN, xuN))
                self.assertTrue(torch.allclose(fruN, xuN))


    def test_merge(self):
        bigram_sector, bigram_base = self.get_test_bigram_sector()

        # Make a similar bigram, but change some of the bigram statistics
        decremented_Nxx = bigram_base.Nxx.toarray() - 1
        decremented_Nxx[decremented_Nxx<0] = 0
        decremented_bigram_sector = h.bigram.BigramSector(
            bigram_base.unigram, 
            decremented_Nxx[bigram_sector.sector],
            np.sum(decremented_Nxx, axis=1, keepdims=True),
            np.sum(decremented_Nxx, axis=0, keepdims=True),
            bigram_sector.sector
        )

        # Merge the two bigram instances
        bigram_sector.merge(decremented_bigram_sector)

        # The merged bigram should have the sum of the individual bigrams'
        # statistics.
        self.assertTrue(np.allclose(
            bigram_sector.Nxx.toarray(), 
            (bigram_base.Nxx.toarray() + decremented_Nxx)[bigram_sector.sector]
        ))


    def test_apply_w2v_undersampling(self):

        t = 1e-5
        bigram_base, unigram, Nxx = h.corpus_stats.get_test_bigram_base()

        sector_factor = 3
        for sector in h.shards.Shards(sector_factor):
            args = {
                'unigram':unigram,
                'Nxx':Nxx[sector],
                'Nx':bigram_base.Nx,
                'Nxt':bigram_base.Nxt,
                'sector':sector
            }
            bigram_sector = h.bigram.BigramSector(**args)

            # Initially the counts reflect the provided cooccurrence matrix
            sNxx, sNx, sNxt, sN = bigram_sector.load_shard()
            suNx, suNxt, suN = bigram_sector.load_unigram_shard()
            Nxx, Nx, Nxt, N = bigram_base.load_shard()
            uNx, uNxt, uN = bigram_base.load_unigram_shard()
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

            #pre_PMI = h.corpus_stats.calc_PMI((Nxx, Nx, Nxt, N))
            #bigram_base.apply_w2v_undersampling(t)
            bigram_sector.apply_w2v_undersampling(t)
            #nNxx, nNx, nNxt, nN = bigram_base.load_shard()
            #post_PMI = h.corpus_stats.calc_PMI((nNxx, nNx, nNxt, nN))
            #diff = torch.sum((pre_PMI - post_PMI) / pre_PMI) / (500*500)

            found_Nxx, found_Nx, found_Nxt, found_N = bigram_sector.load_shard()

            self.assertTrue(torch.allclose(found_Nxx, expected_Nxx[sector]))
            self.assertTrue(torch.allclose(found_Nx, expected_Nx[sector[0]]))
            self.assertTrue(torch.allclose(
                found_Nxt, expected_Nxt[:,sector[1]]))
            self.assertTrue(torch.allclose(found_N, expected_N))


    def test_get_sector(self):
        # This method of `BigramBase` should raise a NotImplementedError in 
        # `BigramSector`.
        bigram_sector, bigram_base = self.get_test_bigram_sector()
        with self.assertRaises(NotImplementedError):
            bigram_sector.get_sector(bigram_sector.sector)


    def test_count(self):
        bigram_base, unigram, Nxx = h.corpus_stats.get_test_bigram_base()
        sector_factor = 3
        num_to_sample = 5
        for sector in h.shards.Shards(sector_factor):
            args = {
                'unigram':unigram,
                'Nxx':Nxx[sector],
                'Nx':bigram_base.Nx,
                'Nxt':bigram_base.Nxt,
                'sector':sector
            }
            bigram_sector = h.bigram.BigramSector(**args)
            row_tokens = random.sample(
                unigram.dictionary.tokens[sector[0]], num_to_sample)
            col_tokens = random.sample(
                unigram.dictionary.tokens[sector[1]], num_to_sample)
            for row_token, col_token in zip(row_tokens, col_tokens):
                self.assertEqual(
                    bigram_base.count(row_token, col_token),
                    bigram_sector.count(row_token, col_token)
                )


    def test_density(self):
        bigram_base, unigram, Nxx = h.corpus_stats.get_test_bigram_base()
        sector_factor = 3
        num_to_sample = 5
        for sector in h.shards.Shards(sector_factor):
            args = {
                'unigram':unigram,
                'Nxx':Nxx[sector],
                'Nx':bigram_base.Nx,
                'Nxt':bigram_base.Nxt,
                'sector':sector
            }
            bigram_sector = h.bigram.BigramSector(**args)
            expected_sector = bigram_base.Nxx[sector]
            size = np.prod(expected_sector.shape)
            for thresh in [0,1,25]:
                num_above_thresh = np.sum(expected_sector > thresh)
                expected_density = num_above_thresh / size
                self.assertEqual(
                    bigram_sector.density(thresh), expected_density)


