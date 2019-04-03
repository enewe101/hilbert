from unittest import TestCase, main
from itertools import product
import hilbert as h
import torch
import scipy
import numpy as np
import os


class TestGenericDatastructs(TestCase):


    def get_test_cooccurrence_stats(self):
        dictionary = h.dictionary.Dictionary([
            'banana','socks','car','field','radio','hamburger'
        ])
        Nxx = np.array([
            [0,3,1,1,0,0],
            [3,0,1,0,1,0],
            [1,1,0,0,0,1],
            [1,0,0,1,0,0],
            [0,1,0,0,0,1],
            [0,0,1,0,1,0]
        ])
        unigram = h.unigram.Unigram(dictionary, Nxx.sum(axis=1))
        return dictionary, Nxx, unigram


    def test_get_Nxx_coo(self):

        # Settings for test
        sector_factor = 3

        # Create sharded bigram data to test the loader
        dictionary, Nxx, unigram = self.get_test_cooccurrence_stats()
        bigram = h.bigram.BigramMutable(unigram, Nxx)

        # Save the bigram data on disk in shards to test the loader
        save_path = os.path.join(h.CONSTANTS.TEST_DIR, 'test-sample-loader')
        sectors = h.shards.Shards(sector_factor)
        bigram.save_sectors(save_path, sectors)

        # Get the loader to re-produce the original matrix
        Nxx_data, I, J, Nx, Nxt = h.generic_datastructs.get_Nxx_coo(
            save_path, sector_factor)
        sparse = scipy.sparse.coo_matrix((
            np.array(Nxx_data), (np.array(I), np.array(J))
        ))
        dense = sparse.toarray()

        # Test equality
        self.assertTrue(np.allclose(dense, Nxx))





if __name__ == '__main__':
    main()

