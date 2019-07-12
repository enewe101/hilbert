import os
import shutil
from unittest import main
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


class TestDependencySampler(TestCase):
    torch.random.manual_seed(0)
    def test_dependency_sampler(self):
        V = torch.tensor([
            [0,0,0,0],
            [1,1,1,1],
            [-1,1,-1,1],
            [2,0,2,0],
            [-1,-1,1,1],
            [-1,-1,-1,-1],
        ], dtype=torch.float32)
        W = torch.tensor([
            [-1,1,-1,1],
            [0.5,0.5,0.5,0.5],
            [-1,-1,1,1],
            [1,1,1,1],
            [-1,-1,-1,-1],
            [2,0,2,0],
        ], dtype=torch.float32)
        sampler = h.loader.DependencySampler(V=V, W=W)
        positives = torch.tensor([
            [[0,1,2,3,0], [3,2,1,0,0]],
            [[1,2,3,0,0], [3,2,0,1,0]],
            [[2,3,4,5,0], [5,4,2,3,0]],
            [[1,2,0,0,0], [2,1,0,0,0]]
        ], dtype=torch.int64)
        mask = torch.tensor([
            [1,1,1,1,0],
            [1,1,1,0,0],
            [1,1,1,1,0],
            [1,1,0,0,0],
        ], dtype=torch.uint8)
      
        sampling_rows = [1,2,3,6,7,11,12,13,16]
        padding_rows = [0,4,5,8,9,10,14,15,17,18,19]

        words = positives[:,0,:]
        covectors = W[words]
        self.assertEqual(covectors.size(), (4,5,4))
        self.assertTrue(torch.equal(covectors[0,0,:], torch.tensor([-1,1,-1,1], dtype=torch.float32)))
        
        vectors = V[words]
        product = torch.bmm(covectors, vectors.transpose(1,2))
        #print(product)
        
        reflected_mask = (mask.unsqueeze(1) * mask.unsqueeze(2)).byte()
        self.assertTrue(torch.equal(reflected_mask[0,:,:], torch.tensor([[1,1,1,1,0],
                                                                         [1,1,1,1,0],
                                                                         [1,1,1,1,0],
                                                                         [1,1,1,1,0],
                                                                         [0,0,0,0,0]], dtype=torch.uint8)))
        identities_idx = (slice(None), range(5), range(5))
        product[identities_idx] = torch.tensor(-float('inf'))
        product[1-reflected_mask] = torch.tensor(-float('inf'))

        self.assertEqual(product.size(), (4,5,5))
        self.assertTrue(torch.equal(product[0,:,:], torch.tensor([[-float('inf'),0,4,-4,-float('inf')],
                                                                  [0,-float('inf'),0,2,-float('inf')],
                                                                  [0,0,-float('inf'),0,-float('inf')],
                                                                  [0,4,0,-float('inf'),-float('inf')],
                                                                  [-float('inf'),-float('inf'),-float('inf'),-float('inf'),-float('inf')]])))

        unnormalized_probs = torch.exp(product)
        unnormalized_probs_2d = unnormalized_probs.view(-1, 5)
        unnormalized_probs_2d[1-mask.reshape(-1),:] = 1
        totals = unnormalized_probs_2d.sum(dim=1, keepdim=True)
        probs_full = unnormalized_probs_2d / totals
 
        #We only care about the probs from non-padding rows
        probs = torch.zeros(len(sampling_rows),5)

        k = 0
        for i in range(20):
            if i in sampling_rows:
                probs[k,:] = probs_full[i,:]
                k += 1
        
        print("\n")
        print(probs)

        counter = torch.zeros(len(sampling_rows),5)
        iterations = 50000

        # draw negative samples
        # validate shape and range of negative samples
        # check that negative samples are distributed as expected

        for i in range(iterations):
            negatives = sampler.sample(positives, mask)
            for j in range(len(padding_rows)):
                row_idx = padding_rows[j]
                selection = negatives[row_idx//5,1,row_idx%5]
                self.assertTrue(torch.equal(selection,h.dependency.PAD))
            for j in range(len(sampling_rows)):
                row_idx = sampling_rows[j]
                selection = int(negatives[row_idx//5,1,row_idx%5])
                selected_row = row_idx - row_idx%5 + selection
                self.assertTrue(selected_row in sampling_rows or selected_row % 5 == 0)
                counter[j,selection] += 1
        
        #print("\n")
        #print(negatives)
        self.assertEqual(negatives.size(),positives.size())

        found_probs = counter / iterations

        print("\n")
        print(found_probs)

        self.assertTrue(torch.allclose(probs,found_probs,atol=1e-02))


if __name__ == '__main__':
    main()
