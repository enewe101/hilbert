import time
import torch
import numpy as np
from hilbert.cooccurrence.preloader import ZedSampler
from unittest import TestCase, main

VERBOSE = False

def vprint(*args):
    if VERBOSE:
        print(*args)

class TestZedSampler(TestCase):

    def test_functionality(self):
        device = torch.device('cpu')
        limit = 500000 # vocab size

        # make the big boy
        sampler = ZedSampler(limit, device, max_z_samples=100000)

        # list of samples with Nij > 0
        alphas = [
            torch.randint(0, limit, (100,)).sort()[0],
            torch.randint(0, limit, (1000,)).sort()[0],
            torch.randint(0, limit, (10000,)).sort()[0],
            # torch.randint(0, limit, (100000,)).sort()[0], 
                # ~2 seconds per draw
        ]

        # need to remove repeats, and put it into its desired form
        for i in range(len(alphas)):
            alpha_samples_list = sorted(list(set(alphas[i].numpy())))
            alphas[i] = torch.LongTensor(alpha_samples_list).to(device)

        # lets time the runs
        timers = [[] for _ in range(len(alphas))]

        # run each one 50 times to get good time estimates and ensure
        # functionality
        for _ in range(10):
            for i, a_samples in enumerate(alphas):

                # now sample
                timers[i].append(time.clock())
                z_samples, z_ijs = sampler.z_sample(a_samples)
                timers[i][-1] = time.clock() - timers[i][-1]

                # now test them
                self.assertLessEqual(len(z_samples), len(a_samples))
                aset = set(a_samples.numpy())
                zset = set(z_samples.numpy())
                self.assertEqual(len(aset.intersection(zset)), 0)

        vprint('Time averages:')
        for i, times in enumerate(timers):
            mean = np.mean(times)
            vprint('\talpha={}: {:4f}'.format(i, mean))



if __name__ == '__main__':
    main()
