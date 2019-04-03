from unittest import TestCase
from hilbert.categorical import Categorical
import torch
from pdb import set_trace
import time
import os
from contextlib import contextmanager

class TestCategorical(TestCase):

    def test_categorical_sample(self):
        """
        Draw a large number of samples, and calculate the empirical probability
        for each outcome.  It should be close to the probability vector with
        which Categorical was created.
        """
        support_size = int(1e3)
        sample_size = int(1e8)
        probs = torch.rand(support_size, device='cuda')
        probs /= probs.sum()
        cat = Categorical(probs, device='cuda')
        sample = cat.sample((sample_size,))
        found_probs = torch.bincount(sample).float() / sample_size
        self.assertTrue(torch.allclose(found_probs, probs, atol=1e-3))


    def test_categorical_sample_dtype(self):
        """
        Draw a large number of samples, and calculate the empirical probability
        for each outcome.  It should be close to the probability vector with
        which Categorical was created.
        """
        support_size = int(1e3)
        sample_size = int(1e3)
        cuda_probs = torch.rand(support_size, device='cuda')
        cuda_probs /= cuda_probs.sum()
        cpu_probs = cuda_probs.to('cpu')

        cuda_cat = Categorical(cuda_probs)
        cuda_sample = cuda_cat.sample((sample_size,))
        self.assertEqual(cuda_sample.device.type, 'cuda')

        cpu_cat = Categorical(cpu_probs)
        cpu_sample = cpu_cat.sample((sample_size,))
        self.assertEqual(cpu_sample.device.type, 'cpu')

        cuda_cat = Categorical(cpu_probs, device='cuda')
        cuda_sample = cuda_cat.sample((sample_size,))
        self.assertEqual(cuda_sample.device.type, 'cuda')

        cpu_cat = Categorical(cuda_probs, device='cpu')
        cpu_sample = cpu_cat.sample((sample_size,))
        self.assertEqual(cpu_sample.device.type, 'cpu')



    def test_categorical_construction_time(self):
        print('Timing creating sampler with 1 million outcomes')
        # Force synchronous CUDA operation so that CUDA ops can be timed from the
        # cpu.
        with launch_blocking():
            support_size = int(1e6)
            construction_times = int(3)
            probs = torch.rand(support_size, device='cuda')
            probs /= probs.sum()

            start = time.time()
            for i in range(construction_times):
                cat = Categorical(probs, device='cuda')
            this_elapsed = time.time() - start

            print('\tthis implementation took: {}'.format(this_elapsed))

            start = time.time()
            for i in range(construction_times):
                cat = torch.distributions.Categorical(probs)
            torch_elapsed = time.time() - start

            print("\tPyTorch's implementation took: {}".format(torch_elapsed))
            print(
                "\tThis took {} times longer".format(this_elapsed / torch_elapsed))


    def test_categorical_sample_time(self):
        print('\n\nTiming 1 million samples from 1 million outcomes')
        # Force synchronous CUDA operation so that CUDA ops can be timed from the
        # cpu.
        with launch_blocking():
            support_size = int(1e6)
            sample_size = int(1e6)
            sample_times = 3
            probs = torch.rand(support_size, device='cuda')
            probs /= probs.sum()

            cat = Categorical(probs, device='cuda')
            start = time.time()
            for i in range(sample_times):
                sample = cat.sample((sample_size,))

            this_elapsed = time.time() - start

            print('\tthis implementation took: {}'.format(this_elapsed))

            torch_cat = torch.distributions.Categorical(probs)
            start = time.time()
            for i in range(sample_times):
                sample = torch_cat.sample((sample_size,))

            torch_elapsed = time.time() - start

            print("\tPyTorch's implementation took: {}".format(torch_elapsed))
            print(
                "\tPyTorch took {} times longer".format(torch_elapsed/this_elapsed))


@contextmanager
def launch_blocking():
    """
    Forces CUDA operations to run synchronously.
    """
    if 'CUDA_LAUNCH_BLOCKING' not in os.environ:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        restore = None
    elif os.environ['CUDA_LAUNCH_BLOCKING'] != '1':
        restore = os.environ['CUDA_LAUNCH_BLOCKING']

    yield

    if restore is None:
        del os.environ['CUDA_LAUNCH_BLOCKING']
    else:
        os.environ['CUDA_LAUNCH_BLOCKING'] = restore

