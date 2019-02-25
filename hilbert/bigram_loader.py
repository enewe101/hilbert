import time
import hilbert as h
from abc import ABC, abstractmethod
from hilbert.loader import Loader, MultiLoader, BufferedLoader
import numpy as np
import warnings
try:
    import torch
except ImportError:
    torch = None



def get_loader(
    loader_class, loader_base_class, *constructor_args, **constructor_kwargs
):
    class TheLoader(loader_class, loader_base_class):
        pass
    return TheLoader(*constructor_args, **constructor_kwargs)



class BigramLoaderBase():

    def __init__(
        self, bigram_path, sector_factor, shard_factor, num_loaders, 
        t_clean_undersample=None, alpha_unigram_smoothing=None,
        queue_size=1, device=None, verbose=True
    ):

        """
        Base class for more specific loaders `BigramLoader` yields tensors 
        representing shards of text cooccurrence data.  Each shard has unigram
        and bigram data, for words and word-pairs, along with totals.

        bigram data:
            `Nxx`   number of times ith word seen with jth word.
            `Nx`    marginalized (summed) counts: num pairs containing ith word
            `Nxt`   marginalized (summed) counts: num pairs containing jth word
            `N`     total number of pairs.

            Note: marginalized counts aren't equal to frequency of the word,
            one word occurrence means participating in ~2 x window-size number
            of pairs.

        unigram data `(uNx, uNxt, uN)`
            `uNx`   Number of times word i occurs.
            `uNxt`  Number of times word j occurs.
            `uN`    total number of words

            Note: Due to unigram-smoothing (e.g. in w2v), uNxt may not equal
            uNx.  In w2v, one gets smoothed, the other is left unchanged (both
            are needed).

        Subclasses can override `_load`, to more specifically choose what
        bigram / unigram data to load, and what other preparations to do to
        make the shard ready to be fed to the model.
        """
        if num_loaders != sector_factor**2:
            warnings.warn(
                "`num_loaders` must equal `sector_factor**2`, so that each "
                "sector can be assigned to one loader.", DeprecationWarning
            )
        self.bigram_path = bigram_path
        self.sector_factor = sector_factor
        self.shard_factor = shard_factor
        self.t_clean_undersample = t_clean_undersample
        self.alpha_unigram_smoothing = alpha_unigram_smoothing
        self.device = device
        self.loaded_sector = None
        self.bigram_sector = None

        super(BigramLoaderBase, self).__init__(
            num_loaders=num_loaders, queue_size=queue_size, verbose=verbose)


    def _preload_iter(self, loader_id):

        for i, sector_id in enumerate(h.shards.Shards(self.sector_factor)):

            # Each worker should handle a subset of the sectors
            if i % self.num_loaders != loader_id:
                continue

            # If we're doing the same sector as last time, no need to reload it
            # This is the advantage of having 
            # num_workers = num_sectors = sector_factor**2, since a given worker
            # will have a dedicated sector.
            if self.loaded_sector != sector_id:
                self.loaded_sector = sector_id
                # Read the sector of bigram data into memory, and transform
                # distributions as desired.
                self.bigram_sector = h.bigram.BigramSector.load(
                    self.bigram_path, sector_id)
                self.bigram_sector.apply_w2v_undersampling(
                    self.t_clean_undersample)
                self.bigram_sector.apply_unigram_smoothing(
                    self.alpha_unigram_smoothing)

            # Start yielding cRAM-preloaded shards
            for shard_id in h.shards.Shards(self.shard_factor):
                bigram_data = self.bigram_sector.load_relative_shard(
                    shard=shard_id, device='cpu')
                unigram_data = self.bigram_sector.load_relative_unigram_shard(
                    shard=shard_id, device='cpu')
                yield shard_id * sector_id, bigram_data, unigram_data


    def _load(self, preloaded):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        shard_id, bigram_data, unigram_data = preloaded
        bigram_data = tuple(tensor.to(device) for tensor in bigram_data)
        unigram_data = tuple(tensor.to(device) for tensor in unigram_data)
        return shard_id, bigram_data, unigram_data


    def describe(self):
        s = '\tbigram_path = {}\n'.format(self.bigram_path)
        s += '\tsector_factor = {}\n'.format(self.sector_factor)
        s += '\tshard_factor = {}\n'.format(self.shard_factor)
        s += '\tnum_loaders = {}\n'.format(self.num_loaders)
        s += '\tt_clean_undersample = {}\n'.format(self.t_clean_undersample)
        s += '\talpha_unigram_smoothing = {}\n'.format(
            self.alpha_unigram_smoothing)
        s += '\tqueue_size = {}\n'.format(self.queue_size)
        s += '\tdevice = {}\n'.format(self.device)
        s += '\tverbose = {}\n'.format(self.verbose)
        return s


class BigramLoader(BigramLoaderBase, Loader):
    pass


class BigramMultiLoader(BigramLoaderBase, MultiLoader):
    pass

class BigramBufferedLoader(BigramLoaderBase, BufferedLoader):
    pass


class PPMILoader(BigramLoaderBase):

    def _load(self, preloaded):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        shard_id, bigram_data, unigram_data = preloaded
        bigram_data = tuple(tensor.to(device) for tensor in bigram_data)
        M = h.corpus_stats.calc_PMI(bigram_data)
        M = torch.clamp(M, min=0)
        return shard_id, {'M':M}

    def describe(self):
        return 'PPMI Sharder\n' + super(PPMILoader, self).describe()



class GloveLoader(BigramLoaderBase):

    def __init__(
        self, bigram_path, sector_factor, shard_factor, num_loaders,
        X_max=100.0, alpha=0.75, t_clean_undersample=None,
        alpha_unigram_smoothing=None, queue_size=1, device=None, verbose=True
    ):
        self.X_max = float(X_max)
        self.alpha = alpha
        super(GloveLoader, self).__init__(
            bigram_path, sector_factor, shard_factor, num_loaders,
            t_clean_undersample=t_clean_undersample,
            alpha_unigram_smoothing=alpha_unigram_smoothing,
            queue_size=queue_size, device=device, verbose=verbose
        )

    def _load(self, preloaded):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        shard_id, bigram_data, unigram_data = preloaded
        Nxx, Nx, Nxt, N = tuple(tensor.to(device) for tensor in bigram_data)
        weights = (Nxx / self.X_max).pow(self.alpha)
        weights = torch.clamp(weights, max=1.)
        M = torch.log(Nxx)
        Nxx_is_zero = (Nxx==0)
        M[Nxx_is_zero] = 0
        weights[Nxx_is_zero] = 0
        weights = weights * 2
        return shard_id, {'M':M, 'weights':weights}

    def describe(self):
        s =  'GloVe Sharder\n' 
        s += '\tX_max = {}\n'.format(self.X_max)
        s += '\talpha = {}\n'.format(self.alpha)
        s += super(GloveLoader, self).describe()
        return s






# noinspection PyCallingNonCallable
class Word2vecLoader(BigramLoaderBase):

    def __init__(
        self, bigram_path, sector_factor, shard_factor, num_loaders, k=15,
        t_clean_undersample=None, alpha_unigram_smoothing=None, queue_size=1,
        device=None, verbose=True
    ):
        super(Word2vecLoader, self).__init__(
            bigram_path, sector_factor, shard_factor, num_loaders,
            t_clean_undersample=t_clean_undersample,
            alpha_unigram_smoothing=alpha_unigram_smoothing,
            queue_size=queue_size, device=device, verbose=verbose
        )
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        device = device or h.CONSTANTS.MATRIX_DEVICE
        self.k = torch.tensor(k, device=device, dtype=dtype)

    def _load(self, preloaded):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        shard_id, bigram_data, unigram_data = preloaded
        Nxx, Nx, Nxt, N = tuple(tensor.to(device) for tensor in bigram_data)
        uNx, uNxt, uN = tuple(tensor.to(device) for tensor in unigram_data)
        N_neg = self.negative_sample(Nxx, Nx, uNxt, uN, self.k)
        return shard_id, {'Nxx':Nxx, 'N_neg': N_neg}

    @staticmethod
    def negative_sample(Nxx, Nx, uNxt, uN, k):
        return k * (Nx - Nxx) * (uNxt / uN)

    def describe(self):
        s = 'Word2Vec Sharder\n'
        s += '\tk = {}\n'.format(self.k)
        s += super(Word2vecLoader, self).describe()
        return s


class TestLoader(BigramLoaderBase):
    def _load(self, preloaded):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        shard_id, bigram_data, unigram_data = preloaded
        Nxx, Nx, Nxt, N = tuple(tensor.to(device) for tensor in bigram_data)
        Pxx_data = Nxx / N
        Pxx_independent = (Nx / N) * (Nxt / N)
        pmi = torch.log(Pxx_data / (Pxx_independent))
        return shard_id, {
            'pmi': pmi
        }

class DiffLoader(BigramLoaderBase):
    def __init__(
        self,bigram_path, sector_factor, shard_factor, num_loaders, 
        t_clean_undersample=None, alpha_unigram_smoothing=None,
        queue_size=1, device=None, verbose=True, w=5
    ):
        super(DiffLoader, self).__init__(
            bigram_path, sector_factor, shard_factor, num_loaders, 
            t_clean_undersample=t_clean_undersample, alpha_unigram_smoothing=alpha_unigram_smoothing,
            queue_size=queue_size, device=device, verbose=verbose
        )

        self.w = w

    def find_stationary_eig(self, trans_mat):
        print("trying to find eigenvalues")
        eigs = torch.eig(torch.t(trans_mat), True)
        print("found eigenvalues")
        index = 0
        for i, evalue in enumerate(eigs[0]):
            if evalue[0] == 1 and evalue[1] == 0:
                index = i
                break

        pi = eigs[1][:,index] / torch.sum(eigs[1][:,index])
        return pi.view(pi.size()[0], 1)

    def find_stationary_np(self, trans_mat):
        """
        Finding the stationary distribution by solving x - xTrans_mat= 0
        Returns pi as a n x 1 tensor
        """
        t_mat = trans_mat.numpy()

        n = A.shape[0]
        a = np.eye(n) - t_mat
        a = np.vstack((a.T, np.ones(n)))
        b = np.matrix( [0] * n + [1]).T
        stationary = np.linalg.lstsq(a,b)[0]

        return torch.from_numpy(stationary)

    def find_stationary_svd(self, trans_mat):
        """
        Finds the stationary distribution by solving xTrans_mat - x = 0
        With the added constraint that sum(pi) = 1
        Returns pi as a n x 1 tensor
        """

        n = trans_mat.size()[0]
        a = torch.eye(4) - trans_mat
        a = torch.cat((torch.t(a), torch.ones((1,4)))) #Adding the sum(pi) = 1 constraint
        b = torch.cat((torch.zeros((n,1)), torch.ones(1,1)))

        stationary, _ = torch.gels(b,a)

        return stationary

    def _load(self, preloaded):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        shard_id, bigram_data, unigram_data = preloaded
        Nxx, Nx, Nxt, N = tuple(tensor.to(device) for tensor in bigram_data)
        trans_M = Nxx / Nx
        print("found initial trans matrix")
        altered = torch.zeros(trans_M.size(), dtype=h.CONSTANTS.DEFAULT_DTYPE, device=device)
        denom = 0
        for i in range(self.w):
            k = i + 1
            term = (self.w - i)*torch.matrix_power(trans_M, k)
            altered = altered + term
            denom += self.w - i

        altered = altered / denom
        print("found altered")
        pi = torch.matrix_power(altered, 1000)[0]
        pi = pi.view(pi.size()[0],1)
        #pi = self.find_stationary_np(altered)
        torch.svd(altered)
        Pxx_data = torch.mm(altered, pi)
        Pxx_independent = torch.mm(pi, torch.t(pi))

        return shard_id, {
            'Pxx_data' : Pxx_data, 'Pxx_independent' : Pxx_independent,
            'pi': pi, 'altered': altered,
            'Nxx': Nxx, 'Nx': Nx, 'N': N, 'trans_M' : trans_M
        }

class MaxLikelihoodLoader(BigramLoaderBase):
    def _load(self, preloaded):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        shard_id, bigram_data, unigram_data = preloaded
        Nxx, Nx, Nxt, N = tuple(tensor.to(device) for tensor in bigram_data)
        Pxx_data = Nxx / N
        Pxx_independent = (Nx / N) * (Nxt / N)
        return shard_id, {
            'Pxx_data':Pxx_data, 'Pxx_independent':Pxx_independent}

    def describe(self):
        return 'Max Likelihood Sharder\n' + super(
            MaxLikelihoodLoader, self).describe()




class MaxPosteriorLoader(BigramLoaderBase):
    def _load(self, preloaded):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        shard_id, bigram_data, unigram_data = preloaded
        Nxx, Nx, Nxt, N = tuple(tensor.to(device) for tensor in bigram_data)
        Pxx_independent = (Nx / N) * (Nxt / N)
        exp_mean, exp_std =  h.corpus_stats.calc_exp_pmi_stats((Nxx,Nx,Nxt,N))
        alpha, beta = h.corpus_stats.calc_prior_beta_params(
            (Nxx, Nx, Nxt, N), exp_mean, exp_std, Pxx_independent)
        N_posterior = N + alpha + beta - 1
        Pxx_posterior = (Nxx + alpha) / N_posterior
        return shard_id, {
            'N':N, 'N_posterior':N_posterior, 'Pxx_posterior': Pxx_posterior,
            'Pxx_independent': Pxx_independent
        }

    def describe(self):
        return 'Max Posterior Probability Sharder\n' + super(
            MaxPosteriorLoader, self).describe()


class KLLoader(BigramLoaderBase):
    def _load(self, preloaded):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        shard_id, bigram_data, unigram_data = preloaded
        Nxx, Nx, Nxt, N = tuple(tensor.to(device) for tensor in bigram_data)
        Pxx_independent = (Nx / N) * (Nxt / N)
        exp_mean, exp_std =  h.corpus_stats.calc_exp_pmi_stats((Nxx,Nx,Nxt,N))
        alpha, beta = h.corpus_stats.calc_prior_beta_params(
            (Nxx,Nx,Nxt,N), exp_mean, exp_std, Pxx_independent)
        N_posterior = N + alpha + beta - 1
        a = Nxx + alpha
        b = N - Nxx + beta
        digamma_a = torch.digamma(a) - torch.digamma(a+b)
        digamma_b = torch.digamma(b) - torch.digamma(a+b)
        return shard_id, {
                'digamma_a': digamma_a, 'digamma_b': digamma_b, 'N':N,
                'N_posterior':N_posterior, 'Pxx_independent': Pxx_independent, 
        }

    def describe(self):
        return 'KL Sharder\n' + super(KLLoader, self).describe()



