import hilbert as h
import torch
import numpy as np
from scipy import sparse
from queue import Empty
import time


# base abstract class for the other sharders
class MSharder(object):

    criterion_class = None

    def __init__(
        self, bigram, update_density=1, mask_diagonal=False, device=None
    ):
        self.bigram = bigram
        self.device = device or h.CONSTANTS.MATRIX_DEVICE
        self.update_density = update_density
        self.mask_diagonal = mask_diagonal
        self._get_criterion()

        # False indicates last_shard has not been set.  Note `None` may be
        # interpreted as a valid shard.
        self.last_shard = False 


    def _get_criterion(self):
        if self.criterion_class is None:
            raise NotImplementedError(
                "Subclasses must provide an instance of `hilbert.HilbertLoss` "
                "as `self.criterion_class`."
            )
        self.criterion = self.criterion_class(
            self.update_density, np.prod(self.bigram.Nxx.shape),
            self.mask_diagonal
        )


    def calc_shard_loss(self, M_hat, shard):
        if shard != self.last_shard:
            self.last_shard = shard
            self._load_shard(shard)

        return self._get_loss(M_hat, shard)


    def describe(self):
        raise NotImplementedError('Subclasses must override `describe`.')


    def _load_shard(self,  shard):
        raise NotImplementedError('Subclasses must override `_load_shard`.')


    def _get_loss(self, M_hat, shard):
        raise NotImplementedError('Subclasses must override `_get_loss`.')



class PPMISharder(MSharder):

    criterion_class = h.hilbert_loss.MSELoss

    def describe(self):
        s = 'PPMI Sharder\n'
        s += '\tupdate_density = {}\n'.format(self.update_density)
        s += '\tmask_diagonal = {}\n'.format(self.mask_diagonal)
        return s


    def _load_shard(self, shard):
        Nxx, Nx, Nxt, N = self.bigram.load_shard(shard, device=self.device)
        self.M = h.corpus_stats.calc_PMI((Nxx, Nx, Nxt, N))
        self.M = torch.clamp(self.M, min=0)


    def _get_loss(self, M_hat, shard):
        return self.criterion(M_hat, shard, self.M)



class GloveSharder(MSharder):

    criterion_class = h.hilbert_loss.MSELoss

    def __init__(
        self, bigram, X_max=100.0, alpha=0.75,
        update_density=1, mask_diagonal=False, device=None,
    ):
        super(GloveSharder, self).__init__(
            bigram, update_density, mask_diagonal, device)
        self.X_max = float(X_max)
        self.alpha = alpha


    def describe(self):
        s = 'GloVe Sharder\n'
        s += '\tupdate_density = {}\n'.format(self.update_density)
        s += '\tX_max = {}\n'.format(self.X_max)
        s += '\talpha = {}\n'.format(self.alpha)
        s += '\tmask_diagonal = {}\n'.format(self.mask_diagonal)
        return s


    def _load_shard(self, shard):
        Nxx, Nx, Nxt, N = self.bigram.load_shard(shard, device=self.device)
        self.multiplier = (Nxx / self.X_max).pow(self.alpha)
        self.multiplier = torch.clamp(self.multiplier, max=1.)
        self.M = torch.log(Nxx)

        # Zero-out cells that have Nxx==0, since GloVe ignores these
        self.M[Nxx==0] = 0
        self.multiplier[Nxx==0] = 0


    def _get_loss(self, M_hat, shard):
        return 2 * self.criterion(M_hat, shard, self.M, weights=self.multiplier)



# sharder for W2V
# noinspection PyCallingNonCallable
class Word2vecSharder(MSharder):

    criterion_class = h.hilbert_loss.W2VLoss

    def __init__(
        self, bigram, k=15, update_density=1, mask_diagonal=False, device=None
    ):
        super(Word2vecSharder, self).__init__(
            bigram, update_density, mask_diagonal, device)
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        self.k = torch.tensor(k, device=self.device, dtype=dtype)


    def describe(self):
        s = 'Word2Vec Sharder\n'
        s += '\tupdate_density = {}\n'.format(self.update_density)
        s += '\tk = {}\n'.format(self.k)
        s += '\tmask_diagonal = {}\n'.format(self.mask_diagonal)
        return s


    def _load_shard(self, shard):
        self.Nxx, Nx, _, _ = self.bigram.load_shard(shard, device=self.device)
        _, uNxt, uN = self.bigram.unigram.load_shard(shard,device=self.device)
        self.N_neg = self.negative_sample(self.Nxx, Nx, uNxt, uN, self.k)


    def _get_loss(self, M_hat, shard):
        return self.criterion(M_hat, shard, self.Nxx, self.N_neg)

    @staticmethod
    def negative_sample(Nxx, Nx, uNxt, uN, k):
        return k * (Nx - Nxx) * (uNxt / uN)


class MaxLikelihoodSharder(MSharder):

    criterion_class = h.hilbert_loss.MaxLikelihoodLoss

    def __init__(
        self, bigram, temperature=1, update_density=1, mask_diagonal=False,
        device=None
    ):
        super(MaxLikelihoodSharder, self).__init__(
            bigram, update_density, mask_diagonal, device)
        self.temperature = temperature


    def _load_shard(self, shard):
        Nxx, Nx, Nxt, N = self.bigram.load_shard(shard, device=self.device)
        self.Pxx_data = Nxx / N
        self.Pxx_independent = (Nx / N) * (Nxt / N)


    def _get_loss(self, M_hat, shard):
        return self.criterion(
            M_hat, shard, self.Pxx_data, self.Pxx_independent, self.temperature)


    def describe(self):
        s = 'Max Likelihood Sharder\n'
        s += '\tupdate_density = {}\n'.format(self.update_density)
        s += '\ttemperature = {}\n'.format(self.temperature)
        s += '\tmask_diagonal = {}\n'.format(self.mask_diagonal)
        return s



class MaxPosteriorSharder(MSharder):

    criterion_class = h.hilbert_loss.MaxPosteriorLoss

    def __init__(
        self, bigram, temperature=1, update_density=1, mask_diagonal=False,
        device=None
    ):
        super(MaxPosteriorSharder, self).__init__(
            bigram, update_density, mask_diagonal, device)
        self.temperature = temperature


    def _load_shard(self, shard):

        self.bigram_shard = self.bigram.load_shard(shard, device=self.device)
        Nxx, Nx, Nxt, self.N = self.bigram_shard
        self.Pxx_independent = (Nx / self.N) * (Nxt / self.N)
        exp_mean, exp_std =  h.corpus_stats.calc_exp_pmi_stats(
            self.bigram_shard)
        alpha, beta = h.corpus_stats.calc_prior_beta_params(
            self.bigram_shard, exp_mean, exp_std, self.Pxx_independent)
        self.N_posterior = self.N + alpha + beta - 1
        self.Pxx_posterior = (Nxx + alpha) / self.N_posterior


    def _get_loss(self, M_hat, shard):
        return self.criterion(
            M_hat, shard, self.N, self.N_posterior, 
            self.Pxx_posterior, self.Pxx_independent, self.temperature
        )


    def describe(self):
        s = 'Max Posterior Probability Sharder\n'
        s += '\tupdate_density = {}\n'.format(self.update_density)
        s += '\ttemperature = {}\n'.format(self.temperature)
        s += '\tmask_diagonal = {}\n'.format(self.mask_diagonal)
        return s


class KLSharder(MSharder):

    criterion_class = h.hilbert_loss.KLLoss

    def __init__(
        self, bigram, temperature=1, update_density=1, mask_diagonal=False,
        device=None
    ):
        super(KLSharder, self).__init__(
            bigram, update_density, mask_diagonal, device)
        self.temperature = temperature


    def _load_shard(self, shard):

        self.bigram_shard = self.bigram.load_shard(shard, device=self.device)
        Nxx, Nx, Nxt, self.N = self.bigram_shard
        self.Pxx_independent = (Nx / self.N) * (Nxt / self.N)

        exp_mean, exp_std =  h.corpus_stats.calc_exp_pmi_stats(
            self.bigram_shard)
        alpha, beta = h.corpus_stats.calc_prior_beta_params(
            self.bigram_shard, exp_mean, exp_std, self.Pxx_independent)

        self.N_posterior = self.N + alpha + beta - 1

        a = Nxx + alpha
        b = self.N - Nxx + beta
        self.digamma_a = torch.digamma(a) - torch.digamma(a+b)
        self.digamma_b = torch.digamma(b) - torch.digamma(a+b)


    def _get_loss(self, M_hat, shard):
        return self.criterion(
            M_hat, shard, self.N, self.N_posterior, self.Pxx_independent,
            self.digamma_a, self.digamma_b, self.temperature
        )


    def describe(self):
        s = 'KL Sharder\n'
        s += '\tupdate_density = {}\n'.format(self.update_density)
        s += '\ttemperature = {}\n'.format(self.temperature)
        s += '\tmask_diagonal = {}\n'.format(self.mask_diagonal)
        return s



