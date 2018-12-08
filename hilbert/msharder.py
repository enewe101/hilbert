import hilbert as h
import torch
import numpy as np
from scipy import sparse


# base abstract class for the other sharders
class MSharder(object):

    def __init__(self, bigram, update_density=1, device=None):
        self.bigram = bigram
        self.device = device or h.CONSTANTS.MATRIX_DEVICE
        self.update_density = update_density
        self.last_shard = None


    def calc_shard_loss(self, M_hat, shard):
        if shard != self.last_shard:
            self.last_shard = shard
            self._load_shard(shard)

        return self._get_loss(M_hat, self.update_density)


    def _load_shard(self,  shard):
        raise NotImplementedError('Subclasses must override `load_shard`.')


    def _get_loss(self, M_hat, keep_prob=1):
        raise NotImplementedError('Subclasses must override `get_loss`.')



class PPMISharder(MSharder):

    def __init__(self, *args, **kwargs):
        super(PPMISharder, self).__init__(*args, **kwargs)
        self.criterion = h.hilbert_loss.MSELoss()


    def _load_shard(self, shard):
        Nxx, Nx, Nxt, N = self.bigram.load_shard(shard, device=self.device)
        self.M = h.corpus_stats.calc_PMI((Nxx, Nx, Nxt, N))
        self.M = torch.clamp(self.M, min=0)


    def _get_loss(self, M_hat, keep_prob=1):
        return self.criterion(M_hat, self.M, keep_prob)



class GloveSharder(MSharder):

    def __init__(
        self, bigram, X_max=100.0, alpha=0.75,
        update_density=1, device=None,
    ):
        super(GloveSharder, self).__init__(bigram, update_density, device)
        self.X_max = float(X_max)
        self.alpha = alpha
        self.criterion = h.hilbert_loss.MSELoss()


    def _load_shard(self, shard):
        Nxx, Nx, Nxt, N = self.bigram.load_shard(shard, device=self.device)
        self.multiplier = (Nxx / self.X_max).pow(self.alpha)
        self.multiplier = torch.clamp(self.multiplier, max=1.)
        self.M = torch.log(Nxx)

        # Zero-out cells that have Nxx==0, since GloVe ignores these
        self.M[Nxx==0] = 0
        self.multiplier[Nxx==0] = 0


    def _get_loss(self, M_hat, keep_prob=1):
        return 2 * self.criterion(M_hat, self.M, keep_prob, self.multiplier)



# sharder for W2V
# noinspection PyCallingNonCallable
class Word2vecSharder(MSharder):

    def __init__(self, bigram, k, update_density=1, device=None):
        super(Word2vecSharder, self).__init__(
            bigram, update_density, device)
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        self.k = torch.tensor(k, device=self.device, dtype=dtype)


    def load_shard(self, shard):
        self.Nxx, Nx, Nxt, N = self.bigram.load_shard(shard, device=self.device)
        uNx, uNxt, uN = self.bigram.unigram.load_shard(shard,device=self.device)
        self.N_neg = h.M.negative_sample(self.Nxx, Nx, uNxt, uN, self.k)
        self.multiplier = self.Nxx + self.N_neg


    def _get_loss(self, M_hat, keep_prob=1):
        raise NotImplementedError('need to do this!')

        # This simplified form is equivalent to (but a bit cheaper than):
        #      (Nxx + N_Neg) * (sigmoid(M) - sigmoid(M_hat))
        # return self.Nxx - (self.Nxx + self.N_neg) * M_hat.sigmoid()



