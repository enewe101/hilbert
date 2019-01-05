import hilbert as h
import torch
import torch.nn as nn
from torch.nn.functional import dropout


# function for applying the minibatching dropout and then
# rescaling back so that it doesn't overweight samples
def keep(tensor, keep_p):
    return dropout(tensor, p=1-keep_p, training=True) * keep_p


def lbeta(a,b):
    """Log of the Beta function."""
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a+b)


def temper(loss, Pxx_independent, temperature):
    """
    Reweights an array of pairwise losses, used when losses are proportional
    to the cooccurrence probabilities of token pairs assuming independence.
    High `temperature`, e.g t=100 leads to equalized weights.  
    `temperature = 1` provide no reweighting.  `temperature` should be 
    greater than or equal to 1.
    """
    if temperature != 1:
        return loss * Pxx_independent ** (1/temperature - 1)
    return loss


def mask_diagonal(tensor):
    """
    Multiplies the main diagonal of `tensor` by zero.  This prevents updates
    to parameters that would result due to their influence on the main diagonal.
    """
    # When viewing the tensor as a 1D list, the entries corresponding to
    # diagonals happen every row_length + 1 elements.
    tensor.view(-1)[::tensor.shape[1]+1] *= 0


class HilbertLoss(nn.Module):

    def __init__(self, keep_prob, ncomponents, mask_diagonal=False):
        super(HilbertLoss, self).__init__()
        self.keep_prob = keep_prob
        self.mask_diagonal = mask_diagonal
        self.rescale = float(keep_prob * ncomponents)

    def forward(self, M_hat, shard, *args, **kwargs):
        elementwise_loss = self._forward(M_hat, shard, *args, **kwargs)
        if self.mask_diagonal and h.shards.on_diag(shard):
            mask_diagonal(elementwise_loss)
        minibatched_loss = keep(elementwise_loss, self.keep_prob)
        return torch.sum(minibatched_loss) / self.rescale

    def _forward(self):
        raise NotImplementedError('Subclasses must override `_forward`.')



class MSELoss(HilbertLoss):
    def _forward(self, M_hat, shard, M, weights=None):
        weights = 1 if weights is None else weights
        return 0.5 * weights * ((M_hat - M) ** 2)


class W2VLoss(HilbertLoss):
    def _forward(self, M_hat, shard, Nxx, N_neg):
        logfactor = torch.log(torch.exp(M_hat) + 1)
        term1 = N_neg * logfactor
        term2 = Nxx * (logfactor - M_hat)
        return term1 + term2


class MaxLikelihoodLoss(HilbertLoss):
    def _forward(self, M_hat, shard, Pxx_data, Pxx_independent, temperature):
        Pxx_model = Pxx_independent * torch.exp(M_hat)
        term1 = Pxx_data * M_hat 
        term2 = (1-Pxx_data) * torch.log(1-Pxx_model)
        result = - (term1 + term2)
        return temper(result, Pxx_independent, temperature)


#class MaxLikelihoodLoss(HilbertLoss):
#    def _forward(self, M_hat, shard, Pxx_data, Pxx_independent, temperature):
#        Pxx_model = Pxx_independent * torch.exp(M_hat)
#        term1 = Pxx_data * torch.log(Pxx_model) 
#        term2 = (1-Pxx_data) * torch.log(1-Pxx_model)
#        result = - (term1 + term2)
#        return temper(result, Pxx_independent, temperature)


class MaxPosteriorLoss(HilbertLoss):
    def _forward(
        self, M_hat, shard, N, N_posterior, Pxx_posterior, Pxx_independent,
        temperature
    ):
        Pxx_model = Pxx_independent * torch.exp(M_hat)
        term1 = Pxx_posterior * M_hat
        term2 = (1-Pxx_posterior) * torch.log(1-Pxx_model)
        result =  - (N_posterior / N) * (term1 + term2)
        return temper(result, Pxx_independent, temperature)

#class MaxPosteriorLoss(HilbertLoss):
#    def _forward(
#        self, M_hat, shard, N, N_posterior, Pxx_posterior, Pxx_independent,
#        temperature
#    ):
#        Pxx_model = Pxx_independent * torch.exp(M_hat)
#        term1 = Pxx_posterior * torch.log(Pxx_model) 
#        term2 = (1-Pxx_posterior) * torch.log(1-Pxx_model)
#        result =  - (N_posterior / N) * (term1 + term2)
#        return temper(result, Pxx_independent, temperature)


class KLLoss(HilbertLoss):
    def _forward(
        self, M_hat, shard, N, N_posterior, Pxx_independent, digamma_a,
        digamma_b, temperature
    ):
        Pxx_model = Pxx_independent * torch.exp(M_hat)
        a_hat = N_posterior * Pxx_model
        b_hat = N_posterior * (1 - Pxx_model) + 1
        result = (lbeta(a_hat,b_hat) - a_hat*digamma_a - b_hat*digamma_b) / N
        return temper(result, Pxx_independent, temperature)


