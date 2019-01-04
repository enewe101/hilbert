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

    def forward(self, shard_id, M_hat, shard_data, *args, **kwargs):
        elementwise_loss = self._forward(
            shard_id, M_hat, shard_data, *args, **kwargs)
        if self.mask_diagonal and h.shards.on_diag(shard_id):
            mask_diagonal(elementwise_loss)
        minibatched_loss = keep(elementwise_loss, self.keep_prob)
        return torch.sum(minibatched_loss) / self.rescale

    def _forward(self):
        raise NotImplementedError('Subclasses must override `_forward`.')



class MSELoss(HilbertLoss):
    def _forward(self, shard_id, M_hat, shard_data):
        weights = shard_data.get('weights', 1)
        M = shard_data['M']
        return 0.5 * weights * ((M_hat - M) ** 2)


class Word2vecLoss(HilbertLoss):
    def _forward(self, shard_id, M_hat, shard_data):
        logfactor = torch.log(torch.exp(M_hat) + 1)
        term1 = shard_data['N_neg'] * logfactor
        term2 = shard_data['Nxx'] * (logfactor - M_hat)
        return term1 + term2


class MaxLikelihoodLoss(HilbertLoss):
    def _forward(self, shard_id, M_hat, shard_data):
        Pxx_model = shard_data['Pxx_independent'] * torch.exp(M_hat)
        term1 = shard_data['Pxx_data'] * torch.log(Pxx_model) 
        term2 = (1-shard_data['Pxx_data']) * torch.log(1-Pxx_model)
        result = - (term1 + term2)
        return temper(
            result, shard_data['Pxx_independent'], shard_data['temperature'])


class MaxPosteriorLoss(HilbertLoss):
    def _forward(self, shard_id, M_hat, shard_data):
        Pxx_model = shard_data['Pxx_independent'] * torch.exp(M_hat)
        term1 = shard_data['Pxx_posterior'] * torch.log(Pxx_model) 
        term2 = (1-shard_data['Pxx_posterior']) * torch.log(1-Pxx_model)
        result =  -(shard_data['N_posterior']/shard_data['N']) * (term1 + term2)
        return temper(
            result, shard_data['Pxx_independent'], shard_data['temperature'])


class KLLoss(HilbertLoss):
    def _forward(self, shard_id, M_hat, shard_data):
        Pxx_model = shard_data['Pxx_independent'] * torch.exp(M_hat)
        a_hat = shard_data['N_posterior'] * Pxx_model
        a_term = a_hat * shard_data['digamma_a']
        b_hat = shard_data['N_posterior'] * (1 - Pxx_model) + 1
        b_term = b_hat * shard_data['digamma_b']
        result = (lbeta(a_hat, b_hat) - a_term - b_term) / shard_data['N']
        return temper(
            result, shard_data['Pxx_independent'], shard_data['temperature'])


