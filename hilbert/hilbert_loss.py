import torch
import torch.nn as nn
from torch.nn.functional import dropout


# function for applying the minibatching dropout and then
# rescaling back so that it doesn't overweight samples
def keep(tensor, keep_p):
    return dropout(tensor, p=1-keep_p, training=True) * keep_p


class MSELoss(nn.Module):

    def __init__(self, keep_prob, ncomponents):
        super(MSELoss, self).__init__()
        self.keep_prob = keep_prob
        self.rescale = float(keep_prob * ncomponents)

    def forward(self, M_hat, M, weights=None):
        weights = 1 if weights is None else weights
        mse = weights * ((M_hat - M) ** 2)
        mse = keep(mse, self.keep_prob)
        return 0.5 * torch.sum(mse) / self.rescale


class W2VLoss(nn.Module):

    def __init__(self, keep_prob, ncomponents):
        super(W2VLoss, self).__init__()
        self.keep_prob = keep_prob
        self.rescale = float(keep_prob * ncomponents)

    def forward(self, M_hat, Nxx, N_neg):
        logfactor = torch.log(torch.exp(M_hat) + 1)
        term1 = N_neg * logfactor
        term2 = Nxx * (logfactor - M_hat)
        result = keep(term1 + term2, self.keep_prob)
        return torch.sum(result) / self.rescale

