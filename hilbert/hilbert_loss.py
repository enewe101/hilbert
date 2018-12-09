import torch
import torch.nn as nn
from torch.nn.functional import dropout, sigmoid


# function for applying the minibatching dropout and then
# rescaling back so that it doesn't overweight samples
def keep(tensor, keep_p):
    return dropout(tensor, p=1-keep_p, training=True) * keep_p


class MSELoss(nn.Module):

    def forward(self, M_hat, M, keep_prob, weights=None):
        weights = 1 if weights is None else weights
        mse = weights * ((M_hat - M) ** 2)
        mse = keep(mse, keep_prob)
        return 0.5 * torch.sum(mse)


# TODO: verify correctness!
class W2VLoss(nn.Module):

    def forward(self, M_hat, Nxx, N_neg, keep_prob):
        smhat = sigmoid(M_hat)
        total = (
            (Nxx * torch.log(smhat)) +
            (N_neg * torch.log(1 - smhat))
        )
        return -torch.sum(keep(total, keep_prob))

