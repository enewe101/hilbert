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


class MaxLikelihoodLoss(nn.Module):

    def __init__(self, keep_prob, ncomponents):
        super(MaxLikelihoodLoss, self).__init__()
        self.keep_prob = keep_prob
        self.rescale = float(keep_prob * ncomponents)

    def forward(self, M_hat, Pxx_data, Pxx_independent):
        Pxx_model = Pxx_independent * torch.exp(M_hat)
        term1 = Pxx_data * torch.log(Pxx_model) 
        term2 = (1-Pxx_data) * torch.log(1-Pxx_model)
        result = keep(term1 + term2, self.keep_prob)

        # We want to maximize log likelihood, so minimize it's negative
        return - torch.sum(result) / self.rescale


class MaxPosteriorLoss(nn.Module):

    def __init__(self, keep_prob, ncomponents):
        super(MaxPosteriorLoss, self).__init__()
        self.keep_prob = keep_prob
        self.rescale = float(keep_prob * ncomponents)

    def forward(self, M_hat, N, N_posterior, Pxx_posterior, Pxx_independent):
        Pxx_model = Pxx_independent * torch.exp(M_hat)
        term1 = Pxx_posterior * torch.log(Pxx_model) 
        term2 = (1-Pxx_posterior) * torch.log(1-Pxx_model)
        result = (N_posterior / N) * (term1 + term2)
        keep_result = keep(result, self.keep_prob)

        # Want to maximize posterior log probability, so minimize its negative
        return - torch.sum(keep_result) / self.rescale



class KLLoss(nn.Module):

    def __init__(self, keep_prob, ncomponents):
        super(KLLoss, self).__init__()
        self.keep_prob = keep_prob
        self.rescale = float(keep_prob * ncomponents)

    def forward(M_hat, N, N_posterior, Pxx_independent, digamma_a, digamma_b):

        Pxx_model = Pxx_independent * torch.exp(M_hat)
        a_hat = N_posterior * Pxx_model
        b_hat = N_posterior * (1 - Pxx_model) + 1
        ln_beta = ln_beta(a_hat, b_hat)

        result = (ln_beta - a_hat * digamma_a - b_hat * digamma_b) / N
        keep_result = keep(result, self.keep_prob)

        # Should this return the negative?
        return torch.sum(keep_result) / self.rescale


def ln_beta(a,b):
    return (
        torch.lgamma(a_hat) + torch.lgamma(b_hat) 
        - torch.lgamma(a_hat + b_hat)
    )
