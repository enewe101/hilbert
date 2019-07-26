import torch
import torch.nn as nn

import hilbert as h


### Base class for losses
class HilbertLoss(nn.Module):
    def __init__(self, ncomponents):
        super(HilbertLoss, self).__init__()
        # self.rescale = float(ncomponents)

    def forward(self, M_hat, batch_data):
        return torch.sum(self._forward(M_hat, batch_data))  # / self.rescale

    def _forward(self, M_hat, batch_data):
        raise NotImplementedError('Subclasses must override `_forward`.')


# Special tempered base class for losses that use Pij under independence.
class TemperedLoss(HilbertLoss):
    def __init__(self, ncomponents, temperature=1.):
        super(TemperedLoss, self).__init__(ncomponents)
        self.temperature = temperature
        self.pxx_independent = None

    def _forward(self, M_hat, batch_data):
        untempered, pxx_independent = self._forward_temper(M_hat, batch_data)
        if self.temperature != 1:
            # Generally already computed, and could be memoised on self.
            tempering = pxx_independent ** (1 / self.temperature - 1)
            return untempered * tempering
        return untempered

    def get_pxx_independent(self, batch_data):
        cooccurrence_data, unigram_data = batch_data
        Nxx, Nx, Nxt, N = cooccurrence_data
        return (Nx / N) * (Nxt / N)

    def _forward_temper(self, M_hat, batch_data):
        raise NotImplementedError("Subclasses must override `_forward_temper`.")


class GloveLoss(HilbertLoss):
    REQUIRES_UNIGRAMS = False

    def __init__(self, ncomponents, X_max=100, alpha=3 / 4):
        super(GloveLoss, self).__init__(ncomponents)
        self.X_max = X_max
        self.alpha = alpha

    def _forward(self, response, batch_data):
        cooccurrence_data, unigram_data = batch_data
        Nxx, Nx, Nxt, N = cooccurrence_data
        expected_response = torch.log(Nxx)
        Nxx_is_zero = (Nxx == 0)
        expected_response[Nxx_is_zero] = 0
        weights = (Nxx / self.X_max).pow(self.alpha)
        weights = torch.clamp(weights, max=1.)
        weights[Nxx_is_zero] = 0
        weights = weights * 2
        return 0.5 * weights * ((response - expected_response) ** 2)


class SGNSLoss(HilbertLoss):
    REQUIRES_UNIGRAMS = True

    def __init__(self, ncomponents, k=15, device=None):
        super(SGNSLoss, self).__init__(ncomponents)
        self.device = h.utils.get_device(device)
        self.k = torch.tensor(
            k, device=self.device, dtype=h.CONSTANTS.DEFAULT_DTYPE)

    def _forward(self, response, batch_data):
        cooccurrence_data, unigram_data = batch_data
        Nxx, Nx, Nxt, N = cooccurrence_data
        uNx, uNxt, uN = unigram_data
        N_neg = self.negative_sample(Nxx, Nx, uNxt, uN, self.k)
        logfactor = torch.log(torch.exp(response) + 1)
        term1 = N_neg * logfactor
        term2 = Nxx * (logfactor - response)
        return term1 + term2

    @staticmethod
    def negative_sample(Nxx, Nx, uNxt, uN, k):
        return k * (Nx - Nxx) * (uNxt / uN)


class MLELoss(TemperedLoss):
    REQUIRES_UNIGRAMS = False

    def _forward_temper(self, response, batch_data):
        cooccurrence_data, unigram_data = batch_data
        Nxx, Nx, Nxt, N = cooccurrence_data
        Pxx_data = Nxx / N
        term1 = Pxx_data * response
        pxx_independent = self.get_pxx_independent(batch_data)
        term2 = pxx_independent * torch.exp(response)
        return - (term1 - term2), pxx_independent


class SampleMLELoss(nn.Module):
    def forward(self, response, batch_data):
        boundary = int(response.shape[0] / 2)
        term1 = response[:boundary].sum()
        term2 = torch.exp(response[boundary:]).sum()
        return - (term1 - term2) / float(boundary)


class BalancedSampleMLELoss(nn.Module):
    def forward(self, response, batch_data):
        deviations = torch.exp(response) - batch_data['exp_pmi'] * response
        return deviations.sum() / response.shape[0]


class GibbsSampleMLELoss(nn.Module):
    def forward(self, response, batch_data):
        boundary = int(response.shape[0] / 2)
        term1 = response[:boundary].sum()
        term2 = response[boundary:].sum()
        return -(term1 - term2) / float(boundary)
