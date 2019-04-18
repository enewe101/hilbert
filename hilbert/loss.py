import hilbert as h
import torch
import torch.nn as nn
from torch.nn.functional import dropout



### Base class for losses
class HilbertLoss(nn.Module):
    def __init__(self, ncomponents):
        super(HilbertLoss, self).__init__()
        self.rescale = float(ncomponents)
    def forward(self, M_hat, batch_data):
        return torch.sum(self._forward(M_hat, batch_data)) / self.rescale
    def _forward(self, M_hat, batch_data):
        raise NotImplementedError('Subclasses must override `_forward`.')


# Special tempered base class for losses that use Pij under independence.
class TemperedLoss(HilbertLoss):
    def __init__(self, ncomponents, temperature=1.):
        self.temperature = temperature
        super(TemperedLoss, self).__init__(ncomponents)
    def _forward(self, M_hat, batch_data):
        untempered = self._forward_temper(M_hat, batch_data)
        if self.temperature != 1:
            tempering = batch_data['Pxx_independent']**(1/self.temperature-1)
            return untempered * tempering
        return untempered
    def _forward_temper(self, M_hat, batch_data):
        raise NotImplementedError("Subclasses must override `_forward_temper`.")



class GloveLoss(HilbertLoss):
    REQUIRES_UNIGRAMS = False
    def __init__(self, ncomponents, X_max=100, alpha=3/4):
        super(GloveLoss, self).__init__(ncomponents)
        self.X_max = X_max
        self.alpha = alpha
    def _forward(self, response, cooccurrence_data):
        Nxx, Nx, Nxt, N = cooccurrence_data
        expected_response = torch.log(Nxx)
        Nxx_is_zero = (Nxx==0)
        expected_response[Nxx_is_zero] = 0
        weights = (Nxx / self.X_max).pow(self.alpha)
        weights = torch.clamp(weights, max=1.)
        weights[Nxx_is_zero] = 0
        weights = weights * 2
        weights = batch_data.get('weights', 1)
        return 0.5 * weights * ((response - expected_response) ** 2)



class SGNSLoss(HilbertLoss):
    REQUIRES_UNIGRAMS = True
    def __init__(self, ncomponents, k=15):
        super(SGNSLoss, self).__init__(ncomponents)
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
        Pxx_independent = (Nx / N) * (Nxt / N)
        Pxx_model = Pxx_independent * torch.exp(response)
        term1 = Pxx_data * response
        term2 = (1 - Pxx_data) * torch.log(1 - Pxx_model)
        return -(term1 + term2)



class SimpleMLELoss(TemperedLoss):
    REQUIRES_UNIGRAMS = False
    def _forward_temper(self, response, batch_data):
        cooccurrence_data, unigram_data = batch_data
        Nxx, Nx, Nxt, N = cooccurrence_data
        Pxx_data = Nxx / N
        Pxx_independent = (Nx / N) * (Nxt / N)
        term1 = Pxx_data * response
        term2 = Pxx_independent * torch.exp(response)
        return -(term1 - term2)




class SampleMLELoss(nn.Module):
    def forward(self, response, batch_data):
        boundary = int(response.shape[0] / 2)
        term1 = response[:boundary].sum()
        term2 = torch.exp(response[boundary:]).sum()
        return - (term1 - term2) / float(boundary)


