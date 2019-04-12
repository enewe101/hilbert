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



### All specific losses, GloVe uses MSE
class MSELoss(HilbertLoss):
    def _forward(self, M_hat, batch_data):
        weights = batch_data.get('weights', 1)
        M = batch_data['M']
        return 0.5 * weights * ((M_hat - M) ** 2)



class Word2vecLoss(HilbertLoss):
    def _forward(self, M_hat, batch_data):
        logfactor = torch.log(torch.exp(M_hat) + 1)
        term1 = batch_data['N_neg'] * logfactor
        term2 = batch_data['Nxx'] * (logfactor - M_hat)
        return term1 + term2



class MaxLikelihoodLoss(TemperedLoss):
    def _forward_temper(self, M_hat, batch_data):
        Pxx_model = batch_data['Pxx_independent'] * torch.exp(M_hat)
        term1 = batch_data['Pxx_data'] * M_hat
        term2 = (1 - batch_data['Pxx_data']) * torch.log(1 - Pxx_model)
        return -(term1 + term2)


class SampleMaxLikelihoodLoss(nn.Module):
    def forward(self, M_hat, batch_data):
        boundary = int(M_hat.shape[0] / 2)
        return - (M_hat[:boundary].sum() - torch.exp(M_hat[boundary:]).sum())


class SimpleMaxLikelihoodLoss(TemperedLoss):
    def _forward_temper(self, M_hat, batch_data):
        term1 = batch_data['Pxx_data'] * M_hat
        term2 = batch_data['Pxx_independent'] * torch.exp(M_hat)
        return -(term1 - term2)


