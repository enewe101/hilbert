import hilbert as h
import torch
import torch.nn as nn

# noinspection PyCallingNonCallable
def xavier(shape, device):
    return nn.init.xavier_uniform_(torch.zeros(shape, device=device))



class EmbeddingLearner(nn.Module):

    def __init__(
            self,
            vocab=None,
            covocab=None,
            d=None,
            bias=False,
            init=None,
            device=None
        ):

        super(EmbeddingLearner, self).__init__()

        # Own it
        self.V_shape = (vocab, d)
        self.W_shape = (covocab, d)
        self.vb_shape = (1, vocab)
        self.wb_shape = (1, covocab)
        self.bias = bias
        self.device = h.utils.get_device(device)

        # Initialize the model parameters.
        if init is None:
            self.V, self.W, self.vb, self.wb = None, None, None, None
            self.reset()
        else:
            self.V = nn.Parameter(init[0], True)
            self.W = nn.Parameter(init[1], True) 
            self.vb = None if init[2] is None else nn.Parameter(init[2], True)
            self.wb = None if init[3] is None else nn.Parameter(init[3], True)
            self._validate_initialization()


    def _validate_initialization(self):
        """Raise an error if the caller passed in bad inits."""
        if not self.bias and (self.vb is not None or self.wb is not None):
            raise ValueError('No-bias model initialized with biases.')
        elif self.bias and (self.vb is None or self.wb is None):
            raise ValueError('Bias model initialized without biases.')
        if self.V.shape != self.V_shape or self.W.shape != self.W_shape:
            raise ValueError(
                "Model parameters have initialized with incorrect shape. "
                "Got {}, but expected {}.".format(self.V.shape, self.V_shape)
            )


    def reset(self):
        self.V = nn.Parameter(xavier(self.V_shape, self.device), True)
        self.W = nn.Parameter(xavier(self.W_shape, self.device), True)
        if self.bias:
            self.vb = nn.Parameter(
                xavier(self.vb_shape, self.device).squeeze(), True)
            self.wb = nn.Parameter(
                xavier(self.wb_shape, self.device).squeeze(), True)


    def get_embedding_params(self):
        """Return just the model parameters that constitute "embeddings"."""
        return self.V, self.W, self.vb, self.wb


    # TODO: this shouldn't be necessary, self.parameters will automatically
    # collect all tensors that are Parameters.
    def get_params(self):
        """Return *all* the model params."""
        return self.V, self.W, self.vb, self.wb


    def forward(self, *input):
        raise NotImplementedError('Not implemented!')



class DenseLearner(EmbeddingLearner):
    def forward(self, shard):
        V = self.V[shard[1]].squeeze()
        W = self.W[shard[0]].squeeze()
        response = W @ V.t()
        if self.bias:
            response += self.vb[shard[1]].view(1, -1)
            response += self.wb[shard[0]].view(-1, 1)
        return response



class SampleLearner(EmbeddingLearner):
    def forward(self, IJ):
        response = torch.sum(self.V[IJ[:,0]] * self.W[IJ[:,1]], dim=1)
        if self.bias:
            response += self.v_bias[IJ[:,0]]
            response += self.w_bias[IJ[:,1]]
        return response



class MultisenseLearner(nn.Module):

    def __init__(
            self,
            vocab=None,
            covocab=None,
            d=None,
            num_senses=None,
            bias=False,
            init=None,
            device=None
        ):

        super(MultisenseLearner, self).__init__()

        # Own it
        self.V_shape = (vocab, d, num_senses)
        self.W_shape = (covocab, d, num_senses)
        self.vb_shape = (vocab, num_senses)
        self.wb_shape = (covocab, num_senses)
        self.vocab = vocab
        self.covocab = covocab
        self.num_senses = num_senses
        self.bias = bias
        self.device = h.utils.get_device(device)

        # Initialize the model parameters.
        if init is None:
            self.V, self.W, self.vb, self.wb = None, None, None, None
            self.reset()
        else:
            self.V = nn.Parameter(init[0], True)
            self.W = nn.Parameter(init[1], True) 
            self.vb = None if init[2] is None else nn.Parameter(init[2], True)
            self.wb = None if init[3] is None else nn.Parameter(init[3], True)
            self._validate_initialization()


    def _validate_initialization(self):
        """Raise an error if the caller passed in bad inits."""
        if not self.bias and (self.vb is not None or self.wb is not None):
            raise ValueError('No-bias model initialized with biases.')
        elif self.bias and (self.vb is None or self.wb is None):
            raise ValueError('Bias model initialized without biases.')
        if self.V.shape != self.V_shape or self.W.shape != self.W_shape:
            raise ValueError(
                "Model parameters have initialized with incorrect shape. "
                "Got {}, but expected {}.".format(self.V.shape, self.V_shape)
            )


    def reset(self):
        self.V = nn.Parameter(xavier(self.V_shape, self.device), True)
        self.W = nn.Parameter(xavier(self.W_shape, self.device), True)
        if self.bias:
            self.vb = nn.Parameter(
                xavier(self.vb_shape, self.device).squeeze(), True)
            self.wb = nn.Parameter(
                xavier(self.wb_shape, self.device).squeeze(), True)


    def get_embedding_params(self):
        """Return just the model parameters that constitute "embeddings"."""
        return self.V, self.W, self.vb, self.wb


    # TODO: this shouldn't be necessary, self.parameters will automatically
    # collect all tensors that are Parameters.
    def get_params(self):
        """Return *all* the model params."""
        return self.V, self.W, self.vb, self.wb


    def forward(self, IJ):

        # Sum the bias terms for each sense combination in each sample.
        bias = 0
        if self.bias:
            bias = (
                self.wb[IJ[:,1]].view(IJ.shape[0], self.num_senses, 1)
                + self.vb[IJ[:,0]].view(IJ.shape[0], 1, self.num_senses)
            )

        # Calculate the inner product of each sense combination in each sample.
        mat_muls = torch.bmm(
            self.W[IJ[:,1]].transpose(dim0=1, dim1=2),
            self.V[IJ[:,0]]
        ) 

        # Calculate the super-dot-product for each sample.
        mat_muls += bias
        exped = torch.exp(mat_muls)
        response = torch.log(exped.sum(dim=2).sum(dim=1))

        return response







