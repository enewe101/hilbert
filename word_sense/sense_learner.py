
# import hilbert as h
import torch
import torch.nn as nn


def xavier(shape, device):
    return nn.init.xavier_uniform_(torch.zeros(shape, device=device))

class SenseEmbeddingLearner(nn.Module):
    '''
    New embedding learner that can account for words as multiple senses representations
    '''

    def __init__(
            self,
            vocab=None,
            covocab=None,
            d=None,
            bias=False,
            num_sense=None,
            init=None,
            device=None

        ):

        super(SenseEmbeddingLearner, self).__init__()
        self.V_shape = (vocab, num_sense, d)
        self.W_shape = (covocab, num_sense, d)
        # bias is not used in the original code, but still want to make it compatible
        self.vb_shape = (vocab, num_sense, 1)
        self.wb_shape = (covocab, num_sense, 1)
        self.bias = bias
        self.device = device

        if init is None:
                    self.V, self.W, self.vb, self.wb = None, None, None, None
                    self.reset()
        else:
            # if there is a initialized pretrained embedding; embedding is a 4 tuple parameters
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

    def forward(self, *input):
        raise NotImplementedError('Not implemented!')

class SenseLearner(SenseEmbeddingLearner):
    def forward(self, shard):
        '''
        For each sense in word matrix V, repeat each sense such that the dimension of the embedding axis is |S|*d.
            Dim(V) = [vocab, senses, d*senses]
        For each word in co-matrix W, flatten all senses along the embedding axis
            Dim(W) = [covocab, 1, d*senses]
        For each sense in V, compute the dot product with W, giving |S| matrices
        where M(i,j)_{a_sense_of_vi} = sum_(for_all_m)(vi_sense*wj_sense(m)).
        Summing over all such matrices along senses axis will give M(i,j) = sum_{for_all_n} sum_{for_all_m} (vi_sense_n)*(wj_sense_m)
        n m are number of senses in each word as term/context.
        Note V is the matrices cube; W is the co-matrices cube

        Operation computing: W [dot] V. the [dot] operation between two cubes is defined above

        1. concat all senses of words in W along the embedding axis
        2. for each senses of words in V, paste the same sense embedding along the embedding space axis
        '''

        V_cube = self.V[shard[1]].squeeze()
        W_cube = self.W[shard[0]].squeeze()

        response = torch.zeros(W_cube.shape[0], V_cube.shape[0], device=self.device)


        # W  W_shape = (covocab, num_sense, d)
        # print(" ------------------ W ----------------------")

        w_num_sense = self.W_shape[1]
        # TODO: use non memory copying function instead of repeat
        W_cube = W_cube.repeat(1, 1, w_num_sense)
        # W_cube = W_cube.expand(1,1,w_num_sense)
        W_cube = W_cube.transpose(0,1)
        # V_shape = (vocab, num_sense, d)
        V_cube = torch.flatten(V_cube, start_dim=1)

        for sense in range(w_num_sense):
            response += W_cube[sense] @ V_cube.t()


        return response



