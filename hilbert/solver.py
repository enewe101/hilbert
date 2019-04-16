import time
import hilbert as h
import torch
import torch.nn as nn
from progress.bar import ChargingBar



# noinspection PyCallingNonCallable
def xavier(shape, device):
    return nn.init.xavier_uniform_(torch.zeros(shape, device=device))


class Solver(object):

    def __init__(
        self,
        loader,
        loss,
        learner,
        optimizer,
        schedulers=None,
        dictionary=None,
        verbose=True,
    ):
        """
        This is the base class for a Hilbert Embedder model. It uses pytorch's
        automatic differentiation for the primary heavy lifting. The fundamental
        components of this class are:
            (1) the loss function (loss.py)
            (2) the optimizer *constructor* (from torch.optim)
            (3) the dimensionality (an integer for the size of the embeddings)

        :param loader: a Loader object that iterates gpu-loaded shards
        :param optimizer: a constructor from torch.optim that we 
                will build later
        """

        # Own it like you do
        self.loader = loader
        self.loss = loss
        self.optmizer = optimizer
        self.optimizer = optimizer
        self.learner = learner
        self.schedulers = schedulers or []
        self.dictionary = dictionary
        self.verbose = verbose

        # Other solver state
        self.cur_loss = None


    def describe(self):
        return "I'm the solver."


    def get_embeddings(self):
        detached_embedding_params = (
            p.detach() if p is not None else None
            for p in self.learner.get_embedding_params()
        )
        return h.embeddings.Embeddings(
            *detached_embedding_params, 
            dictionary=self.dictionary,
            verbose=self.verbose
        )
        

    def get_params(self):
        return learner.get_params()


    def cycle(self, updates_per_cycle=1):

        # Run a bunch of updates.
        for update_id in range(updates_per_cycle):

            # Train on as many batches as the loader deems to be one update.
            for batch_id, batch_data in self.loader:

                # Consider this batch and learn.
                self.optimizer.zero_grad()
                M_hat = self.learner(batch_id)
                self.cur_loss = self.loss(M_hat, batch_data)
                self.cur_loss.backward()

                # Take steps
                self.optimizer.step()
                for scheduler in self.schedulers:
                    scheduler.step()

                # Nan police.
                if torch.isnan(self.cur_loss):
                    # Drop your tensors! You're under arrest!
                    del M_hat
                    del self.cur_loss
                    torch.cuda.empty_cache()
                    raise h.exceptions.DivergenceError('Model has diverged!')




####
####
#### Main classes that integrates with Pytorch Autodiff API.
####
####
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
        """Error if the analyst passed in bad inits."""
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
        M_hat = W @ V.t()
        if self.bias:
            M_hat += self.vb[shard[1]].view(1, -1)
            M_hat += self.wb[shard[0]].view(-1, 1)
        return M_hat



class SparseLearner(EmbeddingLearner):

    def forward(self, IJ):
        tc_hat = torch.sum(self.V[IJ[:,0]] * self.W[IJ[:,1]], dim=1)
        if self.bias:
            tc_hat += self.v_bias[IJ[:,0]]
            tc_hat += self.w_bias[IJ[:,1]]

        # andddd that's all folks!
        return tc_hat

