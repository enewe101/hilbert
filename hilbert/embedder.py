import time
import hilbert as h
import torch
import torch.nn as nn
from progress.bar import ChargingBar

class DivergenceError(Exception):
    pass


# noinspection PyCallingNonCallable
def xavier(shape, device):
    return nn.init.xavier_uniform_(torch.zeros(shape, device=device))


class HilbertEmbedderSolver(object):

    def __init__(
        self,
        loader,
        loss,
        optimizer_constructor,
        d=300,
        learning_rate=1e-6,
        opt_kwargs=None,
        init_vecs=None,
        dictionary=None,
        shape=None,
        one_sided=False,
        learn_bias=False,
        seed=1917,
        device=None,
        learner='dense',
        verbose=True
    ):
        """
        This is the base class for a Hilbert Embedder model. It uses pytorch's
        automatic differentiation for the primary heavy lifting. The fundamental
        components of this class are:
            (1) the loss function (hilbert_loss.py)
            (2) the optimizer *constructor* (from torch.optim)
            (3) the dimensionality (an integer for the size of the embeddings)

        :param loader: a Loader object that iterates gpu-loaded shards
        :param optimizer_constructor: a constructor from torch.optim that we 
                will build later
        :param d: the desired dimensionality
        :param learning_rate: learning rate
        :param opt_kwargs: dictionary of keyword arguments for optimizer 
                constructor
        :param init_vecs: vectors to initialize with, as an Embeddings object
        :param shape: the desired shape of the vectors, if no initials passed
        :param one_sided: whether to only learn a set of vectors, forcing the
                covectors to be their transpose
        :param learn_bias: boolean, whether or not to learn bias values for each
                vector and covector (like GloVe does!)
        :param seed: random seed number to use
        :param verbose: verbose
        :param device: gpu or cpu
        """
        if shape is None and init_vecs is None:
            raise ValueError("Provide `shape` or `init_vecs`.")

        self.loader = loader
        self.loss = loss
        self.optimizer_constructor = optimizer_constructor
        self.d = d
        self.learning_rate = learning_rate
        self.dictionary = dictionary
        self.one_sided = one_sided
        self.learn_bias = learn_bias
        self.seed = seed
        self.verbose = verbose
        self.device = device
        self.learner_class = DenseLearner if learner=='dense' else SparseLearner

        opt_kwargs = {} if opt_kwargs is None else opt_kwargs
        self.opt_kwargs = {**{'lr': learning_rate}, **opt_kwargs}

        # code for initializing the vectors & covectors
        self.epoch_loss = None
        self.V, self.W = None, None
        self.vb, self.wb = None, None

        if init_vecs is not None:
            if isinstance(init_vecs, h.embeddings.Embeddings):
                self.V, self.W = (init_vecs.V, None) if one_sided else (
                                 (init_vecs.V, init_vecs.W))
            else:
                self.V, self.W = (init_vecs, None) if one_sided else init_vecs

            self.shape = (self.V.shape[0],) if self.one_sided else (
                         (self.V.shape[0], self.W.shape[0]))

        # If  not initial vectors are given, get random ones
        else:
            self.shape = shape
            self.validate_shape()

        # this sets in motion the torch API big boi
        self.optimizer = None
        self.learner = None
        self.restart(resample_vectors=init_vecs is None)
        self.validate_vectors()


    def validate_vectors(self):
        V_okay = self.V.shape[1] == self.d
        W_okay = self.one_sided or self.W.shape[1] == self.d
        if not V_okay or not W_okay:
            raise ValueError(
                "Embeddings do not have the requested dimension.  Got {}, but "
                "you said d={}".format(self.V.shape[1], self.d)
            )


    def describe(self):
        s = 'self.loader: {}\n--'.format(self.loader.describe())
        sfun = lambda strr, value: '\t{} = {}\n'.format(strr, value)
        s += sfun('optimizer', self.optimizer_constructor)
        s += sfun('d', self.d)
        s += sfun('learning_rate', self.learning_rate)
        s += sfun('one_sided', self.one_sided)
        s += sfun('learn_bias', self.learn_bias)
        s += sfun('seed', self.seed)
        s += sfun('device', self.device)
        return s


    def validate_shape(self):
        if self.one_sided and len(self.shape) != 1:
            raise ValueError(
                "For one-sided embeddings `shape` should be a "
                "tuple containing a single int, e.g. `(10000,)`."
            )

        if not self.one_sided and len(self.shape) != 2:
            raise ValueError(
                "For two-sided embeddings `shape` should be a "
                "tuple containing two ints, e.g. `(10000,10000)`."
            )


    def restart(self, resample_vectors=True):
        device = self.device
        torch.random.manual_seed(self.seed)
        self.opt_kwargs['lr'] = self.learning_rate

        # set the vectors
        vshape = (self.shape[0], self.d)
        if not self.one_sided:
            wshape = (self.shape[1], self.d)

        if resample_vectors:
            self.V = xavier(vshape, device)
            self.W = None if self.one_sided else xavier(wshape, device)

        # initialize the bias vectors, if desired.
        if self.learn_bias:
            self.vb = xavier((1, vshape[0]), device).squeeze()
            if not self.one_sided:
                self.wb = xavier((1, wshape[0]), device).squeeze()

        # now build the auto-embedder
        self.learner = self.learner_class(self.V, self.W, self.vb, self.wb).to(device)

        self.optimizer = self.optimizer_constructor(
            self.learner.parameters(),
            **self.opt_kwargs,
        )


    def get_params(self):
        return self.V, self.W, self.vb, self.wb


    def get_dictionary(self):
        return self.dictionary


    def cycle(self, iters=1, shard_times=1, very_verbose=True):
        losses = []

        for it in range(iters):
            self.epoch_loss = 0

            if very_verbose:
                bar = ChargingBar('Epoch: {:6}'.format(it), max=len(self.loader))

            # iterate over the shards we have to do
            for batch_id, batch_data in self.loader:
                for _ in range(shard_times):

                    # zero out the gradient
                    self.optimizer.zero_grad()

                    # Calculate forward pass, M_hat and loss, for this shard!
                    M_hat = self.learner(batch_id)
                    loss = self.loss(M_hat, batch_data)

                    if torch.isnan(loss):
                        del M_hat
                        del loss
                        raise DivergenceError('Model has completely diverged!')

                    loss.backward()
                    self.optimizer.step()

                    # statistics
                    self.epoch_loss += loss.item()

                    if very_verbose:
                        bar.next()

            losses.append(self.epoch_loss)
            if self.verbose:
                print('  loss\t{}'.format(self.epoch_loss))

        return losses


####
####
#### Main classes that integrates with Pytorch Autodiff API.
####
####
class EmbeddingLearner(nn.Module):
    def __init__(self, V, W, v_bias=None, w_bias=None):
        super(EmbeddingLearner, self).__init__()
        self.learn_bias = v_bias is not None and w_bias is not None
        self.V = nn.Parameter(V)
        self.W = nn.Parameter(W)
        self.v_bias = nn.Parameter(v_bias) if self.learn_bias else None
        self.w_bias = nn.Parameter(w_bias) if self.learn_bias else None

    def forward(self, *input):
        raise NotImplementedError('Not implemented!')



#### Dense, pure MF-based learner
class DenseLearner(EmbeddingLearner):

    def forward(self, shard):
        V = self.V[shard[1]].squeeze()
        W = self.W[shard[0]].squeeze()

        # W is of shape C x d, V is of shape T x d
        # so M_hat is of shape C x T
        M_hat = W @ V.t()

        # add the bias vectors. vbias vector is added to each row
        # while the wbias vector is add to each column
        if self.learn_bias:
            M_hat += self.v_bias[shard[1]].view(1, -1)
            M_hat += self.w_bias[shard[0]].view(-1, 1)

        # andddd that's all folks!
        return M_hat



### A learning based on using a sparse-implementation
# TODO: consider integrating with symmetry.

class SparseLearner(EmbeddingLearner):

    def forward(self, batch_id, symmetric=False):
        row_id, col_ids = batch_id
        v_vec = self.V[row_id]
        W_vecs = self.W[col_ids]

        # v_vec.t() is of shape d x 1, W_vecs is of shape len(js) x d
        # so tc_hat is of shape len(js) x 1
        tc_hat = W_vecs @ v_vec

        if self.learn_bias:
            tc_hat += self.v_bias[row_id]
            tc_hat += self.w_bias[col_ids]

        # andddd that's all folks!
        return tc_hat
