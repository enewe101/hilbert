import hilbert as h
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class DivergenceError(Exception):
    pass


# noinspection PyCallingNonCallable
def xavier(shape, device):
    return nn.init.xavier_uniform_(torch.zeros(shape, device=device))


class HilbertEmbedderSolver(object):

    def __init__(
        self,
        sharder,
        optimizer_constructor,
        d=300,
        learning_rate=1e-6,
        opt_kwargs=None,
        init_vecs=None,
        shape=None,
        one_sided=False,
        learn_bias=False,
        shard_factor=10,
        seed=1917,
        verbose=True,
        device=None
    ):
        """
        This is the base class for a Hilbert Embedder model. It uses pytorch's
        automatic differentiation for the primary heavy lifting. The fundamental
        components of this class are:
            (1) the loss function (hilbert_loss.py)
            (2) the optimizer *constructor* (from torch.optim)
            (3) the dimensionality (an integer for the size of the embeddings)

        :param sharder: an MSharder object that stores the loss
        :param optimizer_constructor: a constructor from torch.optim that we will build later
        :param d: the desired dimensionality
        :param learning_rate: learning rate
        :param opt_kwargs: dictionary of keyword arguments for optimizer constructor
        :param init_vecs: vectors to initialize with, as an Embeddings object
        :param shape: the desired shape of the vectors, if no initials passed
        :param one_sided: whether to only learn a set of vectors, forcing the
                covectors to be their transpose
        :param learn_bias: boolean, whether or not to learn bias values for each
                vector and covector (like GloVe does!)
        :param shard_factor: how much to shard the model
        :param seed: random seed number to use
        :param verbose: verbose
        :param device: gpu or cpu
        """
        if shape is None and init_vecs is None:
            raise ValueError("Provide `shape` or `init_vecs`.")

        self.sharder = sharder
        self.optimizer_constructor = optimizer_constructor
        self.d = d
        self.learning_rate = learning_rate
        self.one_sided = one_sided
        self.learn_bias = learn_bias
        self.shard_factor = shard_factor
        self.seed = seed
        self.verbose = verbose
        self.device = device

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

        # this sets in motion the torch API big boy
        self.optimizer = None
        self.learner = None
        self.restart(resample_vectors=init_vecs is None)


    def describe(self):
        s = 'Sharder: {}\n--'.format(self.sharder.describe())
        sfun = lambda strr, value: '\t{} = {}\n'.format(strr, value)
        s += sfun('optimizer', self.optimizer_constructor)
        s += sfun('d', self.d)
        s += sfun('learning_rate', self.learning_rate)
        s += sfun('one_sided', self.one_sided)
        s += sfun('learn_bias', self.learn_bias)
        s += sfun('shard_factor', self.shard_factor)
        s += sfun('seed', self.seed)
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
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        torch.random.manual_seed(self.seed)

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
        self.learner = AutoEmbedder(self.V, self.W, self.vb, self.wb).to(device)

        self.optimizer = self.optimizer_constructor(
            self.learner.parameters(),
            **self.opt_kwargs,
        )
        self.optimizer = ReduceLROnPlateau(self.optimizer,
            mode='min',
            factor=0.5,
            patience=250,
            verbose=True,
            min_lr=1e-8,
        )


    def get_params(self):
        return self.V, self.W, self.vb, self.wb


    def get_dictionary(self):
        return self.sharder.bigram.dictionary


    def cycle(self, epochs=1, shard_times=1, hold_loss=False):
        losses = [] if hold_loss else None

        for _ in range(epochs):
            self.epoch_loss = 0

            # iterate over the shards we have to do
            for shard in h.shards.Shards(self.shard_factor):
                for _ in range(shard_times):

                    # zero out the gradient
                    self.optimizer.zero_grad()

                    # get our mhat for the shard!
                    M_hat = self.learner(shard)
                    loss = self.sharder.calc_shard_loss(M_hat, shard)

                    if torch.isnan(loss):
                        raise DivergenceError('Model has completely diverged!')

                    loss.backward()
                    self.optimizer.step(loss.item())

                    # statistics
                    self.epoch_loss += loss.item()

            if hold_loss:
                losses.append(self.epoch_loss)

            if self.verbose:
                print('loss\t{}'.format(self.epoch_loss))

        return losses



#### Main class that integrates with Pytorch Autodiff API.
class AutoEmbedder(nn.Module):

    def __init__(self, V, W=None, v_bias=None, w_bias=None):
        super(AutoEmbedder, self).__init__()
        self.learn_w = W is not None
        self.learn_bias = v_bias is not None
        wb = self.learn_w and self.learn_bias

        # annoying initialization
        self.V = nn.Parameter(V)
        self.W = nn.Parameter(W) if self.learn_w else None
        self.v_bias = nn.Parameter(v_bias) if self.learn_bias else None
        self.w_bias = nn.Parameter(w_bias) if wb else None


    def forward(self, shard):
        V = self.V[shard[1]].squeeze()

        if self.learn_w:
            W = self.W[shard[0]].squeeze()
        else:
            W = self.V[shard[0]].squeeze()

        # W is of shape C x d, V is of shape T x d
        # so M_hat is of shape C x T
        M_hat = W @ V.t()

        # add the bias vectors. vbias vector is added to each row
        # while the wbias vector is add to each column
        if self.learn_bias:
            M_hat += self.v_bias[shard[1]].reshape(1, -1)

            if self.learn_w:
                M_hat += self.w_bias[shard[0]].reshape(-1, 1)
            else:
                M_hat += self.v_bias[shard[0]].reshape(-1, 1)

        # andddd that's all folks!
        return M_hat
