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
            print("initialized with bias")
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


    def forward(self):
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
    def forward(self, IJ, _):
        # term-wise multiplication
        response = torch.sum(self.V[IJ[:,0]] * self.W[IJ[:,1]], dim=1)
        if self.bias:
            response += self.vb[IJ[:,0]]
            response += self.wb[IJ[:,1]]
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


    def forward(self, IJ, _):

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
        # mat_muls.shape = (vocab, num_senses, num_senses)

        # Calculate the super-dot-product for each sample.
        mat_muls += bias
        exped = torch.exp(mat_muls)
        response = torch.log(exped.sum(dim=2).sum(dim=1))

        return response




WORDS, HEADS, ARCS = 0, 1, 2
class DependencyLearner(nn.Module):

    def __init__(
            self,
            vocab=None,
            covocab=None,
            d=None,
            init=None,
            device=None
        ):

        super(DependencyLearner, self).__init__()
        if init is not None:
            raise NotImplementedError(
                "supplying initial embeddings is not yet supported!")

        # Own it
        self.V_shape = (vocab, d)
        self.W_shape = (covocab, d)
        self.vb_shape = (vocab, 1)
        self.wb_shape = (covocab, 1)
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
        if self.V.shape != self.V_shape or self.W.shape != self.W_shape:
            raise ValueError(
                "Model parameters have initialized with incorrect shape. "
                "Got {}, but expected {}.".format(self.V.shape, self.V_shape)
            )


    def reset(self):
        self.V = nn.Parameter(xavier(self.V_shape, self.device), True)
        self.W = nn.Parameter(xavier(self.W_shape, self.device), True)
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


    def forward(self, batch_id, batch_data):
        """
        When doing a negative sampling approach, normally the negatives
        would be generated by some external loader.  However, in this case
        we are doing something closer to contrastive divergenece, and therefore,
        negative samples need to be generated by the model itself.
        For this reason, in the forward pass, we generate negative samples
        and include them in the forward pass.  This learner needs to agree
        with the loss function in terms of what part of the output of forward
        corresponds to positives, and what part corresponds to negatives.
        """

        positive_sentences, mask = batch_data
        positive_words = positive_sentences[:,WORDS,:]
        positive_head_ids = positive_sentences[:, HEADS, :]
        positive_score = self.calculate_score(
            positive_words, positive_head_ids, mask)

        negative_head_ids = self.do_inference(positive_words, mask)
        negative_score = self.calculate_score(
            positive_words, negative_head_ids, mask)

        return positive_score, negative_score


    def calculate_score(self, words, head_ids, mask):
        """
        sentences should be a 3-d tensor with shape
            (num_sentences, 3, max_length)
        where:
            - num_sentences is the number of sentences in the batch;
            - 3 corresponds to each sentence having three lists of variables:
                word indexes, head-assignments, and arc-types; and
            - max_length is the length of the longest sentence (all other 
                sentences being padded to this length)

        This calculates the tree score for each sentence, give the choice
        of words, head-assignments, and arc-types.

        Returns a 1-d vector with shape (num_sentences,) containing the tree
        score for each sentence.
        """
            
        # sentences is a (batch-size, _, T) tensor, where T is the max
        # sentence length.  _ can be 2 or 3, depending on whether arc-types 
        # are included.
        batch_size, max_sentence_length = words.shape

        # Get some indexing equipment ready.
        gather_along_dim = HEADS
        mask_incl_root = mask.clone()
        mask_incl_root[:,0] = 1

        # Set padding tokens to the unk id.
        words[mask] = 0

        # Set ROOT and padding heads to be ROOT.
        head_ids[mask_incl_root] = 0

        # Get the vocabulary ids for heads by looking head-assignments.
        try:
            heads = torch.gather(words, gather_along_dim, head_ids)
        except RuntimeError:
            import pdb; pdb.set_trace()

        covectors = self.W[words]
        vectors = self.V[heads]
        covector_biases = self.wb[words]
        vector_biases = self.vb[heads]

        scores = (covectors * vectors).sum(2) + vector_biases + covector_biases
        scores[mask_incl_root] = 0
        scores = scores.sum(1)

        return scores


    def parse_energy(self, positives, mask):

        # Need to add ability to sample the root.  Include it at the end
        # of the sentence.  Include root in dictionary and embeddings.
        assert mask.dtype == torch.uint8

        # Drop tags for now
        positives = positives[:,0:2,:] 
        batch_size, _, sentence_length = positives.shape

        # Access the vectors and covectors
        words = positives[:,0,:]
        vectors = self.V[words]
        covectors = self.W[words]


        # Calculate energies
        energies = torch.bmm(covectors, vectors.transpose(1,2))

        # Block out prohibited states:
        #   Don't choose self as a head
        diag_mask = (
            slice(None), range(sentence_length), range(sentence_length))
        energies[diag_mask] = torch.tensor(-float('inf'))

        # Don't choose padding as a head
        reflected_mask = (mask.unsqueeze(1) * mask.unsqueeze(2)).byte()
        energies[1-reflected_mask] = torch.tensor(-float('inf'))
        return energies


    def do_inference(self, positives, mask):

        batch_size, max_sentence_length = positives.shape
        energies = self.parse_energies(positives, mask)

        inferred_heads = torch.zeros(
            (batch_size,max_sentence_length), torch.int64)


        # To be compatible with torch.distributions.Categorical, reshape
        # probabilities such that each word in each sentence gets a row, and
        # that row contains probabilities for selecting among all possible
        # heads in that sentence.  Normalize to obtain a true probability
        # distribution.
        probs = torch.exp(energies)
        probs = probs.view(-1, sentence_length)
        probs[1-mask.reshape(-1),:] = 1
        totals = probs.sum(dim=1, keepdim=True)
        probs = probs / totals

        sample = torch.distributions.Categorical(probs).sample()

        inferred_heads = sample.reshape(-1, sentence_length)
        #idx1 = [i for i in range(batch_size) for j in range(sentence_length)] 
        #idx2 = [j for i in range(batch_size) for j in range(sentence_length)]
        #inferred_heads[idx1,idx2] = sample

        inferred_heads[1-mask] = 0

        return inferred_heads


