import hilbert as h
import torch
import torch.nn as nn
import numpy as np

# noinspection PyCallingNonCallable
def xavier(shape, device):
    return nn.init.xavier_uniform_(torch.zeros(shape, device=device))

PAD = h.CONSTANTS.PAD


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
    def forward(self, shard, _):
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
            num_negative_samples=1,
            device=None
        ):

        super(DependencyLearner, self).__init__()
        if init is not None:
            raise NotImplementedError(
                "supplying initial embeddings is not yet supported!")
        is_nonnegative_int = (
            num_negative_samples > 0 and 
            isinstance(num_negative_samples, int)
        )
        if not is_nonnegative_int:
            raise ValueError(
                "num_negative_samples must be a nonnegative integer.  "
                "got {}.".format(num_negative_samples)
            )

        # Own it
        self.V_shape = (vocab, d)
        self.W_shape = (covocab, d)
        self.vb_shape = (vocab, 1)
        self.wb_shape = (covocab, 1)
        self.device = h.utils.get_device(device)
        self.num_negative_samples = num_negative_samples

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
        and include them in the response sent to the loss function.  This
        learner needs to agree with the loss function in terms of what part of
        the output of forward corresponds to positives, and what part
        corresponds to negatives.
        """

        positive_sentences, mask = batch_data
        positive_words = positive_sentences[:, WORDS,:]
        positive_head_ids = positive_sentences[:, HEADS, :]
        positive_score = self.calculate_score(
            positive_words, positive_head_ids, mask)

        # Draw one or more negative samples, and calculate the (mean) parse
        # score for the negative sample(s).
        negative_score = torch.zeros_like(positive_score)
        for neg_samp_num in range(self.num_negative_samples):
            negative_head_ids = self.do_inference(positive_words, mask)
            negative_score = negative_score + self.calculate_score(
                positive_words, negative_head_ids, mask)
        negative_score = negative_score / self.num_negative_samples

        return positive_score, negative_score


    def calculate_score(self, words, head_ids, mask):
        """
        words is a 2-d tensor with shape
            (num_sentences, max_length)
        where:
            - num_sentences is the number of sentences in the batch;
            - max_length is the length of the longest sentence (all other 
                sentences being padded to this length)

        head_ids and mask have a similar shape.  For each sentence, head_ids
        stores the index of the head of that word.  For each sentence, the 
        mask indicates where padding has been placed near the end of the 
        sentence.  The [ROOT] is not masked, but it can be easily masked
        because it is always token 0 in each sentence.

        This calculates the tree score for each sentence, give the choice
        of words, and, head-assignments.

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

        # TODO: maybe PAD should jus be zero?  then we would not have to do 
        # the next two steps.

        # Set padding tokens to the unk id.
        words = words.clone()
        words[mask] = 0

        # Set ROOT and padding heads to be ROOT.
        head_ids[mask_incl_root] = 0

        # Get the vocabulary ids for heads by looking head-assignments.
        heads = torch.gather(words, gather_along_dim, head_ids)

        covectors = self.W[words]
        vectors = self.V[heads]
        covector_biases = self.wb[words]
        vector_biases = self.vb[heads]

        scores = (covectors * vectors).sum(2) + vector_biases + covector_biases
        scores[mask_incl_root] = 0
        scores = scores.sum(1)

        return scores


    def parse_energy(self, words, mask):

        # Need to add ability to sample the root.  Include it at the end
        # of the sentence.  Include root in dictionary and embeddings.
        assert mask.dtype == torch.uint8

        # Drop tags for now
        batch_size, sentence_length = words.shape

        # Mask the head and padding
        words = words.clone()
        words[mask] = 0

        # Access the vectors and covectors
        V = self.V[words]
        vb = self.vb[words].unsqueeze(1)
        W = self.W[words]
        wb = self.wb[words].unsqueeze(2)

        # Calculate energies
        energies = torch.bmm(W, V.transpose(1,2)) + vb + wb

        # Tokens cannot choose themselves as a head
        diag_mask = (
            slice(None), range(sentence_length), range(sentence_length))
        energies[diag_mask] = torch.tensor(-float('inf'))

        # Don't choose padding as a head
        reflected_mask = (mask.unsqueeze(1) | mask.unsqueeze(2)).byte()
        energies[reflected_mask] = torch.tensor(-float('inf'))

        # [ROOT] should not choose a head
        energies[:,0,:] = -float('inf')
        return energies


    def parse_probs(self, words, mask):
        batch_size, sentence_length = words.shape
        energy = self.parse_energy(words, mask)
        probs = torch.exp(energy)

        # The root shouldn't choose a head.  However, we have to put non-zero
        # probability somewhere so that probs is compatible with being fed
        # as inputs to a sampler.  Therefore, we assign probability one to 
        # the root selecting itself as a head...
        probs[:,0,:] = 1

        # ... and to padding selecting [ROOT] as head.
        padded_modifiers = mask.unsqueeze(2).expand(-1, -1, sentence_length)
        probs[padded_modifiers] = 1

        # Normalize the distribution for each modifier picking among heads.
        totals = probs.sum(2, keepdim=True)
        probs = probs / totals

        return probs


    def do_inference(self, words, mask, enforce_constraints=True):


        batch_size, sentence_length = words.shape
        probs = self.parse_probs(words, mask)
        probs = probs.view(-1, sentence_length)
        totals = probs.sum(dim=1, keepdim=True)
        probs = probs / totals
        sample = torch.distributions.Categorical(probs).sample()

        inferred_heads = sample.reshape(-1, sentence_length)
        inferred_heads[mask] = 0
        inferred_heads[:,0] = 0

        if enforce_constraints:
            inferred_heads = self.enforce_constraints(inferred_heads, probs)

        return inferred_heads


    def enforce_constraints(self, heads, probs):
        """
        Detect when a parse tree contains cycles.  For each sentence containing
        a cycle, select a single implicated node and resample it.
        Then, re-analyze for cycles, and repeat until no cycles are left.

        For now, I am not enforcing the constraint that there should be exactly
        one token that selects the [ROOT].  I am assuming that a [ROOT], like a 
        verb, will find any obligatory modifiers (in the case of [ROOT],
        exactly one).
        """
        batch_size, sentence_length = heads.shape
        modifiers_to_resample = self.detect_cycles(heads)
        while torch.any(modifiers_to_resample>0):
            nz = torch.nonzero(modifiers_to_resample).view(-1)
            probs_indices = nz * sentence_length + modifiers_to_resample[nz]
            re_probs = probs[probs_indices]
            resampled = torch.distributions.Categorical(re_probs).sample()
            heads[nz, modifiers_to_resample[nz]] = resampled
            modifiers_to_resample = self.detect_cycles(heads)
        return heads


    def detect_cycles(self, heads):

        # Follow link from each modifier toward its head repeatedly.
        # Once this has been done sqrt(2*sentence_length) times, it is 
        # guaranteed that any modifier not leading to a cycle will have reached
        # root.
        batch_size, sentence_length = heads.shape
        max_steps = int(np.ceil(np.sqrt(2*sentence_length)))
        parents = heads
        for i in range(max_steps):
            parents = parents.gather(1, parents)

        implicated = torch.nonzero(parents)

        # Randomly permute the list of tokens that are implicated
        num_cases, _ = implicated.shape
        implicated = implicated[torch.randperm(num_cases)]

        # For each sentence, write the implicated tokens into a single
        # slot.  Only the last one in the permutation survives.  This in effect
        # randomly selects one cycle-implicated token per sentence
        chosen_implicated = torch.zeros(batch_size, dtype=torch.int64)
        chosen_implicated[implicated[:,0]] = (
            parents[implicated[:,0],implicated[:,1]]
        )
        return chosen_implicated


