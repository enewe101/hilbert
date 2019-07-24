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
            one_sided = 'no',
            init=None,
            device=None
        ):

        super(EmbeddingLearner, self).__init__()

        # Own it
        self.V_shape = (vocab, d)
        self.W_shape = (covocab, d)
        self.R_shape = d
        self.vb_shape = (1, vocab)
        self.wb_shape = (1, covocab)
        self.num_labels = 38 #TODO Fix hardcoded number of labels
        self.kb_shape = (1, self.num_labels)
        self.bias = bias
        self.one_sided = one_sided
        self.device = h.utils.get_device(device)

        # Initialize the model parameters.
        if init is None:
            self.V, self.W, self.R, self.vb, self.wb, self.kb = None, None, None, None, None, None
            self.reset()
        else:
            self.V = nn.Parameter(init[0], True)
            self.W = nn.Parameter(init[1], True)
            self.R = torch.eye(self.R_shape)
            self.vb = None if init[2] is None else nn.Parameter(init[2], True)
            self.wb = None if init[3] is None else nn.Parameter(init[3], True)
            self.kb = nn.Parameter(xavier(
                    (self.kb_shape), self.device).squeeze(), True)
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
        if self.one_sided == 'yes':
            self.W = None
            self.R = None
        elif self.one_sided == 'R':
            self.W = None
            self.R = nn.Parameter(nn.init.xavier_uniform_(
                torch.eye(self.R_shape, device=self.device)))
        elif self.one_sided == 'arc_labels':
            self.W = nn.Parameter(xavier(self.W_shape, self.device), True)
            self.R = nn.Parameter(torch.tensor(
                (self.num_labels,self.R_shape,self.R_shape), device=self.device), True)
            for k in range(self.num_labels):
                self.R[k] = nn.init.xavier_uniform_(
                    torch.eye(self.R_shape, device=self.device))
        else:
            self.W = nn.Parameter(xavier(self.W_shape, self.device), True)
            self.R = None 

        if self.bias:
            print("initialized with bias")
            if self.one_sided == 'arc_labels':
                self.vb = nn.Parameter(xavier(
                    (self.num_labels,self.vb_shape[1]), self.device).squeeze(), True)
                self.wb = nn.Parameter(xavier(
                    (self.num_labels,self.wb_shape[1]), self.device).squeeze(), True)
                self.kb = nn.Parameter(xavier(
                    (self.kb_shape), self.device).squeeze(), True)
            
            else:
                self.vb = nn.Parameter(
                    xavier(self.vb_shape, self.device).squeeze(), True)
                            
                if self.one_sided == 'no':
                    self.wb = nn.Parameter(
                        xavier(self.wb_shape, self.device).squeeze(), True)
                else:
                    self.wb = None


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
        if self.one_sided == 'R':
            response = torch.sum(self.V[IJ[:,0]] * torch.mm(self.V[IJ[:,1]],self.R), dim=1)
            if self.bias:
                response += self.vb[IJ[:,0]]
                response += self.vb[IJ[:,1]]

        elif self.one_sided == 'yes':
            response = torch.sum(self.V[IJ[:,0]] *self.V[IJ[:,1]], dim=1)
            if self.bias:
                response += self.vb[IJ[:,0]]
                response += self.vb[IJ[:,1]]
        
        elif self.one_sided == 'arc_labels':
            response = torch.sum(self.V[IJ[:,0]] * torch.bmm(self.W[IJ[:,1]],self.R[IJ[:,2]]), dim=1)
            if self.bias:
                response += self.vb[IJ[:,0],IJ[:,2]]
                response += self.wb[IJ[:,1],IJ[:,2]]
                response += self.kb[IJ[:,2]]

        else:
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
        # mat_muls.shape = (vocab, num_senses, num_senses)

        # Calculate the super-dot-product for each sample.
        mat_muls += bias
        exped = torch.exp(mat_muls)
        response = torch.log(exped.sum(dim=2).sum(dim=1))

        return response





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
        self.vb_shape = (1, vocab)
        self.wb_shape = (1, covocab)
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


    def forward(self, positives, mask):
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

        # positives is a (batch-size, 2, T) tensor, where T is the max
        # sentence length
        batch_size, _, max_sentence_length = positives.shape
        words = positives[:,0,:]

        idx0 = [
            i for i in range(batch_size) 
            for j in range(max_sentence_length)
        ]
        head_ids = positives[:,1,:].contiguous().view(-1)
        heads = positives[idx0,0,head_ids].view(-1, max_sentence_length)

        import pdb; pdb.set_trace()

        covectors = self.W[words]
        vectors = self.V[heads]

        dotted = covectors * vectors

        # Mask contributions from the root choosing a head
        dotted[:,0,:] = 0
        scores = dotted.sum(1).sum(1)
        return scores


    def negative_sweep(self, positives, mask):

        # Need to add ability to sample the root.  Include it at the end
        # of the sentence.  Include root in dictionary and embeddings.
        assert mask.dtype == torch.uint8

        # Drop tags for now
        positives = positives[:,0:2,:] 
        negatives = torch.zeros_like(positives)
        negatives[:,0,:] = positives[:,0,:]
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
        import pdb; pdb.set_trace()



    def negative_sample(self, positives, mask):

        # Need to add ability to sample the root.  Include it at the end
        # of the sentence.  Include root in dictionary and embeddings.
        assert mask.dtype == torch.uint8

        # Drop tags for now
        positives = positives[:,0:2,:] 
        negatives = torch.zeros_like(positives)
        negatives[:,0,:] = positives[:,0,:]
        batch_size, _, sentence_length = positives.shape

        reflected_mask = (mask.unsqueeze(1) * mask.unsqueeze(2)).byte()
        words = positives[:,0,:]
        vectors = self.V[words]
        covectors = self.W[words]
        energies = torch.bmm(covectors, vectors.transpose(1,2))
        diag_mask = (
            slice(None), range(sentence_length), range(sentence_length))
        energies[diag_mask] = torch.tensor(-float('inf'))
        energies[1-reflected_mask] = torch.tensor(-float('inf'))

        unnormalized_probs = torch.exp(energies)
        unnormalized_probs_2d = unnormalized_probs.view(-1, sentence_length)
        unnormalized_probs_2d[1-mask.reshape(-1),:] = 1
        totals = unnormalized_probs_2d.sum(dim=1, keepdim=True)
        probs = unnormalized_probs_2d / totals

        sample = torch.distributions.Categorical(probs).sample()
        reshaped_sample = sample.reshape(-1, 1, sentence_length)

        idx1 = [i for i in range(batch_size) for j in range(sentence_length)]
        idx2 = [j for i in range(batch_size) for j in range(sentence_length)]

        negatives[idx1,1,idx2] = sample

        negatives[:,1,:][1-mask] = 0

        return negatives










