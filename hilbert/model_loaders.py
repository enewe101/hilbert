import hilbert as h
import torch




class PPMILoader(BigramLoaderBase):

    def _load(self, preloaded):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        shard_id, bigram_data, unigram_data = preloaded
        bigram_data = tuple(tensor.to(device) for tensor in bigram_data)
        M = h.corpus_stats.calc_PMI(bigram_data)
        M = torch.clamp(M, min=0)
        return shard_id, {'M':M}

    def describe(self):
        return 'PPMI Sharder\n' + super(PPMILoader, self).describe()



class GloveLoader(BigramLoaderBase):

    def __init__(
        self, bigram_path, sector_factor, shard_factor, num_loaders,
        X_max=100.0, alpha=0.75, t_clean_undersample=None,
        alpha_unigram_smoothing=None, queue_size=1, device=None, verbose=True
    ):
        self.X_max = float(X_max)
        self.alpha = alpha
        super(GloveLoader, self).__init__(
            bigram_path, sector_factor, shard_factor, num_loaders,
            t_clean_undersample=t_clean_undersample,
            alpha_unigram_smoothing=alpha_unigram_smoothing,
            queue_size=queue_size, device=device, verbose=verbose
        )

    def _load(self, preloaded):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        shard_id, bigram_data, unigram_data = preloaded
        Nxx, Nx, Nxt, N = tuple(tensor.to(device) for tensor in bigram_data)
        weights = (Nxx / self.X_max).pow(self.alpha)
        weights = torch.clamp(weights, max=1.)
        M = torch.log(Nxx)
        Nxx_is_zero = (Nxx==0)
        M[Nxx_is_zero] = 0
        weights[Nxx_is_zero] = 0
        weights = weights * 2
        return shard_id, {'M':M, 'weights':weights}

    def describe(self):
        s =  'GloVe Sharder\n'
        s += '\tX_max = {}\n'.format(self.X_max)
        s += '\talpha = {}\n'.format(self.alpha)
        s += super(GloveLoader, self).describe()
        return s


class Word2vecLoader(BigramLoaderBase):

    def __init__(
        self, bigram_path, sector_factor, shard_factor, num_loaders, k=15,
        t_clean_undersample=None, alpha_unigram_smoothing=None, queue_size=1,
        device=None, verbose=True
    ):
        super(Word2vecLoader, self).__init__(
            bigram_path, sector_factor, shard_factor, num_loaders,
            t_clean_undersample=t_clean_undersample,
            alpha_unigram_smoothing=alpha_unigram_smoothing,
            queue_size=queue_size, device=device, verbose=verbose
        )
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        device = device or h.CONSTANTS.MATRIX_DEVICE
        self.k = torch.tensor(k, device=device, dtype=dtype)

    def _load(self, preloaded):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        shard_id, bigram_data, unigram_data = preloaded
        Nxx, Nx, Nxt, N = tuple(tensor.to(device) for tensor in bigram_data)
        uNx, uNxt, uN = tuple(tensor.to(device) for tensor in unigram_data)
        N_neg = self.negative_sample(Nxx, Nx, uNxt, uN, self.k)
        return shard_id, {'Nxx':Nxx, 'N_neg': N_neg}


    @staticmethod
    def negative_sample(Nxx, Nx, uNxt, uN, k):
        return k * (Nx - Nxx) * (uNxt / uN)


    def describe(self):
        s = 'Word2Vec Sharder\n'
        s += '\tk = {}\n'.format(self.k)
        s += super(Word2vecLoader, self).describe()
        return s



class MaxLikelihoodLoader(BigramLoaderBase):
    def _load(self, preloaded):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        shard_id, bigram_data, unigram_data = preloaded
        Nxx, Nx, Nxt, N = tuple(tensor.to(device) for tensor in bigram_data)
        Pxx_data = Nxx / N
        Pxx_independent = (Nx / N) * (Nxt / N)
        return shard_id, {
            'Pxx_data':Pxx_data, 'Pxx_independent':Pxx_independent}

    def describe(self):
        return 'Max Likelihood Sharder\n' + super(
            MaxLikelihoodLoader, self).describe()




class MaxPosteriorLoader(BigramLoaderBase):
    def _load(self, preloaded):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        shard_id, bigram_data, unigram_data = preloaded
        Nxx, Nx, Nxt, N = tuple(tensor.to(device) for tensor in bigram_data)
        Pxx_independent = (Nx / N) * (Nxt / N)
        exp_mean, exp_std =  h.corpus_stats.calc_exp_pmi_stats((Nxx,Nx,Nxt,N))
        alpha, beta = h.corpus_stats.calc_prior_beta_params(
            (Nxx, Nx, Nxt, N), exp_mean, exp_std, Pxx_independent)
        N_posterior = N + alpha + beta - 1
        Pxx_posterior = (Nxx + alpha) / N_posterior
        return shard_id, {
            'N':N, 'N_posterior':N_posterior, 'Pxx_posterior': Pxx_posterior,
            'Pxx_independent': Pxx_independent
        }

    def describe(self):
        return 'Max Posterior Probability Sharder\n' + super(
            MaxPosteriorLoader, self).describe()


class KLLoader(BigramLoaderBase):
    def _load(self, preloaded):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        shard_id, bigram_data, unigram_data = preloaded
        Nxx, Nx, Nxt, N = tuple(tensor.to(device) for tensor in bigram_data)
        Pxx_independent = (Nx / N) * (Nxt / N)
        exp_mean, exp_std =  h.corpus_stats.calc_exp_pmi_stats((Nxx,Nx,Nxt,N))
        alpha, beta = h.corpus_stats.calc_prior_beta_params(
            (Nxx,Nx,Nxt,N), exp_mean, exp_std, Pxx_independent)
        N_posterior = N + alpha + beta - 1
        a = Nxx + alpha
        b = N - Nxx + beta
        digamma_a = torch.digamma(a) - torch.digamma(a+b)
        digamma_b = torch.digamma(b) - torch.digamma(a+b)
        return shard_id, {
                'digamma_a': digamma_a, 'digamma_b': digamma_b, 'N':N,
                'N_posterior':N_posterior, 'Pxx_independent': Pxx_independent,
        }

    def describe(self):
        return 'KL Sharder\n' + super(KLLoader, self).describe()



