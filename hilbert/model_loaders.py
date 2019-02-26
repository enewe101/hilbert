import hilbert as h
import torch


class ModelShardLoader(object):
    """
    Base class for any LoaderModel that implements the common functionality,
    being, iteration over the preloaded shards.
    """

    def __init__(self, bigram_preloader, verbose=True):
        self.preloader = bigram_preloader
        self.device = bigram_preloader.device
        self.verbose = verbose

        # these Nones are utilized in the iterator pattern
        self.preloaded_shards = None
        self.crt_shard_idx = None
        self.preload_all_shards() # fill 'er up!


    def __iter__(self):
        self.crt_shard_idx = -1
        return self


    def __next__(self):
        self.crt_shard_idx += 1

        if self.crt_shard_idx >= len(self.preloaded_shards):
            raise StopIteration

        return self._load(self.preloaded_shards[self.crt_shard_idx])


    def describe(self):
        return self.preloader.describe()


    def preload_all_shards(self):
        self.preloaded_shards = []

        if self.verbose:
            print('Preloading all shards...')

        for preload_data in self.preloader.preload_iter():
            self.preloaded_shards.append(preload_data)

        if self.verbose:
            print('Preloading complete!')


    def _load(self, preloaded):
        raise NotImplementedError('Subclasses must extend `_load`!')



### Below we have specific loaders for every model.
# Each deals with the Nxx data in different ways;
# e.g., some do PMI, some do PMI - ln k, etc.

class PPMILoaderModel(ModelShardLoader):

    def _load(self, preloaded):
        shard_id, bigram_data, unigram_data = preloaded
        bigram_data = tuple(tensor.to(self.device) for tensor in bigram_data)
        M = h.corpus_stats.calc_PMI(bigram_data)
        M = torch.clamp(M, min=0)
        return shard_id, {'M': M}

    def describe(self):
        return 'PPMI Sharder\n' + super(PPMILoaderModel, self).describe()



class GloveLoaderModel(ModelShardLoader):

    def __init__(self, bigram_preloader, verbose, X_max=100.0, alpha=0.75):
        super(GloveLoaderModel, self).__init__(bigram_preloader, verbose)
        self.X_max = float(X_max)
        self.alpha = alpha

    def _load(self, preloaded):
        shard_id, bigram_data, unigram_data = preloaded
        Nxx, Nx, Nxt, N = tuple(tensor.to(self.device) for tensor in bigram_data)
        weights = (Nxx / self.X_max).pow(self.alpha)
        weights = torch.clamp(weights, max=1.)
        M = torch.log(Nxx)
        Nxx_is_zero = (Nxx==0)
        M[Nxx_is_zero] = 0
        weights[Nxx_is_zero] = 0
        weights = weights * 2
        return shard_id, {'M': M, 'weights': weights}

    def describe(self):
        s =  'GloVe Sharder\n'
        s += '\tX_max = {}\n'.format(self.X_max)
        s += '\talpha = {}\n'.format(self.alpha)
        s += super(GloveLoaderModel, self).describe()
        return s



class Word2VecLoaderModel(ModelShardLoader):

    def __init__(self, bigram_preloader, verbose, k=15):
        super(Word2VecLoaderModel, self).__init__(bigram_preloader, verbose)
        self.k = torch.tensor(k, device=self.device, dtype=h.CONSTANTS.DEFAULT_DTYPE)

    def _load(self, preloaded):
        shard_id, bigram_data, unigram_data = preloaded
        Nxx, Nx, Nxt, N = tuple(tensor.to(self.device) for tensor in bigram_data)
        uNx, uNxt, uN = tuple(tensor.to(self.device) for tensor in unigram_data)
        N_neg = self.negative_sample(Nxx, Nx, uNxt, uN, self.k)
        return shard_id, {'Nxx': Nxx, 'N_neg': N_neg}

    def describe(self):
        s = 'Word2Vec Sharder\n'
        s += '\tk = {}\n'.format(self.k)
        s += super(Word2VecLoaderModel, self).describe()
        return s

    @staticmethod
    def negative_sample(Nxx, Nx, uNxt, uN, k):
        return k * (Nx - Nxx) * (uNxt / uN)



class MaxLikelihoodLoaderModel(ModelShardLoader):

    def _load(self, preloaded):
        shard_id, bigram_data, unigram_data = preloaded
        Nxx, Nx, Nxt, N = tuple(tensor.to(self.device) for tensor in bigram_data)
        Pxx_data = Nxx / N
        Pxx_independent = (Nx / N) * (Nxt / N)
        return shard_id, {'Pxx_data': Pxx_data, 'Pxx_independent': Pxx_independent}

    def describe(self):
        return 'Max Likelihood Sharder\n' + super(
            MaxLikelihoodLoaderModel, self).describe()



class MaxPosteriorLoaderModel(ModelShardLoader):

    def _load(self, preloaded):
        shard_id, bigram_data, unigram_data = preloaded
        Nxx, Nx, Nxt, N = tuple(tensor.to(self.device) for tensor in bigram_data)
        Pxx_independent = (Nx / N) * (Nxt / N)
        exp_mean, exp_std =  h.corpus_stats.calc_exp_pmi_stats((Nxx,Nx,Nxt,N))
        alpha, beta = h.corpus_stats.calc_prior_beta_params(
            (Nxx, Nx, Nxt, N), exp_mean, exp_std, Pxx_independent)
        N_posterior = N + alpha + beta - 1
        Pxx_posterior = (Nxx + alpha) / N_posterior
        return shard_id, {
            'N': N, 'N_posterior': N_posterior, 'Pxx_posterior': Pxx_posterior,
            'Pxx_independent': Pxx_independent
        }

    def describe(self):
        return 'Max Posterior Probability Sharder\n' + super(
            MaxPosteriorLoaderModel, self).describe()



class KLLoaderModel(ModelShardLoader):

    def _load(self, preloaded):
        shard_id, bigram_data, unigram_data = preloaded
        Nxx, Nx, Nxt, N = tuple(tensor.to(self.device) for tensor in bigram_data)
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
                'digamma_a': digamma_a, 'digamma_b': digamma_b, 'N': N,
                'N_posterior': N_posterior, 'Pxx_independent': Pxx_independent,
        }

    def describe(self):
        return 'KL Sharder\n' + super(KLLoaderModel, self).describe()



