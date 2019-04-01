import abc
import hilbert as h
import torch
import torch.distributions as dist
from hilbert.generic_interfaces import Describable


class BatchPreloader(Describable):
    """
    Abstract class for batch preloading.
    """

    @abc.abstractmethod
    def __init__(
            self, bigram_path, *args,
            t_clean_undersample=None,
            alpha_unigram_smoothing=None, **kwargs
        ):

        self.bigram_path = bigram_path
        self.t_clean_undersample = t_clean_undersample
        self.alpha_unigram_smoothing = alpha_unigram_smoothing


    @abc.abstractmethod
    def preload_iter(self, *args, **kwargs):
        return


    def prepare(self, preloaded):
        return preloaded


    @abc.abstractmethod
    def describe(self):
        s = '\tbigram_path = {}\n'.format(self.bigram_path)
        s += '\tt_clean_undersample = {}\n'.format(self.t_clean_undersample)
        s += '\talpha_unigram_smoothing = {}\n'.format(
            self.alpha_unigram_smoothing)
        return s




"""
Class for dense matrix factorization data loading.
"""
class DenseShardPreloader(BatchPreloader):

    def __init__(
        self, bigram_path, sector_factor, shard_factor,
        t_clean_undersample=None,
        alpha_unigram_smoothing=None,
    ):
        """
        Base class for more specific loaders `BigramLoader` yields tensors 
        representing shards of text cooccurrence data.  Each shard has unigram
        and bigram data, for words and word-pairs, along with totals.

        bigram data:
            `Nxx`   number of times ith word seen with jth word.
            `Nx`    marginalized (summed) counts: num pairs containing ith word
            `Nxt`   marginalized (summed) counts: num pairs containing jth word
            `N`     total number of pairs.

            Note: marginalized counts aren't equal to frequency of the word,
            one word occurrence means participating in ~2 x window-size number
            of pairs.

        unigram data `(uNx, uNxt, uN)`
            `uNx`   Number of times word i occurs.
            `uNxt`  Number of times word j occurs.
            `uN`    total number of words

            Note: Due to unigram-smoothing (e.g. in w2v), uNxt may not equal
            uNx.  In w2v, one gets smoothed, the other is left unchanged (both
            are needed).
        """
        super(DenseShardPreloader, self).__init__(
            bigram_path, t_clean_undersample=t_clean_undersample,
            alpha_unigram_smoothing=alpha_unigram_smoothing
        )
        self.sector_factor = sector_factor
        self.shard_factor = shard_factor
        self.bigram_sector = None


    def preload_iter(self, *args, **kwargs):
        super(DenseShardPreloader, self).preload_iter(*args, **kwargs)

        for i, sector_id in enumerate(h.shards.Shards(self.sector_factor)):
            print('loading sector {}'.format(i))

            # Read the sector of bigram data into memory, and transform
            # distributions as desired.
            self.bigram_sector = h.bigram.BigramSector.load(
                self.bigram_path, sector_id)

            self.bigram_sector.apply_w2v_undersampling(
                self.t_clean_undersample)

            self.bigram_sector.apply_unigram_smoothing(
                self.alpha_unigram_smoothing)

            # Start yielding cRAM-preloaded shards
            for shard_id in h.shards.Shards(self.shard_factor):

                bigram_data = self.bigram_sector.load_relative_shard(
                    shard=shard_id, device='cpu')

                unigram_data = self.bigram_sector.load_relative_unigram_shard(
                    shard=shard_id, device='cpu')

                yield shard_id * sector_id, bigram_data, unigram_data
        return


    def describe(self):
        s = super(DenseShardPreloader, self).describe()
        s += 'Dense Preloader\n'
        s += '\tsector_factor = {}\n'.format(self.sector_factor)
        s += '\tshard_factor = {}\n'.format(self.shard_factor)
        return s



"""
Class for dense matrix factorization data loading.
"""
class SampleMaxLikelihoodLoader:

    def __init__(self, bigram_path, sector_factor):
        """
        """
        self.bigram_path = bigram_path
        self.sector_factor = sector_factor
        self.bigram_sector = None


    def accumulate_statistics(self):

        # Go though each sector and accumulate all of the non-zero data
        # into a single sparse tensor representation.
        self.data = torch.tensor([], dtype=torch.float32)
        self.I = torch.tensor([], dtype=torch.long)
        self.J = torch.tensor([], dtype=torch.long)
        for sector_id in h.shards.Shards(self.sector_factor):

            # Read the sector, and get the statistics in sparse COO-format
            sector = h.bigram.BigramSector.load(
                self.bigram_path, sector_id
            ).Nxx.tocoo()
            assert not any(sector.data == 0)

            # Tensorfy the data, and the row and column indices
            add_Nxx = torch.tensor(sector.data, dtype=torch.float32)
            add_i_idxs = torch.tensor(sector.row, dtype=torch.long)
            add_j_idxs = torch.tensor(sector.col, dtype=torch.long)

            # Adjust the row and column indices to account for sharding
            add_i_idxs = add_i_idxs * sector_id.step + sector_id.i
            add_j_idxs = add_j_idxs * sector_id.step + sector_id.j

            # Concatenate
            self.data = torch.cat((self.data, add_Nxx))
            self.I = torch.cat((self.I, add_i_idxs))
            self.J = torch.cat((self.J, add_j_idxs))





"""
Class for smart compressed data loading & iteration.
"""
class SparsePreloader(BatchPreloader):

    def __init__(
            self, bigram_path,
            zk=1000,
            t_clean_undersample=None,
            alpha_unigram_smoothing=None,
            include_unigram_data=False,
            filter_repeats=False,
            device=None
        ):
        """
        :param bigram_path:
        :param zk:
        :param t_clean_undersample:
        :param alpha_unigram_smoothing:
        :param include_unigram_data:
        :param filter_repeats:
        :param device:
        """

        super(SparsePreloader, self).__init__(
            bigram_path, t_clean_undersample=t_clean_undersample,
            alpha_unigram_smoothing=alpha_unigram_smoothing,
        )
        self.zk = zk # max number of z-samples to draw
        self.include_unigram_data = include_unigram_data
        self.filter_repeats = filter_repeats
        self.device = device

        # put the other attributes in init for transparency
        self.n_nonzeros = 0
        self.n_batches = 0
        self.z_sampler = None
        self.sparse_nxx = None
        self.Nx, self.Nxt, self.N = None, None, None
        self.uNx, self.uNxt, self.uN = None, None, None


    def preload_iter(self, *args, **kwargs):
        super(SparsePreloader, self).preload_iter(*args, **kwargs)

        bigram = h.bigram.BigramBase.load(self.bigram_path, marginalize=False)

        # Number of nonzero elements
        self.n_nonzeros = bigram.Nxx.nnz

        # Number of batches, equivalent to vocab size
        self.n_batches = len(bigram.Nxx.data)
        self.z_sampler = ZedSampler(
            self.n_batches, self.device, self.zk,
            filter_repeats=self.filter_repeats
        )

        # Iterate over each row index in the sparse matrix
        self.sparse_nxx = []

        # Iterate over each row in the sparse matrix and get marginals
        self.Nx = torch.zeros((self.n_batches,), device=self.device)
        self.Nxt = torch.zeros((self.n_batches,), device=self.device)

        for i in range(len(bigram.Nxx.data)):
            js_tensor = torch.LongTensor(bigram.Nxx.rows[i]).to(self.device)
            nijs_tensor = torch.FloatTensor(bigram.Nxx.data[i]).to(self.device)

            # put in the marginal sums!
            self.Nx[i] = nijs_tensor.sum()
            self.Nxt[js_tensor] += nijs_tensor

            # store the implicit sparse matrix as a series
            # of tuples, J-indexes, then Nij values.
            self.sparse_nxx.append((js_tensor, nijs_tensor,))
            bigram.Nxx.rows[i].clear()
            bigram.Nxx.data[i].clear()

        # now we need to store the other statistics
        self.N = self.Nx.sum().to(self.device)

        if self.include_unigram_data:
            self.uNx = bigram.uNx.flatten().to(self.device)
            self.uNxt = bigram.uNxt.flatten().to(self.device)
            self.uN = bigram.uN.to(self.device)

        # implicitly store the preloaded batches
        return range(self.n_batches)


    def prepare(self, preloaded):
        # alpha-samples are the js
        i, js = preloaded, self.sparse_nxx[preloaded][0]

        # zed-samples
        z_js, z_nijs = self.z_sampler.z_sample(js)
        all_js = torch.cat((js, z_js))
        all_nxx = torch.cat((self.sparse_nxx[preloaded][1], z_nijs))

        # prepare the data for learning
        bigram_data = (all_nxx,
                       self.Nx[i],
                       self.Nxt[all_js],
                       self.N)

        # fill up unigram data only if necessary
        unigram_data = None
        if self.include_unigram_data:
            unigram_data = (self.uNx[i], self.uNxt[all_js], self.uN)

        batch_id = (i, all_js)
        return batch_id, bigram_data, unigram_data


    def describe(self):
        s = super(SparsePreloader, self).describe()
        s += 'Sparse preloader\n'
        s += '\tzk = {}\n'.format(self.zk)
        s += '\tfilter repeats = {}\n'.format(self.filter_repeats)
        s += '\tinclude unigram data = {}\n'.format(self.include_unigram_data)
        return s



"""
Utility class for Z-sampling on the GPU.
"""
class ZedSampler(object):

    def __init__(self, upper_limit, device, max_z_samples=1000, filter_repeats=False):
        self.max_z_samples = max_z_samples
        self.upper_limit = upper_limit
        self.device = device
        self.zeds = torch.zeros((max_z_samples,), device=device)
        self.filter_repeats = filter_repeats


    def z_sample(self, a_samples):
        """
        We can expect approximately 1% of the uniform random samples to
        be repeats from the alpha-samples. If you don't mind your loss to be
        noisy (perhaps it could even work as regularization), and you want
        to draw samples at constant time, then pass filter_repeats=False.

        Otherwise, it averages at about .18s to draw 10,000
        Z-samples with filter_repeats=True, and only .018s for 1,000.
        Both of these are much much too time consuming, so do not do this.
        (E.g., if vocab is O(10^5), then drawing 10,000 will require
        approximately 5 hours per epoch just for doing this.)

        If vocab size is O(10^5) and we are drawing O(10^3) samples, we can
        be almost certain that there will be no repeats, so use
        filter_repeats=False when max_z_samples=1000.

        :param a_samples: tensor of the Nij > 0 samples that we are comparing against
        :param filter_repeats: parameter of whether or not we want to filter
            out the repeated samples, if we do we are true to the real loss.
        :return: tensor of Z-samples with Nij=0
                (99.99% chance that Nij=0 if filter_repeats=False)
        """

        # sort the samples and grab the values, [0] (args are in [1]
        num_zeds = min(len(a_samples), self.max_z_samples)
        samples = torch.randint(self.upper_limit,
                                device=self.device,
                                size=(num_zeds,),
                                ).long()

        if not self.filter_repeats:
            return samples, self.zeds[:num_zeds]
        else:
            samples = samples.sort()[0]

        # filter so we don't have repeats, taking advantage of the fact
        # that both sets are sorted.
        bits = torch.ones((len(samples),), device=samples.device, dtype=torch.uint8)
        a_idx = 0 # torch.LongTensor([0], device=samples.device)[0]
        s_idx = 0 # torch.LongTensor([0], device=samples.device)[0]

        # TODO: fix algorithm so that it properly handles when the same value is repeated.
        try:
            while True:

                while samples[s_idx] != a_samples[a_idx]:

                    while samples[s_idx] < a_samples[a_idx]:
                        s_idx += 1

                    while a_samples[a_idx] < samples[s_idx]:
                        a_idx += 1

                bits[s_idx] = 0
                s_idx += 1
                a_idx += 1

        except IndexError:
            pass

        good_samples = samples[bits.nonzero().flatten()]
        return good_samples, self.zeds[:len(good_samples)]



