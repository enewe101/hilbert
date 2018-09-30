
###
#
#   I have not been able to get this working in a way that will hae reasonable
#   performance.
#
###

def calc_M_neg_samp(
    cooc_stats,
    k_samples=15,
    k_weight=None,
    alpha=0.75,
    shard_num=None
):
    """
    Returns a PMI-like matrix based on negative sampling, simulating sampling 
    used in Mikolov's word2vec.
    ``k_samples`` (int): number of times noise distribution is sampled, per
        corpus sample.  Default is to sample 15 times from noise, for every 1
        sample from the corpus, because it was found to be good in Mikolov's
        2013 paper.
    ``k_weight`` (float): total weight of noise distribution.  If this isn't set
        then the total weight is just equal to the number of negative samples
        per corpus sample.  But if the value is provided, then that's what 
        is used as the weight.
    ``alpha`` (float): exponent applied to unigram distribution, which distorts
        it.  As alpha becomes < 1, it makes the distribution flatter.
        Conceptually, alpha is inverse temperature.
    ``shard_num`` (int): used to seed randomness so that the same randomness
        is used for a given shard, keeping the values in M stable, as if they
        had been sampled once and for all at the beginning (that would take
        too much memory)

    When sharding is used (by passing a loaded shard as cooc_stats object),
    it isn't a perfect simulation---variance in the negative samples is a little
    lower than in the original word2vec sampling, because the total number
    of samples within each shard is correct, a constraint that normally doesn't
    hold.  It is probably negligible.
    """
    # Unpack args and apply defaults.
    k_weight = k_samples if k_weight is None else k_weight
    Nxx, Nx, Nxt, N = cooc_stats

    # Calculate the unigram distribution
    maybe_distorted_Nx = Nx if alpha is None else Nx**alpha
    maybe_distorted_N = N if alpha is None else torch.sum(maybe_distorted_Nx)
    px = maybe_distorted_Nx / maybe_distorted_N

    # We are about to draw samples.  Notice that we're drawing samples
    # within the calculation of an M shard.  This isn't ideal.
    #
    # We can't sample the negative samples up front, because they result in a 
    # dense array that literally won't fit in memory (unlike the sparse Nxx).
    # The best I can do right now, is to sample it freshly each time, but make
    # sure that the seed is always the same for a given shard, so that we are
    # using the same random samples each time the same shard comes up.
    if shard_num is not None:
        np.random.seed(shard_num)

    # Samples are going to be done by numpy, so we have no choice but to 
    # do this in CPU world.  
    #
    # Also, unrelated, we need to pad px with one buffer parameter. It will
    # account for all the probability mass of drawing a sample outside the
    # shard.  This will accumulate samples that we'll discard, by shaving that
    # column of the samples array.
    px = np.append(px.cpu().numpy(), 0)

    return px

    # Draw negative samples
    samples = torch.tensor(
        _sample_multi_multinomial(k_samples * Nx, px),
        dtype=h.CONSTANTS.DEFAULT_DTYPE,
        device=h.CONSTANTS.MATRIX_DEVICE
    )[:,:-1]    # Shave off the element containing out-of-shard samples

    # Set negative sample wieght, if provided Note that if k_weight is None,
    # effective weight is k_samples
    if k_weight is not None:
        samples = k_weight * (samples / k_samples)
    # Return the negative sample objective as defined.
    return torch.log(Nxx) - torch.log(samples)


# This remains implementated using numpy instead of torch, because numpy 
# provides much faster multinomial samples.
def _sample_multi_multinomial(kNx, px):
    kNx = kNx.reshape(-1)
    px = px.reshape(-1)
    samples = np.zeros((len(kNx), len(kNx)))
    for i in range(len(kNx)):
        samples[i,:] = np.random.multinomial(kNx[i], px)
    return samples

# This torch implementation is too slow for some reason.  Use the numpy 
# implementation.
def _sample_multi_multinomial_torch(kNx, px):
    kNx = kNx.view(-1)
    px = px.view(-1)
    samples = torch.empty((len(kNx), len(kNx)))
    for i in range(len(kNx)):
        sampler = torch.distributions.multinomial.Multinomial(
            int(kNx[i].item()), px)
        samples[i,:] = sampler.sample()
    return samples


# Interesting idea, but didn't make it any faster.
#
## This torch implementation is too slow for some reason.  Use the numpy 
## implementation.
#def _sample_multi_multinomial_torch_alt(kNx, px):
#    kNx = kNx.view(-1,1)
#    px = px.view(-1)
#    samples = torch.empty((len(kNx), len(kNx)))
#    sampler = torch.distributions.Categorical(px)
#    for i in range(len(kNx)):
#        placer = torch.arange(px.shape[0], device='cuda').view(-1,1)
#        sample = sampler.sample(torch.Size(kNx[i]))
#        one_hot_sample = sample == placer
#        multinomial_sample = torch.sum(one_hot_sample, dim=1)
#        samples[i,:] = multinomial_sample
#    return samples
