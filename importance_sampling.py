import numpy as np
import scipy
import hilbert as h
import torch
import pytorch_categorical

cooc_path = '/home/jingyihe/Documents/cooc/5w-dynamic-v10k'
if __name__ == '__main__':
    res = []
    factory = h.factories.build_mle_sample_solver(cooc_path, batch_size=30)
    sample_loader = factory.loader
    Q_sampler_i = sample_loader.negative_sampler
    Q_sampler_j = sample_loader.negative_sampler_t
    np.random.seed(1)
    list_of_seed = np.random.random_integers(1,200,100)

    for i in range(100):
        np.random.seed(list_of_seed[i])
        n = 30

        IJ_sample = torch.empty(n, 2)
        IJ_sample[:, 1] = Q_sampler_i.sample(
            sample_shape=(n,))
        IJ_sample[:, 0] = Q_sampler_j.sample(
            sample_shape=(n,))

        # print(IJ_sample[:,1].long())
        covec = factory.learner.V[IJ_sample[:,1].long()]
        vec = factory.learner.V[IJ_sample[:,0].long()]
        weight = torch.exp(torch.sum(covec * vec, dim=1))

        weight_dist = weight/weight.sum()
        print("the weight distribution shape is: ", weight_dist)
        print("mean of the weight distribution is: ", torch.mean(weight_dist))
        num_sample = 1
        secondary_sampler = torch.distributions.categorical.Categorical(weight_dist)
        Xk = secondary_sampler.sample((num_sample,))
        res.append(list(Xk.cpu().numpy()))

    print(res)

