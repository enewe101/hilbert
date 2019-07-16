import hilbert as h
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys


def tester(lr):
    path = '/home/jingyihe/Documents/cooc/5w-dynamic-50k'
    fac = h.factories.build_mle_sample_solver(path,
                                              temperature=1,
                                              batch_size=35000,
                                              balanced=True,
                                              gradient_clipping=None,
                                              learning_rate=lr)

    fig, axs = plt.subplots(1,2, figsize=(15,6))
    V_norm = []
    W_norm = []
    try:
        for i in range(1000):
            # for _ in range(10000): #10k updates
            # sample = fac.loader.sample(100000)
            # idx_i, idx_j = sample[0][:, 0], sample[0][:, 1]
            # exp_ij = torch.exp(fac.learner(sample[0]).detach().cpu())
            # exp_pmi = sample[1]['exp_pmi'].cpu()
            # I_probs = fac.loader.I_sampler.probs[idx_i]
            # J_probs = fac.loader.J_sampler.probs[idx_j]
            # pipj = I_probs * J_probs
            # true_gradient = pipj * (exp_pmi - exp_ij)
            #
            # # in solver function operation
            # response = fac.learner(sample[0])
            # fac.optimizer.zero_grad()
            # fac.cur_loss = fac.loss(response, sample[1])
            # fac.cur_loss.backward()
            # if fac.gradient_clipping is not None:
            #     # Gradient clipping
            #     torch.nn.utils.clip_grad_value_(fac.learner.parameters(), clip_value=fac.gradient_clipping)
            # # start testing
            fac.cycle(1, monitor_closely=True)
            if fac.V_norm.item() >= 10 or fac.W_norm.item() >= 10:
                continue
            V_norm.append(fac.V_norm.item())
            W_norm.append(fac.W_norm.item())
    except h.exceptions.DivergenceError:
        print("collected {} pairs of norms".format(len(V_norm)))
        pass

        # print("set {} experiment... ".format(i))
    print("mean of the norm of gradient is: ", np.mean(V_norm))
    print("std of the norm of gradient is: ", np.std(V_norm))
        # res.append(gradient)

    axs[0].boxplot(V_norm, showmeans=True)
    axs[0].set_title("gradient norm boxplot of V lr={}".format(lr))
    axs[1].boxplot(W_norm, showmeans=True)
    axs[1].set_title("gradient norm boxplot of W lr={}".format(lr))


    plt.savefig("/home/jingyihe/boxplot_norms_lr={}.png".format(lr))




if __name__ == '__main__':

    tester(float(sys.argv[1]))