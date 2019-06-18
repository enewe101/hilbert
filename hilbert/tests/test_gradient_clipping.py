import hilbert as h
import torch
from unittest import main, TestCase
import matplotlib.pyplot as plt


class TestGradientClipping(TestCase):


    def do_test(self):
        path = '/home/jingyihe/Documents/cooc/5w-dynamic-v10k'
        fac = h.factories.build_mle_sample_solver(path,
                                                  temperature=1,
                                                  batch_size=10000,
                                                  balanced=True,
                                                  gradient_clipping=1.8e-4)

        # fig, axs = plt.subplots(1,3, figsize=(15,6))
        # for i in range(3):
            # for _ in range(10000):
        sample = fac.loader.sample(100000)
        idx_i, idx_j = sample[0][:, 0], sample[0][:, 1]
        exp_ij = torch.exp(fac.learner(sample[0]).detach().cpu())
        exp_pmi = sample[1]['exp_pmi'].cpu()
        I_probs = fac.loader.I_sampler.probs[idx_i]
        J_probs = fac.loader.J_sampler.probs[idx_j]
        pipj = I_probs * J_probs
        true_gradient = pipj * (exp_pmi - exp_ij)

        # in solver function operation
        response = fac.learner(sample[0])
        fac.optimizer.zero_grad()
        fac.cur_loss = fac.loss(response, sample[1])
        fac.cur_loss.backward()
        if fac.gradient_clipping is not None:
            # Gradient clipping
            torch.nn.utils.clip_grad_value_(fac.learner.parameters(), clip_value=fac.gradient_clipping)
        # start testing
        if true_gradient < 1.8e-4:
            self.assertEqual(true_gradient, )

        # print("set {} experiment... ".format(i))
        # print("mean of gradient is: ", gradient.mean())
        # print("std of gradient is: ", gradient.std())
        # res.append(gradient)

        # axs[i].boxplot(gradient)
        # axs[i].set_title("gradient boxplot -- {}".format(i))


        # plt.savefig("/home/jingyihe/boxplot.png")




if __name__ == '__main__':
    main()