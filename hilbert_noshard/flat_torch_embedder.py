from time import time
import hilbert_noshard as h
try:
    import numpy as np
    import torch
except ImporError:
    np = None
    torch = None


def train(
    cooc_stats,
    d=300,
    learning_rate=0.00001,
    device='cuda',
    times=1,
    t=1.0
):

    # Initialize the target statistics
    M = h.corpus_stats.calc_PMI(cooc_stats)

    # Make them torchy
    Nx = torch.tensor(cooc_stats.Nx, dtype=torch.float32, device=device)
    M = torch.tensor(M, dtype=torch.float32, device=device)

    # Calculate some useful transformations of those statistics
    multiplier = Nx * Nx.t()
    multiplier = multiplier / torch.max(multiplier)
    exp_M = np.e**M

    # Initialize V
    V = torch.rand((d, M.shape[0]), device=device).mul_(2).sub_(1)
    V.div_(torch.norm(V, 2, dim=1).view(d,1))

    # Initialize W
    W = torch.rand((d, M.shape[0]), device=device).mul_(2).sub_(1)
    W.div_(torch.norm(V, 2, dim=1).view(d,1))
    W = W.t()

    start = time()
    for i in range(times):

        # Calculate the delta
        M_hat = torch.mm(W, V)
        delta = (exp_M - np.e**M_hat)
        tempered_multiplier = multiplier**(1.0/t)
        delta.mul_(tempered_multiplier)
        badness = torch.sum(abs(delta)) / (M.shape[0] * M.shape[1])

        # Determine the gradient
        nabla_V = torch.mm(W.t(), delta)
        nabla_W = torch.mm(delta, V.t())

        # Apply updates
        V += nabla_V * learning_rate
        W += nabla_W * learning_rate

        print(badness)

    print(time() - start)


