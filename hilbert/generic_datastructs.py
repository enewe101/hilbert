import abc
import torch

"""
This file is used to store some generic datastructures and 
interfaces that classes either extend or use. This simply centralizes
the design patterns used in this codebase.
"""


class Describable(abc.ABC):

    @abc.abstractmethod
    def describe(self):
        return



######### Data structure functionality #########
def get_unigram_data(bigram, include_unigram_data, device):
    uNx, uNxt, uN = None, None, None
    if include_unigram_data:
        uNx = bigram.uNx.flatten().to(device)
        uNxt = bigram.uNxt.flatten().to(device)
        uN = bigram.uN.to(device)
    return uNx, uNxt, uN


def build_sparse_lil_nxx(bigram, include_unigram_data, device):

    # sparse linked-list representation of Nij
    sparse_nxx = []

    # number of rows
    n_rows = len(bigram.Nxx.data)

    # Iterate over each row in the sparse matrix and get marginals
    Nx = torch.zeros((n_rows,), device=device)
    Nxt = torch.zeros((n_rows,), device=device)

    for i in range(len(bigram.Nxx.data)):
        js_tensor = torch.LongTensor(bigram.Nxx.rows[i]).to(device)
        nijs_tensor = torch.FloatTensor(bigram.Nxx.data[i]).to(device)

        # put in the marginal sums!
        Nx[i] = nijs_tensor.sum()
        Nxt[js_tensor] += nijs_tensor

        # store the implicit sparse matrix as a series
        # of tuples, J-indexes, then Nij values.
        sparse_nxx.append((js_tensor, nijs_tensor,))
        bigram.Nxx.rows[i].clear()
        bigram.Nxx.data[i].clear()

    # now we need to store the other statistics
    N = Nx.sum().to(device)
    return sparse_nxx, (Nx, Nxt, N,), \
           get_unigram_data(bigram, include_unigram_data, device)


def build_sparse_tup_nxx(bigram, include_unigram_data, device):

    # hold the ij indices and nij values separately
    indices = torch.ones((2, bigram.Nxx.nnz,)).long()
    values = torch.zeros((bigram.Nxx.nnz,)).float()

    # number of rows
    n_rows = len(bigram.Nxx.data)

    # Iterate over each row in the sparse matrix and get marginals
    Nx = torch.zeros((n_rows,), device=device)
    Nxt = torch.zeros((n_rows,), device=device)
    start_fill = 0

    for i in range(len(bigram.Nxx.data)):
        js = torch.LongTensor(bigram.Nxx.rows[i])
        nijs = torch.FloatTensor(bigram.Nxx.data[i])

        # set the slice object we are using
        slice_ind = slice(start_fill, start_fill + len(js))

        # set the i indices cleverly by multiplying, rather than assigning
        indices[0, slice_ind] *= i

        # set the j indices by adding them into it (subtract one since 1 init)
        indices[1, slice_ind] += -1 + js

        # set the values by just adding them into it
        values[slice_ind] += nijs

        # put in the marginal sums and clear out
        Nx[i] = nijs.sum()
        Nxt[js] += nijs
        bigram.Nxx.rows[i].clear()
        bigram.Nxx.data[i].clear()

        # repeat
        start_fill += len(js)

    # now we need to store the other statistics
    N = Nx.sum().to(device)
    return (indices, values,), (Nx, Nxt, N,), \
           get_unigram_data(bigram, include_unigram_data, device)