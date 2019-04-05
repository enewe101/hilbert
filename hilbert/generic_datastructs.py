import hilbert as h
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


def get_Nxx_coo(
    bigram_path, sector_factor, include_marginals=True, verbose=True
):
    """
    Reads in sectorized bigram data from disk, and converts it into a sparse
    tensor representation using COO format.  If desired, marginal sums are 
    included.
    """

    # Go though each sector and accumulate all of the non-zero data
    # into a single sparse tensor representation.
    float_dtype = h.CONSTANTS.DEFAULT_DTYPE
    device = device=h.CONSTANTS.MEMORY_DEVICE
    data = torch.tensor([], dtype=float_dtype, device=device)
    I = torch.tensor([], dtype=torch.int32, device=device)
    J = torch.tensor([], dtype=torch.int32, device=device)
    for sector_id in h.shards.Shards(sector_factor):

        if verbose:
            print('loading sector {}'.format(sector_id.serialize()))

        # Read the sector, and get the statistics in sparse COO-format
        sector = h.bigram.BigramSector.load(bigram_path, sector_id)
        sector_coo = sector.Nxx.tocoo()

        # Tensorfy the data, and the row and column indices
        add_Nxx = torch.tensor(sector_coo.data, dtype=float_dtype)
        add_i_idxs = torch.tensor(sector_coo.row, dtype=torch.int)
        add_j_idxs = torch.tensor(sector_coo.col, dtype=torch.int)

        # Adjust the row and column indices to account for sharding
        add_i_idxs = add_i_idxs * sector_id.step + sector_id.i
        add_j_idxs = add_j_idxs * sector_id.step + sector_id.j

        # Concatenate
        data = torch.cat((data, add_Nxx))
        I = torch.cat((I, add_i_idxs))
        J = torch.cat((J, add_j_idxs))

    if include_marginals:
        # Every sector has global marginals, so get marginals from last sector.
        Nx = torch.tensor(sector._Nx, dtype=float_dtype)
        Nxt = torch.tensor(sector._Nxt, dtype=float_dtype)
        return data, I, J, Nx, Nxt
    else:
        return data, I, J




def build_sparse_tup_nxx(bigram, include_unigram_data, device):

    # hold the ij indices and nij values separately
    indices = torch.ones((2, bigram.Nxx.nnz,), device=device).int()
    values = torch.zeros((bigram.Nxx.nnz,), device=device).float()

    # number of rows
    n_rows = len(bigram.Nxx.data)

    # Iterate over each row in the sparse matrix and get marginals
    Nx = torch.zeros((n_rows,), device=device)
    Nxt = torch.zeros((n_rows,), device=device)
    start_fill = 0

    for i in range(len(bigram.Nxx.data)):
        js = torch.IntTensor(bigram.Nxx.rows[i]).to(device)
        nijs = torch.FloatTensor(bigram.Nxx.data[i]).to(device)

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
        Nxt[js.long()] += nijs
        bigram.Nxx.rows[i].clear()
        bigram.Nxx.data[i].clear()

        # repeat
        start_fill += len(js)

    # now we need to store the other statistics
    N = Nx.sum().to(device)
    return (indices, values,), (Nx, Nxt, N,), \
           get_unigram_data(bigram, include_unigram_data, device)
