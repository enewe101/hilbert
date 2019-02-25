import hilbert as h
import numpy as np
import torch
from scipy import sparse


def load_shard(
    source,
    shard=None,
    dtype=h.CONSTANTS.DEFAULT_DTYPE,
    device=None
):
    device = device or h.CONSTANTS.MATRIX_DEVICE

    # Handle Scipy sparse matrix types
    if isinstance(source, (sparse.csr_matrix, sparse.lil_matrix)):
        shard = shard or slice(None)
        return torch.tensor(source[shard].toarray(), dtype=dtype, device=device)

    # Handle Numpy matrix types
    elif isinstance(source, np.matrix):
        return torch.tensor(
            np.asarray(source[shard]), dtype=dtype, device=device)

    # Handle primitive values (don't subscript with shard).
    elif isinstance(source, (int, float)):
        return torch.tensor(source, dtype=dtype, device=device)

    # Handle Numpy arrays and lists.
    return torch.tensor(source[shard], dtype=dtype, device=device)


def norm(array_or_tensor, ord=2, axis=None, keepdims=False):

    if isinstance(array_or_tensor, np.ndarray):
        return np.linalg.norm(array_or_tensor, ord, axis, keepdims)
    elif isinstance(array_or_tensor, torch.Tensor):
        if axis is None:
            return torch.norm(array_or_tensor, ord)
        else:
            return torch.norm(array_or_tensor, ord, axis, keepdims)
    raise ValueError(
        'Expected either numpy ndarray or torch Tensor.  Got %s'
        % type(array_or_tensor).__name__
    )


def normalize(array_or_tensor, ord=2, axis=None):
    if axis is None:
        raise ValueError('axis must be specifiec (int)')

    return array_or_tensor / norm(array_or_tensor, ord, axis, keepdims=True)



