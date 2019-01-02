import hilbert as h
from queue import Empty
from multiprocessing import Queue, JoinableQueue

try:
    from scipy import sparse
    import numpy as np
    import torch
except ImportError:
    sparse = None
    np = None
    torch = None


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


#def dot(array_or_tensor_1, array_or_tensor_2):
#    if isinstance(array_or_tensor_1, np.ndarray):
#        return np.dot(
#            array_or_tensor_1.reshape(-1), array_or_tensor_2.reshape(-1))
#    elif isinstance(array_or_tensor_1, torch.Tensor):
#        return torch.dot(
#            array_or_tensor_1.reshape(-1), array_or_tensor_2.reshape(-1))
#    raise ValueError(
#        'Expected either numpy ndarray or torch Tensor.  Got %s'
#        % type(array_or_tensor).__name__
#    )



#def transpose(array_or_tensor):
#    if isinstance(array_or_tensor, np.ndarray):
#        return array_or_tensor.T
#    elif isinstance(array_or_tensor, torch.Tensor):
#        return array_or_tensor.t()
#    raise ValueError('Expected either numpy ndarray or torch Tensor.')


#def ensure_implementation_valid(implementation, device=None):
#    if implementation != 'torch' and implementation != 'numpy':
#        raise ValueError(
#            "implementation must be 'torch' or 'numpy'.  Got %s."
#            % repr(implementation)
#        )
#
#    if device is not None and device != 'cuda' and device != 'cpu':
#        raise ValueError(
#            "`device` must be None, 'cuda', or 'cpu'.  Got %s."
#            % repr(device)
#        )


#def fill_diagonal(tensor_2d, diag):
#    """
#    Return a copy of tensor_2d in which the diagonal has been replaced by 
#    the value ``diag``.  The copy has the same dtype and device as the original.
#    """
#    eye = torch.eye(
#        tensor_2d.shape[0], dtype=tensor_2d.dtype, device=tensor_2d.device)
#    return tensor_2d * (1. - eye) + eye * float(diag)


#def clip(min, max, val):
#    return min if val < min else max if val > max else val


#def sample_sphere(num_vecs, d, device=None):
#    device = device or h.CONSTANTS.MATRIX_DEVICE
#    sample = torch.rand((num_vecs, d), device=device).mul_(2).sub_(1)
#    return sample.div_(torch.norm(sample, 2, dim=1,keepdim=True))


def iterate_queue(
    queue, 
    stop_when_empty=True,
    sentinal=None,
    num_sentinals=1,
    poll_frequency=0.1
):
    while True:

        # Try to pull an item from the queue, wait as long as `poll_frequency`.
        try:
            item = queue.get(timeout=poll_frequency)

        # If it's empty, either stop iterating, or try again.
        except Empty:
            if stop_when_empty:
                raise StopIteration
            else:
                print('waiting')

        # If it isn't empty, note any sentinal or yield the item.
        else:
            if sentinal is not None and isinstance(item, sentinal):
                num_sentinals -= 1
                #try:
                #    queue.task_done()
                #except AttributeError:
                #    pass
                if num_sentinals == 0:
                    raise StopIteration
            else:
                yield item




