import hilbert as h

try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None


def load_shard(
    source,
    shard=None,
    from_sparse=False,
    dtype=torch.float32,
    device=h.CONSTANTS.MATRIX_DEVICE,
):
    if from_sparse:
        shard = shard or slice(None)
        return torch.tensor(source[shard].toarray(), dtype=dtype, device=device)
    return torch.tensor(source[shard], dtype=dtype, device=device)


def dot(array_or_tensor_1, array_or_tensor_2):
    if isinstance(array_or_tensor_1, np.ndarray):
        return np.dot(
            array_or_tensor_1.reshape(-1), array_or_tensor_2.reshape(-1))
    elif isinstance(array_or_tensor_1, torch.Tensor):
        return torch.dot(
            array_or_tensor_1.reshape(-1), array_or_tensor_2.reshape(-1))
    raise ValueError(
        'Expected either numpy ndarray or torch Tensor.  Got %s'
        % type(array_or_tensor).__name__
    )


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



def transpose(array_or_tensor):
    if isinstance(array_or_tensor, np.ndarray):
        return array_or_tensor.T
    elif isinstance(array_or_tensor, torch.Tensor):
        return array_or_tensor.t()
    raise ValueError('Expected either numpy ndarray or torch Tensor.')


def ensure_implementation_valid(implementation, device=None):
    if implementation != 'torch' and implementation != 'numpy':
        raise ValueError(
            "implementation must be 'torch' or 'numpy'.  Got %s."
            % repr(implementation)
        )

    if device is not None and device != 'cuda' and device != 'cpu':
        raise ValueError(
            "`device` must be None, 'cuda', or 'cpu'.  Got %s."
            % repr(device)
        )


def fill_diagonal(tensor_2d, diag):
    """
    Return a copy of tensor_2d in which the diagonal has been replaced by 
    the value ``diag``.  The copy has the same dtype and device as the original.
    """
    eye = torch.eye(
        tensor_2d.shape[0], dtype=h.CONSTANTS.DEFAULT_DTYPE, 
        device=tensor_2d.device
    )
    return tensor_2d * (1. - eye) + eye * float(diag)


