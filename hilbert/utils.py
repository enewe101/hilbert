try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None


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


def ensure_implementation_valid(implementation):
    if implementation != 'torch' and implementation != 'numpy':
        raise ValueError(
            "implementation must be 'torch' or 'numpy'.  Got %s."
            % repr(implementation)
        )

