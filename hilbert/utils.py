import numpy as np
import torch


def norm(array_or_tensor, ord=2, axis=None, keepdims=False):
    if axis is None:
        raise ValueError('axis must be specifiec (int)')

    if isinstance(array_or_tensor, np.ndarray):
        return np.linalg.norm(array_or_tensor, ord, axis, keepdims)
    elif isinstance(array_or_tensor, torch.Tensor):
        return torch.norm(array_or_tensor, ord, axis, keepdims)
    raise ValueError('Expected either numpy ndarray or torch Tensor.')


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

