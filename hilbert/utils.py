
def ensure_implementation_valid(implementation):
    if implementation != 'torch' and implementation != 'numpy':
        raise ValueError(
            "implementation must be 'torch' or 'numpy'.  Got %s."
            % repr(implementation)
        )

