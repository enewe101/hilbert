from os import path
try:
    import torch
except ImportError:
    torch = None

MATRIX_DEVICE = 'cuda'
MEMORY_DEVICE = 'cpu'
DEFAULT_DTYPE = torch.float32
CODE_DIR = path.abspath(path.join(__file__, '..'))
TEST_DIR = path.join(CODE_DIR, 'test-data')
TEST_TOKEN_PATH = path.join(CODE_DIR, 'test-data', 'test_doc.txt')

TEST_DOCS_DIR = path.join(CODE_DIR, 'test-data', 'test-docs')
