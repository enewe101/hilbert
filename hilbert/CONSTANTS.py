import torch
from os import path

MATRIX_DEVICE = 'cuda'
MEMORY_DEVICE = 'cpu'
DEFAULT_DTYPE = torch.float32
CODE_DIR = path.abspath(path.join(__file__, '..'))
TEST_DIR = path.join(CODE_DIR, 'tests', 'test-data')
TEST_TOKEN_PATH = path.join(TEST_DIR, 'test_doc.txt')
TEST_DOCS_DIR = path.join(TEST_DIR, 'test-docs')
