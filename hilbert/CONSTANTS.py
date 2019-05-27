import torch
import json
import os

def read_rc():
    RC = {
        'cooccurrence_dir': None,
        'corpus_dir': None,
        'embeddings_dir':None,
        'task_data_path': None,
        'device':'cuda',
        'dtype': '32',
        'max_sector_size': '12000',
    }
    try:
        with open(os.path.expanduser('~/.hilbertrc')) as rc_file:
            found_rc = json.loads(rc_file.read())
            for key in found_rc:
                if key not in RC:
                    raise BadRCFile(
                        'Unexpected parameter in rc file: {}'.format(key))
                RC[key] = found_rc[key]
    except OSError:
        pass

    # Interpret ints
    for int_field in ['max_sector_size']:
        RC[int_field] = int(RC[int_field])

    # Convert dtype specification into an actual torch dtype.
    RC['dtype'] = {
        'half': torch.float16, 'float': torch.float32, 'double': torch.float64,
        '16': torch.float16, '32': torch.float32, '64': torch.float64,
    }[RC['dtype']]

    return RC


RC = read_rc()

MATRIX_DEVICE = 'cuda' # TODO: purge this global away
MEMORY_DEVICE = 'cpu' # TODO: purge this global away
DEFAULT_DTYPE = torch.float32
CODE_DIR = os.path.abspath(os.path.join(__file__, '..'))
TEST_DIR = os.path.join(CODE_DIR, 'tests', 'test-data')
TEST_TOKEN_PATH = os.path.join(TEST_DIR, 'test_doc.txt')
TEST_DOCS_DIR = os.path.join(TEST_DIR, 'test-docs')
