import os
from unittest import TestCase
from copy import copy, deepcopy
import hilbert as h


# These functions came from hilbert-experiments, where they were only being
# used to support testing.  Now that the Dictionary and it's testing have moved
# here, I have copied these helper functions and changed them minimally.
def iter_test_fnames():
    for path in os.listdir(h.CONSTANTS.TEST_DOCS_DIR):
        if not skip_file(path):
            yield os.path.basename(path)
def iter_test_paths():
    for fname in iter_test_fnames():
        yield get_test_path(fname)
def get_test_tokens():
    paths = iter_test_paths()
    return read_tokens(paths)
def read_tokens(paths):
    tokens = []
    for path in paths:
        with open(path) as f:
            tokens.extend([token for token in f.read().split()])
    return tokens
def skip_file(fname):
    if fname.startswith('.'):
        return True
    if fname.endswith('.swp') or fname.endswith('.swo'):
        return True
    return False
def get_test_path(fname):
    return os.path.join(h.CONSTANTS.TEST_DOCS_DIR, fname)


class TestDictionary(TestCase):

    def get_test_dictionary(self):

        tokens = get_test_tokens()
        return tokens, h.dictionary.Dictionary(tokens)


    def test_copy(self):

        # NOTE: currently implementation of copy is simply deferred to deepcopy

        tokens, dictionary1 = self.get_test_dictionary()
        dictionary2 = copy(dictionary1)

        # None of the obejects are the same
        self.assertTrue(dictionary2 is not dictionary1)
        self.assertTrue(dictionary2.tokens is not dictionary1.tokens)
        self.assertTrue(dictionary2.token_ids is not dictionary1.token_ids)

        # But they are equal
        self.assertEqual(dictionary2.tokens, dictionary1.tokens)
        self.assertEqual(dictionary2.token_ids, dictionary1.token_ids)


    def test_deepcopy(self):

        # NOTE: currently implementation of copy is simply deferred to deepcopy

        tokens, dictionary1 = self.get_test_dictionary()
        dictionary2 = deepcopy(dictionary1)

        # None of the obejects are the same
        self.assertTrue(dictionary2 is not dictionary1)
        self.assertTrue(dictionary2.tokens is not dictionary1.tokens)
        self.assertTrue(dictionary2.token_ids is not dictionary1.token_ids)

        # But they are equal
        self.assertEqual(dictionary2.tokens, dictionary1.tokens)
        self.assertEqual(dictionary2.token_ids, dictionary1.token_ids)


    def test_dictionary(self):
        tokens, dictionary = self.get_test_dictionary()
        for token in tokens:
            dictionary.add_token(token)

        self.assertEqual(set(tokens), set(dictionary.tokens))
        expected_token_ids = {
            token:idx for idx, token in enumerate(dictionary.tokens)}
        self.assertEqual(expected_token_ids, dictionary.token_ids)


    def test_save_load_dictionary(self):
        write_path = os.path.join(h.CONSTANTS.TEST_DIR, 'test.dictionary')

        # Remove files that could be left from a previous test.
        if os.path.exists(write_path):
            os.remove(write_path)

        tokens, dictionary = self.get_test_dictionary()
        dictionary.save(write_path)
        loaded_dictionary = h.dictionary.Dictionary.load(
            write_path)

        self.assertEqual(loaded_dictionary.tokens, dictionary.tokens)
        self.assertEqual(loaded_dictionary.token_ids, dictionary.token_ids)

        # Cleanup
        os.remove(write_path)

