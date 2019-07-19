import hilbert as h
import os

def dependency_corpus_path():
    return os.path.join(h.CONSTANTS.TEST_DIR, 'test-dependency-corpus')

def load_dependency_corpus():
    return h.dependency.DependencyCorpus(dependency_corpus_path())
