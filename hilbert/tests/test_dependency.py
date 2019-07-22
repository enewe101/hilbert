import os
from unittest import TestCase
import hilbert as h
try:
    import torch
except ImportError:
    torch = None

PAD = h.CONSTANTS.PAD

class TestDependencyCorpus(TestCase):

    def test_dependency_corpus(self):

        dependency_corpus = h.tests.load_test_data.load_dependency_corpus()
        dependency_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-dependency-corpus')
        dependency_corpus_path = os.path.join(dependency_path, 'corpus')
        sentences = open(dependency_corpus_path).read().split('\n\n')
        dictionary_path = os.path.join(dependency_path, 'dictionary')
        dictionary = h.dictionary.Dictionary.load(dictionary_path)
        arc_dictionary_path = os.path.join(dependency_path, 'arc-dictionary')
        arc_dictionary = h.dictionary.Dictionary.load(arc_dictionary_path)

        # Generate the expected words, head_ids, and arc_types that 
        # characterize the expected sentences in the corpus.
        words = []
        heads = []
        arc_types = []
        for sent in sentences:
            if sent == '': 
                continue

            sent_words = []
            sent_heads = []
            sent_arc_types = []
            filtered_head_ids = [0]
            curr_filtered_head_id = 1
            for row in sent.split('\n'):
                fields = row.split('\t')
                word, head, arc = fields[1], fields[6], fields[7]
                if head == '_' or arc == '_':
                    filtered_head_ids.append(None)
                    continue
                filtered_head_ids.append(curr_filtered_head_id)
                curr_filtered_head_id += 1
                sent_words.append(dictionary.get_id(word))
                sent_heads.append(int(head))
                sent_arc_types.append(arc_dictionary.get_id(arc))

            # Remap head indexes to point to correct post-filtering locations.
            sent_heads = [
                filtered_head_ids[orig_head_id] 
                for orig_head_id in sent_heads
            ]

            # Don't include too long sentences.
            if len(sent_words) > h.dependency.MAX_SENTENCE_LENGTH:
                continue

            words.append(sent_words)
            heads.append(sent_heads)
            arc_types.append(sent_arc_types)

        sentences = list(zip(words, heads, arc_types))
        sentence_lengths = torch.tensor([len(sent[0]) for sent in sentences])
        # Add one, because ROOT was not yet counted.
        sentence_lengths += 1
        sort_idxs = torch.argsort(sentence_lengths)

        for sent_idx in range(len(sentences)):
            found_idx = dependency_corpus.sort_idxs[sent_idx]
            found_sentence = dependency_corpus.sentences[found_idx]
            expected_idx = sort_idxs[sent_idx]
            expected_sentence = sentences[expected_idx]

            sentence_len = len(expected_sentence[0])

            # Actual observed words should be the same
            self.assertEqual(
                found_sentence[0][1:sentence_len+1],
                expected_sentence[0]
            )

            # The first token shoudl be root
            self.assertEqual(found_sentence[0][0], 1)

            # Heads should be as expected
            self.assertEqual(
                found_sentence[1][1:sentence_len+1],
                expected_sentence[1]
            )

            # The root has no head (indicated by padding)
            self.assertEqual(found_sentence[1][0], h.dependency.PAD)

            # Arc types should be as expected.
            self.assertEqual(
                found_sentence[2][1:sentence_len+1],
                expected_sentence[2]
            )
            
            # The root has no incoming arc_type (has padding)
            self.assertEqual(found_sentence[2][0], h.dependency.PAD)








