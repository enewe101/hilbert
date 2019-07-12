import os
from unittest import TestCase
import hilbert as h
try:
    import torch
except ImportError:
    torch = None


class TestDependencyCorpus(TestCase):

    def test_dependency_corpus(self):

        dep_corp = h.tests.load_test_data.load_dependency_corpus()
        dependency_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-dependency-corpus')
        dependency_corpus_path = os.path.join(dependency_path, 'corpus')
        sentences = open(dependency_corpus_path).read().split('\n\n')
        dictionary_path = os.path.join(dependency_path, 'dictionary')
        dictionary = h.dictionary.Dictionary.load(dictionary_path)
        arc_dictionary_path = os.path.join(dependency_path, 'arc-dictionary')
        arc_dictionary = h.dictionary.Dictionary.load(arc_dictionary_path)

        words = []
        heads = []
        arc_types = []
        for sent in sentences:
            if sent == '': 
                continue
            
            sent_words = []
            sent_heads = []
            sent_arc_types = []
            for row in sent.split('\n'):
                fields = row.split('\t')
                word, head, arc = fields[1], fields[6], fields[7]
                if head == '_' or arc == '_':
                    continue
                sent_words.append(dictionary.get_id(word))
                sent_heads.append(int(head))
                sent_arc_types.append(arc_dictionary.get_id(arc))

            words.append(sent_words)
            heads.append(sent_heads)
            arc_types.append(sent_arc_types)

        sentences = list(zip(words, heads, arc_types))
        sentence_lengths = torch.tensor([len(sent[0]) for sent in sentences])
        sort_idxs = torch.argsort(sentence_lengths)

        for sent_idx in range(len(sentences)):
            found_sentence = dep_corp.data[sent_idx]
            expected_sentence = sentences[dep_corp.sort_idxs[sent_idx]]

            sentence_len = len(expected_sentence[0])

            # Actual observed words should be the same
            self.assertTrue(torch.equal(
                found_sentence[0][1:sentence_len+1],
                torch.tensor(expected_sentence[0])
            ))

            # The first token shoudl be root
            self.assertEqual(found_sentence[0][0].item(), 1)

            # Sentence should be padded.
            self.assertTrue(torch.equal(
                found_sentence[0][sentence_len+1:],
                torch.tensor(
                    [h.dependency.PAD] 
                    * (h.dependency.MAX_SENTENCE_LENGTH - sentence_len)
                )
            ))

            # Heads should be as expected
            self.assertTrue(torch.equal(
                found_sentence[1][1:sentence_len+1],
                torch.tensor(expected_sentence[1])
            ))

            # The root has no head (indicated by padding)
            self.assertEqual(found_sentence[1][0].item(), h.dependency.PAD)

            # The list of heads for the sentence is padded.
            self.assertTrue(torch.equal(
                found_sentence[1][sentence_len+1:],
                torch.tensor(
                    [h.dependency.PAD] 
                    * (h.dependency.MAX_SENTENCE_LENGTH - sentence_len)
                )
            ))

            # Arc types should be as expected.
            self.assertTrue(torch.equal(
                found_sentence[2][1:sentence_len+1],
                torch.tensor(expected_sentence[2])
            ))
            
            # The root has no incoming arc_type (has padding)
            self.assertEqual(found_sentence[2][0].item(), h.dependency.PAD)

            # The list of arc-types should be padded.
            self.assertTrue(torch.equal(
                found_sentence[2][sentence_len+1:],
                torch.tensor(
                    [h.dependency.PAD] 
                    * (h.dependency.MAX_SENTENCE_LENGTH - sentence_len)
                )
            ))







