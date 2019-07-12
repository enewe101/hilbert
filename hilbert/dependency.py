import hilbert as h
import torch
import os
import sys

MAX_SENTENCE_LENGTH = 30
PAD = torch.tensor(sys.maxsize, dtype=torch.int64) 

class DependencyCorpus:

    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.corpus_text_path = os.path.join(corpus_path, 'corpus')
        self.dictionary = h.dictionary.Dictionary.load(os.path.join(
            corpus_path, 'dictionary'
        ))
        self.arc_dictionary = h.dictionary.Dictionary.load(os.path.join(
            corpus_path, 'arc-dictionary'
        ))
        self.data, self.sort_idxs = self.read_all()


    def read_all(self):
        sentences = []
        sentence_lengths = []
        i = 0
        for i, sentence_rows in enumerate(self.iter_sentence_rows()):
            # DEBUG: just handle the first 1000 sentences.
            if i > 1000:
                break

            sentence_data, sentence_len = self.compile_sentence(sentence_rows)
            if sentence_len > MAX_SENTENCE_LENGTH:
                continue

            sentence_lengths.append(sentence_len)
            sentences.append(sentence_data)

        sentences = torch.tensor(sentences, dtype=torch.int64)
        sentence_lengths = torch.tensor(sentence_lengths, dtype=torch.int64)
        sort_idxs = torch.argsort(sentence_lengths)
        sentences = sentences[sort_idxs]
        return sentences, sort_idxs



    def iter_sentence_rows(self):
        lines = []
        with open(self.corpus_text_path) as corpus_file:
            for line in corpus_file:
                if line.strip() == '':
                    if len(lines) > 0:
                        yield lines
                        lines = []
                else:
                    lines.append(line)
            if len(lines) > 0:
                yield lines
        


    def compile_arc_types(self, lines):
        for line in lines:
            fields = line.strip().split('\t')
            self.arc_dictionary.add_token(fields[7])


    def compile_sentence(self, lines):

        # Every sentence has first token [ROOT].
        arcs = []
        for line in lines:
            fields = line.strip().split('\t')
            arcs.append((fields[1], fields[6], fields[7]))

        encoded_arcs = [
            (
                self.dictionary.get_id_safe(arc[0], 0), 
                int(arc[1]), 
                self.arc_dictionary.get_id(arc[2])
            ) 
            for arc in arcs if arc[1] != '_' and arc[2] != '_'
        ]

        padding_length = MAX_SENTENCE_LENGTH - len(encoded_arcs)

       modifiers = (
            [self.dictionary.get_id('[ROOT]')]  # First pos is root
            + [arc[0] for arc in encoded_arcs] 
            + [PAD]*padding_length  # Add padding so sentences have equal length
        )
        heads = (
            [PAD]   # Root has no head (it modifies nothing)
            + [arc[1] for arc in encoded_arcs] 
            + [PAD]*padding_length  # Add padding so sentences have equal length
        )
        arc_types = (
            [PAD]   # Root has no incoming arc (it modifies nothing)
            + [arc[2] for arc in encoded_arcs] 
            + [PAD]*padding_length  # Add padding so sentences have equal length
        )

        assert len(modifiers) == len(heads)
        assert len(modifiers) == len(arc_types)

        return [modifiers, heads, arc_types], len(arcs)

