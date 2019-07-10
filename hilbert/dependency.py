import hilbert as h
import torch
import os


MAX_SENTENCE_LENGTH = 30
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
        self.data = self.read_all()


    def read_all(self):
        self.sentences = []
        lines = []
        i = 0
        with open(self.corpus_text_path) as corpus_file:
            for line in corpus_file:

                # DEBUG: just handle the first 1000 lines.
                if i > 1000:
                    break

                if line.strip() == '':
                    i += 1
                    if len(lines) > 0:
                        self.compile_sentence(lines)
                        lines = []

                else:
                    lines.append(line)

            if len(lines) > 0:
                self.compile_sentence(lines)
        
        return torch.tensor(self.sentences, dtype=torch.int32)


    def compile_arc_types(self, lines):
        for line in lines:
            fields = line.strip().split('\t')
            self.arc_dictionary.add_token(fields[7])


    def compile_sentence(self, lines):

        # Filter out long sentences, since this will add too much padding to
        # other sentences.
        if len(lines) > MAX_SENTENCE_LENGTH:
            return

        arcs = [('[ROOT]', 0, 'root')] # Every sentence has first token [ROOT].
        for line in lines:
            fields = line.strip().split('\t')
            arcs.append((fields[1], fields[6], fields[7]))

        resolved_arcs = [
            (arc[0], arcs[int(arc[1])][0], arc[2]) 
            for arc in arcs if arc[1] != '_' and arc[2] != '_'
        ]

        encoded_arcs = [
            (
                self.dictionary.get_id_safe(arc[0], 0), 
                self.dictionary.get_id_safe(arc[1], 0), 
                self.arc_dictionary.get_id(arc[2])
            ) 
            for arc in resolved_arcs
        ]

        padding_length = MAX_SENTENCE_LENGTH - len(encoded_arcs)

        modifiers = [arc[0] for arc in encoded_arcs] + [-1]*padding_length
        heads = [arc[1] for arc in encoded_arcs] + [-1]*padding_length
        arc_types = [arc[2] for arc in encoded_arcs] + [-1]*padding_length

        assert len(modifiers) == len(heads)
        assert len(modifiers) == len(arc_types)

        self.sentences.append([modifiers, heads, arc_types])

