import hilbert as h
import torch
import os
import sys

MAX_SENTENCE_LENGTH = 30
PAD = h.CONSTANTS.PAD

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
        self.sentences, self.sentence_lengths, self.sort_idxs = self.read_all()


    def read_all(self):
        sentences = []
        sentence_lengths = []
        for sent_id, sentence_rows in enumerate(self.iter_sentence_rows()):
            if sent_id % 10000 == 0:
                print('reading sentence', sent_id)

            sentence_data, sentence_len = self.compile_sentence(sentence_rows)

            was_degenerate_tree = (sentence_data == None)
            if was_degenerate_tree:
                continue

            if sentence_len > MAX_SENTENCE_LENGTH:
                continue

            sentence_lengths.append(sentence_len)
            sentences.append(sentence_data)

        sentence_lengths = torch.tensor(sentence_lengths, dtype=torch.int64)
        sort_idxs = torch.argsort(sentence_lengths)
        return sentences, sentence_lengths, sort_idxs



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


    def print_sentence(self, modifiers, head_ids, arc_types):
        for modifier, head_id, arc_type in zip(modifiers, head_ids, arc_types):
            modifier = self.dictionary.get_token(modifier)
            head = (
                None if head_id == PAD else
                self.dictionary.get_token(modifiers[head_id])
            )
            arc_type = (
                None if arc_type == PAD else
                self.arc_dictionary.get_token(arc_type)
            )
            print(modifier, head, arc_type)


    def compile_sentence(self, lines):
        """
        Compiling the sentence involves the following subtasks:
         - Parsing three fields out of the conll format: the word (aka
           modifier), the head-pointer, and the arc-type.
         - Mapping the list of words (aka modifiers) into their dictionary
            IDs.
         - Mapping the list of arc types into their dictionary IDs.
         - Filtering out any words that don't have heads (i.e. that don't
            participate in the dependency tree).
         - Re-mapping the head-pointers so that they point to the correct
            head (in the modifiers list) after filtering words without heads.
        """

        # We're going to assemble three lists to characterize this sentence
        # We pre-fill each list with one element representing the ROOT.
        # The ROOT has no head, so we use padding for its head and arc-type.
        modifiers = [self.dictionary.get_id('[ROOT]')]
        head_ids = [PAD]
        arc_types = [PAD]

        # We'll parse each line, extracting the modifier, head-pointer, and
        # arc-type.  Meanwhile, we will filter out non-dependency-linked
        # tokens (usually punctuation), and we will keep track of where the
        # head-pointers should point to in order to address a modifier at 
        # it's new post-filtering index.
        filtered_token_ids = [0]    # Pre-insert ROOT's filtered id.
        current_token_id = 1
        for line in lines:

            # Only include tokens that are in the parse tree.
            fields = line.strip().split('\t')
            head_id = int(fields[6]) if fields[6] != '_' else None
            if head_id is None:
                filtered_token_ids.append(None)
                continue

            # OK to include this token in the three lists!
            modifiers.append(self.dictionary.get_id_safe(fields[1], 0))
            head_ids.append(head_id)
            arc_types.append(self.arc_dictionary.get_id(fields[7]))

            # Maintain the mapping between original id and filtered id.
            filtered_token_ids.append(current_token_id)
            current_token_id += 1

        # Re-map all head-pointers so they refer to correct (filtered) ids.
        head_ids = [
            PAD if head_id == PAD else filtered_token_ids[head_id] 
            for head_id in head_ids
        ]
        if None in head_ids:
            return None, None


        assert len(modifiers) == len(head_ids)
        assert len(modifiers) == len(arc_types)

        return (modifiers, head_ids, arc_types), len(modifiers)




        # Every sentence has first token [ROOT].
        root_arc = (self.dictionary.get_id('[ROOT]'), PAD, PAD)
        encoded_arcs = [root_arc]
        head_decrement = 0
        for arc in arcs:
            if arc[1] == '_' or arc[2] == '_':
                head_decrement += 1
                continue
            encoded_arcs.append((
                self.dictionary.get_id_safe(arc[0], 0), 
                int(arc[1]) - head_decrement, 
                self.arc_dictionary.get_id(arc[2])
            )) 

        padding_length = MAX_SENTENCE_LENGTH - len(encoded_arcs)

        modifiers = [arc[0] for arc in encoded_arcs]
        heads = [arc[1] for arc in encoded_arcs]
        arc_types = [arc[2] for arc in encoded_arcs]


