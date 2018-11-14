from copy import deepcopy
import codecs

class Dictionary(object):

    def __init__(self, tokens=None):
        self.tokens = []
        self.token_ids = {}
        if tokens is not None:
            for token in tokens:
                self.add_token(token)


    def __copy__(self):
        return deepcopy(self)


    def __contains__(self, key):
        return key in self.token_ids


    def __deepcopy__(self, memo):
        result = Dictionary(self.tokens)
        memo[id(self)] = result
        return result


    def __len__(self):
        return len(self.tokens)


    def get_id(self, token):
        return self.token_ids[token]

    def get_id_safe(self, token):
        """
        Do not raise KeyError if the token is not in the dictionary, instead
        return None.
        """
        if token in self.token_ids:
            return self.token_ids[token]
        return None

    def get_token(self, idx):
        return self.tokens[idx]


    def add_token(self, token):
        if token not in self.token_ids:
            idx = len(self.tokens)
            self.token_ids[token] = idx
            self.tokens.append(token)
            return idx
        return self.token_ids[token]


    def save(self, path):
        with codecs.open(path, 'w', 'utf8') as f:
            f.write('\n'.join(self.tokens))


    @staticmethod
    def load(path):
        dictionary = Dictionary()
        with open(path) as f:
            dictionary.tokens = f.read().split('\n')
        dictionary.token_ids = {
            token: idx 
            for idx, token in enumerate(dictionary.tokens)
        }
        return dictionary


