import codecs

class Dictionary(object):

    def __init__(self, tokens=None):
        self.tokens = []
        self.token_ids = {}
        if tokens is not None:
            for token in tokens:
                self.add_token(token)

    def get_id(self, token):
        return self.token_ids[token]

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


    def __len__(self):
        return len(self.tokens)

