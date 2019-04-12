import torch


class SparseArray:

    def __init__(self, N, I, J, t=100):
        self.t = t
        self.N = N.float()
        self.I = I
        self.J = J
        self.max_i = torch.max(I)
        self.max_j = torch.max(J)
        self.posI = self.calc_pos_lookup(I)
        self.posJ = self.calc_pos_lookup(J)
        self.validate_ordering(I, J)


    def validate_ordering(self, I, J):
        # Check whether the I and J index lists are sorted by I then by J.
        pass


    def calc_pos_lookup(self, X):
        bc = torch.bincount(X)
        posX = torch.empty(bc.shape[0] + 1, dtype=torch.long)
        posX[:-1] = torch.cumsum(bc, dim=0)
        posX[-1] = posX[-2]
        posX[:-1] -= bc
        return posX


    def resolve_indexes(self, Is, Js):
        i_offsets = self.posI[Is]
        i_nums = self.posI[Is+1] - i_offsets
        j_anchors = i_offsets + self.posJ[Js] * i_nums / self.N.shape[0]
        guess_range = torch.arange(-self.t, self.t).view((1, -1))
        j_guesses = j_anchors.view((-1,1)) + guess_range
        guess_is_good = (
            (self.I[j_guesses] == Is.view((-1,1))) *
            (self.J[j_guesses] == Js.view((-1,1)))
        )
        success = guess_is_good.sum(dim=1).float()
        correct = guess_is_good.argmax(dim=1)
        full_offset = j_guesses[torch.arange(Is.shape[0]), correct]
        return full_offset, success


    def __getitem__(self, key):
        Is, Js = key
        indexes, success = self.resolve_indexes(Is, Js)
        return self.N[indexes] * success



