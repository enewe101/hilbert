
# A "shard" representing the whole array
whole = (slice(None), slice(None))


def on_diag(shard):
    """
    True if `shard` contains elements from the main diagonal.
    """
    return shard is None or shard[0] == shard[1]


class Shards:
    """
    Given some ``shard_factor``, a Shards instance will allow you to address
    many submatrices in a given torch 2D tensor.

    A shards instance helps you break down a 2D array into ``shard_factor**2``
    number of separate pieces called *shards*.  These pieces are organized in 
    a special way.  They aren't blocks; in fact, none of the elements in a 
    shard are continuous---they are spread out evenly in a regular way.

    A Shards instance gives you *indices*, more specifically, a tuple of slice
    objects.  If you use these to key into a 2D tensor using square bracket
    notation then you will get the corresponding shard from that 2D array.
    E.g.:

        ``indices = shards[7]``
        ``my_shard = my_tensor[indices]``

    The ``shard_num``th shard contains the ``i,j``th element if:

        ``i // shard_factor == shard_num and``
        ``j % shard_factor == shard_num``
    """

    def __init__(self, shard_factor):
        self.shard_factor = shard_factor
        self.shard_pointer = 0
        self.num_shards = shard_factor * shard_factor

    def __getitem__(self, shard_num):
        if shard_num >= self.num_shards:
            raise IndexError('Shards index out of range: {}'.format(shard_num))
        if shard_num < 0:
            old_shard_num = shard_num
            shard_num += self.num_shards
            if shard_num < 0:
                raise IndexError(
                    'Shards index out of range: {}'.format(old_shard_num))
        shard_i = shard_num // self.shard_factor
        shard_j = shard_num % self.shard_factor
        return (
            slice(shard_i, None, self.shard_factor), 
            slice(shard_j, None, self.shard_factor)
        )

    def __len__(self):
        return self.num_shards

    def __iter__(self):
        return self.copy()

    def __next__(self):
        if self.shard_pointer >= self.num_shards:
            raise StopIteration

        self.shard_pointer += 1
        return self[self.shard_pointer - 1]

    def copy(self):
        return Shards(self.shard_factor)



