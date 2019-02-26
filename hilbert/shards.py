
def absolutize(self, base_shard):
    """
    Converts a `self` spec, defined relative to `base_shard` spec
    into the absolute shard spec that provides the same addressing as the
    composition of the relative and base shards.  I.e.:

        array[base_shard][self] == array[absolute_shard]

    """

    abs_shard_slices = []
    for rel_slice, base_slice in zip(self, base_shard):
        abs_shard_slices.append(absolutize_slice(rel_slice, base_slice))

    abs_shard = Shard(abs_shard_slices)

    return abs_shard


def absolutize_slice(relative_slice, base_slice):
    abs_start = relative_slice.start * base_slice.step + base_slice.start
    abs_step = relative_slice.step * base_slice.step
    abs_slice = slice(abs_start, None, abs_step)
    return abs_slice


def relativize(absolute_shard, base_shard):
    """
    Converts an `absolute_shard` spec, into a relative shard spec, defined 
    relative to `base_shard` spec.  I.e.:

        array[base_shard][relative_shard] == array[absolute_shard]

    """
    rel_shard_slices = []
    for abs_slice, base_slice in zip(absolute_shard, base_shard):
        rel_shard_slices.append(relativize_slice(abs_slice, base_slice))

    rel_shard = Shard(rel_shard_slices)

    return rel_shard


def relativize_slice(absolute_slice, base_slice):
    relative_offset = absolute_slice.start - base_slice.start
    start_divisible = (relative_offset % base_slice.step) == 0
    step_divisible = (absolute_slice.step % base_slice.step) == 0

    if not start_divisible or not step_divisible:
        raise ValueError('`absolute_slice` is not within `base_slice`.')

    rel_start = (absolute_slice.start - base_slice.start) // base_slice.step
    rel_step = absolute_slice.step // base_slice.step
    rel_slice = slice(rel_start, None, rel_step)

    return rel_slice


def on_diag(shard):
    """
    True if `shard` contains elements from the main diagonal.
    """
    return shard is None or shard[0] == shard[1]


def serialize(shard):
    if shard is None:
        return 0,0,1

    if not isinstance(shard, tuple) or len(shard) != 2:
        raise ValueError('Value cannot be interpreted as a shard')

    if shard == (None, None):
        return 0,0,1

    shard0, shard1 = shard
    if not(isinstance(shard0, slice)) or not(isinstance(shard1, slice)):
        raise ValueError('Value cannot be interpreted as a shard')

    return shard0.start, shard1.start, shard0.step



class Shard(tuple):
    """
    Represents a subset of the elements of a 2D matrix.  A shard is a tuple of
    two slice objects:

        slice(i, None, step), slice(j, None, step)

    To visualize what this selects, think of how plaid is made of vertically
    oriented stripes intersecting with horizontally oriented stripes... at the
    intersection points, the stripes make squares (which represent the selected
    elements).

    The first slice object represents horizontal stripes.  The start index `i`,
    is index of the row corresponding to the the first stripe.  
    
    The second slice represents vertical stripes, and the first vertical stripe
    is at column `j`.

    Both sets of stripes are separated by a common stride.

    See `Shards` (plural) for more about how shards are used to partition 2D
    arrays.

    You can multiply and divide shards.  For example, define:

        shard3 = shard1 * shard2

    Then, the following expressions are all `True`:

        shard3 == shard2 * shard1
        some_array[shard3] == some_array[shard1][shard2]
        some_array[shard3/shard2] == some_array[shard1]
        some_array[shard3/shard1] == some_array[shard1]

    """


    def __new__(self, slices):
        return tuple.__new__(Shard, slices)


    def __init__(self, slices):
        slice1, slice2 = slices
        assert isinstance(slice1, slice)
        assert isinstance(slice2, slice)
        assert len(self) == 2
        self.step = self[0].step
        self.i = self[0].start
        self.j = self[1].start


    def __rmul__(self, other):
        if not isinstance(other, Shard):
            try:
                other = Shard(other)
            except TypeError: 
                return NotImplemented
        return other.absolutize(self)


    def __mul__(self, other):
        if not isinstance(other, Shard):
            try:
                other = Shard(other)
            except TypeError: 
                return NotImplemented
        return self.absolutize(other)


    def __truediv__(self, other):
        if not isinstance(other, Shard):
            try:
                other = Shard(other)
            except TypeError: 
                return NotImplemented
        return other.relativize(self)


    def __truediv__(self, other):
        if not isinstance(other, Shard):
            try:
                other = Shard(other)
            except TypeError: 
                return NotImplemented
        return self.relativize(other)


    def absolutize(self, base_shard):
        return absolutize(self, base_shard)
    def relativize(self, base_shard):
        return relativize(self, base_shard)
    def serialize(shard):
        return serialize(shard)
    def on_diag(shard):
        return on_diag(shard)

    def __repr__(self):
        return 'Shard' + super(Shard, self).__repr__()


# A "shard" representing the whole array
whole = Shard((slice(0,None,1), slice(0,None,1)))


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
        return Shard((
            slice(shard_i, None, self.shard_factor), 
            slice(shard_j, None, self.shard_factor)
        ))

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



