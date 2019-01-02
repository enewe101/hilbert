from unittest import TestCase
import hilbert as h

try:
    import torch
except ImportError:
    torch = None


class TestShards(TestCase):

    def test_absolutize(self):
        base_factor = 3
        relative_factor = 2
        original_array_size = base_factor * 4
        array = torch.arange(original_array_size * original_array_size)
        array = array.reshape((original_array_size, original_array_size))

        for base_shard in h.shards.Shards(base_factor):
            for relative_shard in h.shards.Shards(relative_factor):

                absolute_shard = h.shards.absolutize(relative_shard, base_shard)
                self.assertTrue(torch.all(torch.eq(
                    array[base_shard][relative_shard],
                    array[absolute_shard]
                )))

                # Absolutizing is like multiplication, so use that notation
                # However, this is non-commutative, 
                #
                #   shard1 * shard2 != to shard2 * shard1
                #
                absolute_shard = relative_shard * base_shard
                self.assertTrue(torch.all(torch.eq(
                    array[base_shard][relative_shard],
                    array[absolute_shard]
                )))


    def test_relativize(self):
        base_factor = 3
        relative_factor = 2
        original_array_size = base_factor * 4
        array = torch.arange(original_array_size * original_array_size)
        array = array.reshape((original_array_size, original_array_size))
        ones = torch.ones_like(array)
        zeros = torch.zeros_like(array)

        # Here we will iterate through a set of absolute shards once, and
        # for each absolute shard, we will iterate over a set of base shards.
        # The absolute shards are chosen so that they should correspond to 
        # exactly one base shard, with all other base shards leading to a 
        # value error when relativize is called.  To ensure that every cell
        # is hit exactly once, we count how many times each cell is touched
        # and we check that each absolute shard selects the same cells as
        # the relativized shard when composed with the base shard
        for absolute_shard in h.shards.Shards(base_factor * relative_factor):

            # Every absolute shard should be inside exactly one of the base
            # shards.
            for base_shard in h.shards.Shards(base_factor):
                offset_x = (absolute_shard[0].start - base_shard[0].start)
                offset_y = (absolute_shard[1].start - base_shard[1].start)

                # Absolute shard should be in this base shard
                if offset_x % base_factor == 0 and offset_y % base_factor == 0:

                    relative_shard = h.shards.relativize(
                        absolute_shard, base_shard)
                    self.assertTrue(torch.all(torch.eq(
                        array[base_shard][relative_shard],
                        array[absolute_shard]
                    )))

                    # Relativizing is like division, so use that notation
                    # However, this is non-commutative, 
                    #
                    #    abs / base == rel
                    #
                    # But in general,
                    #
                    #    abs / rel != base
                    #
                    # In fact, `abs / rel` is generally undefined.
                    #
                    relative_shard = absolute_shard / base_shard
                    self.assertTrue(torch.all(torch.eq(
                        array[base_shard][relative_shard],
                        array[absolute_shard]
                    )))

                    zeros[base_shard][relative_shard] += 1

                # Absolute shard should not be in this base shard and should
                # raise a ValueError
                else:
                    with self.assertRaises(ValueError):
                        relative_shard = h.shards.relativize(
                            absolute_shard, base_shard)

                    with self.assertRaises(ValueError):
                        relative_shard = absolute_shard / base_shard

        # We touched every cell exactly once
        self.assertTrue(torch.all(torch.eq(zeros, ones)))



    def test_shards_iteration(self):
        shard_factor = 4
        shards = h.shards.Shards(shard_factor)
        M = torch.arange(64, dtype=torch.float32).view(8,8)
        self.assertTrue(len(list(shards)))
        for i, shard in enumerate(shards):
            if i == 0:
                expected = torch.Tensor([[0,4],[32,36]])
                self.assertTrue(torch.allclose(M[shard], expected))
            elif i == 1:
                expected = torch.Tensor([[1,5],[33,37]])
                self.assertTrue(torch.allclose(M[shard], expected))
            else:
                expected_shard = torch.Tensor([
                    j for j in range(64)
                    # If the row is among allowed rows
                    if (j // 8) % shard_factor == i // shard_factor
                    # If the column is among alowed columns
                    and (j % 8) % shard_factor == i % shard_factor
                ]).view(2,2)
                self.assertTrue(torch.allclose(
                    M[shard], expected_shard
                ))



