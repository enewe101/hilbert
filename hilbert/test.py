from unittest import main, TestCase
import numpy as np
import hilbert as h

class TestCorpusStats(TestCase):

    UNIQUE_TOKENS = {
    '.': 5, 'Drive': 3, 'Eat': 7, 'The': 10, 'bread': 0, 'car': 6,
    'has': 8, 'sandwich': 9, 'spin': 4, 'the': 1, 'wheels': 2
    }
    N_XX_2 = np.array([
        [ 0.,  4.,  0.,  1.,  0.,  8.,  0., 11.,  4.,  4.,  0.],
        [ 4.,  0.,  0.,  4.,  0., 23.,  4.,  8.,  0.,  4.,  0.],
        [ 0.,  0.,  0.,  0.,  4., 12.,  4.,  0.,  4.,  0.,  8.],
        [ 1.,  4.,  0.,  0.,  3.,  4.,  4.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  4.,  3.,  0.,  4.,  0.,  0.,  0.,  0.,  4.],
        [ 8., 23., 12.,  4.,  4.,  0.,  8.,  7.,  8.,  8., 12.],
        [ 0.,  4.,  4.,  4.,  0.,  8.,  0.,  0.,  4.,  0.,  8.],
        [11.,  8.,  0.,  0.,  0.,  7.,  0.,  0.,  0.,  4.,  0.],
        [ 4.,  0.,  4.,  0.,  0.,  8.,  4.,  0.,  0.,  4.,  8.],
        [ 4.,  4.,  0.,  0.,  0.,  8.,  0.,  4.,  4.,  0.,  8.],
        [ 0.,  0.,  8.,  0.,  4., 12.,  8.,  0.,  8.,  8.,  0.]
    ]) 
    N_XX_3 = np.array([
        [ 0., 12.,  0.,  1.,  0., 12.,  0., 11.,  4.,  4.,  4.],
        [12.,  0.,  0.,  5.,  3., 23.,  4., 11.,  0.,  4.,  8.],
        [ 0.,  0.,  8.,  3.,  4., 12.,  4.,  0.,  4.,  0., 12.],
        [ 1.,  5.,  3.,  0.,  3.,  8.,  4.,  0.,  0.,  0.,  0.],
        [ 0.,  3.,  4.,  3.,  0.,  8.,  0.,  0.,  0.,  0.,  4.],
        [12., 23., 12.,  8.,  8.,  0., 16., 15., 16., 15., 16.],
        [ 0.,  4.,  4.,  4.,  0., 16.,  8.,  0.,  4.,  0.,  8.],
        [11., 11.,  0.,  0.,  0., 15.,  0.,  0.,  4.,  4.,  0.],
        [ 4.,  0.,  4.,  0.,  0., 16.,  4.,  4.,  0.,  4., 12.],
        [ 4.,  4.,  0.,  0.,  0., 15.,  0.,  4.,  4.,  8.,  8.],
        [ 4.,  8., 12.,  0.,  4., 16.,  8.,  0., 12.,  8.,  0.]
    ])


    def test_PMI(self):
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        N_x = np.sum(N_xx, axis=1)
        N = np.sum(N_x)
        with np.errstate(divide='ignore'):
            expected_PMI = np.array([
                [
                    np.log(N * N_xx[i,j] / (N_x[i] * N_x[j])) 
                    for j in range(N_xx.shape[1])
                ] 
                for i in range(N_xx.shape[0])
            ])
        found_PMI = h.corpus_stats.calc_PMI(N_xx, N_x)
        self.assertTrue(np.allclose(found_PMI, expected_PMI))


    def test_calc_postive_PMI(self):
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        with np.errstate(divide='ignore'):
            log_N_xx = np.log(N_xx)
            log_N_x = np.log(np.sum(N_xx, axis=1).reshape((-1,1)))
            log_N = np.log(np.sum(N_xx))
        PMI = log_N + log_N_xx - log_N_x - log_N_x.T
        positive_PMI = h.corpus_stats.calc_positive_PMI(N_xx, N_x)
        for pmi, ppmi in zip(np.nditer(PMI), np.nditer(positive_PMI)):
            self.assertTrue(pmi == ppmi or (pmi < 0 and ppmi == 0))



    def test_calc_shifted_PMI(self):
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        N_x = np.sum(N_xx, axis=1)
        N = np.sum(N_x)
        k = 15.0
        with np.errstate(divide='ignore'):
            expected = np.array([
                [ np.log(N * N_xx[i,j] / (k * N_x[i] * N_x[j])) 
                    for j in range(N_xx.shape[1])
                ]
                for i in range(N_xx.shape[0])
            ])
        found = h.corpus_stats.calc_shifted_w2v_PMI(k, N_xx, N_x)
        self.assertTrue(np.allclose(found, expected))


    def test_get_stats(self):
        # First, test with a cooccurrence window of +/-2
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        for i, token1 in enumerate(dictionary):
            cache_idx1 = self.UNIQUE_TOKENS[token1]
            for j, token2 in enumerate(dictionary):
                cache_idx2 = self.UNIQUE_TOKENS[token2]
                self.assertEqual(
                    self.N_XX_2[cache_idx1, cache_idx2],
                    N_xx[i,j]
                )

        # Next, test with a cooccurrence window of +/-3
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(3)
        for i, token1 in enumerate(dictionary):
            cache_idx1 = self.UNIQUE_TOKENS[token1]
            for j, token2 in enumerate(dictionary):
                cache_idx2 = self.UNIQUE_TOKENS[token2]
                self.assertEqual(
                    self.N_XX_3[cache_idx1, cache_idx2],
                    N_xx[i,j]
                )






class TestFDeltas(TestCase):


    def test_sigmoid(self):
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        PMI = h.corpus_stats.calc_PMI(N_xx, N_x)
        expected = np.array([
            [1/(1+np.e**(-pmi)) for pmi in row]
            for row in PMI
        ])
        result = np.zeros(PMI.shape)
        h.f_delta.sigmoid(PMI, result)
        self.assertTrue(np.allclose(expected, result))



    def test_N_xx_neg(self):
        k = 15.0
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        N_x = np.sum(N_xx, axis=1)
        N = np.sum(N_x)
        expected = np.array([
            [ k * N_x[i] * N_x[j] / N for j in range(N_xx.shape[1])]
            for i in range(N_xx.shape[0])
        ])

        # Compare to manually calculated value above
        found = h.f_delta.calc_N_neg_xx(k, N_x.reshape((-1,1)))
        self.assertTrue(np.allclose(expected, found))



    def test_f_w2v(self):
        k = 15
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)

        M = h.corpus_stats.calc_PMI(N_xx, N_x) - np.log(k)
        M_hat = M + 1
        N_x = np.sum(N_xx, axis=1).reshape((-1,1))
        N_neg_xx = h.f_delta.calc_N_neg_xx(k, N_x)

        difference = h.f_delta.sigmoid(M) - h.f_delta.sigmoid(M_hat)
        multiplier = N_neg_xx + N_xx
        expected = multiplier * difference

        delta = np.zeros(M.shape)
        f_w2v = h.f_delta.get_f_w2v(N_xx, N_x, k)
        found = f_w2v(M, M_hat, delta)

        self.assertTrue(np.allclose(expected, found))


    def test_f_glove(self):
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        with np.errstate(divide='ignore'):
            M = np.log(N_xx)
        M_hat = M_hat = M - 1
        expected = np.array([
            [
                2 * min(1, (N_xx[i,j] / 100.0)**0.75) * (M[i,j] - M_hat[i,j])
                if N_xx[i,j] > 0 else 0 for j in range(N_xx.shape[1])
            ]
            for i in range(N_xx.shape[0])
        ])

        delta = np.zeros(M.shape)
        f_glove = h.f_delta.get_f_glove(N_xx)
        found = f_glove(M, M_hat, delta)

        self.assertTrue(np.allclose(expected, found))
        f_glove = h.f_delta.get_f_glove(N_xx, 10)
        found2 = f_glove(M, M_hat, delta)

        expected2 = np.array([
            [
                2 * min(1, (N_xx[i,j] / 10.0)**0.75) * (M[i,j] - M_hat[i,j])
                if N_xx[i,j] > 0 else 0 for j in range(N_xx.shape[1])
            ]
            for i in range(N_xx.shape[0])
        ])
        self.assertTrue(np.allclose(expected2, found2))
        self.assertFalse(np.allclose(expected2, expected))


    def test_f_mse(self):
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        M = h.corpus_stats.calc_positive_PMI(N_xx, N_x)
        M_hat = M + 1
        expected = M - M_hat
        delta = np.zeros(M.shape)
        found = h.f_delta.f_mse(M, M_hat, delta)
        np.testing.assert_equal(expected, found)


    def test_calc_M_swivel(self):
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        with np.errstate(divide='ignore'):
            log_N_xx = np.log(N_xx)
            log_N_x = np.log(np.sum(N_xx, axis=1))
            log_N = np.log(np.sum(N_xx))
        expected = np.array([
            [
                log_N + log_N_xx[i,j] - log_N_x[i] - log_N_x[j] 
                if N_xx[i,j] > 0 else log_N - log_N_x[i] - log_N_x[j] 
                for j in range(N_xx.shape[1])
            ]
            for i in range(N_xx.shape[0])
        ])
        found = h.f_delta.calc_M_swivel(N_xx, N_x)
        self.assertTrue(np.allclose(expected, found))


    def test_f_swivel(self):
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        M = h.f_delta.calc_M_swivel(N_xx, N_x)
        M_hat = M + 1
        expected = np.array([
            [
                np.sqrt(N_xx[i,j]) * (M[i,j] - M_hat[i,j]) 
                if N_xx[i,j] > 0 else
                (np.e**(M[i,j] - M_hat[i,j]) /
                    (1 + np.e**(M[i,j] - M_hat[i,j])))
                for j in range(M.shape[1])
            ]
            for i in range(M.shape[0])
        ])
        delta = np.zeros(M.shape)
        f_swivel = h.f_delta.get_f_swivel(N_xx, N_x)
        found = f_swivel(M, M_hat, delta)
        self.assertTrue(np.allclose(found, expected))


    def test_f_MLE(self):
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        M = h.corpus_stats.calc_PMI(N_xx, N_x)
        M_hat = M + 1

        N_x = np.sum(N_xx, axis=1).reshape((-1,1))
        N_indep_xx = N_x * N_x.T
        N_indep_max = np.max(N_indep_xx)

        expected = N_indep_xx / N_indep_max * (np.e**M - np.e**M_hat)

        delta = np.zeros(M.shape)
        f_MLE = h.f_delta.get_f_MLE(N_xx, N_x)
        found = f_MLE(M, M_hat, delta)
        self.assertTrue(np.allclose(found, expected))

        t = 10
        expected = (N_indep_xx / N_indep_max)**(1.0/t) * (np.e**M - np.e**M_hat)
        found = f_MLE(M, M_hat, delta, t=t)
        self.assertTrue(np.allclose(found, expected))



class TestConstrainer(TestCase):

    def test_glove_constrainer(self):
        W, V = np.zeros((3,3)), np.zeros((3,3))
        h.constrainer.glove_constrainer(W, V)
        self.assertTrue(np.allclose(W, np.array([[0,1,0]]*3)))
        self.assertTrue(np.allclose(V, np.array([[1,0,0]]*3).T))





class TestHilbertEmbedder(TestCase):

    def test_integration_with_constrainer(self):

        d = 11
        learning_rate = 0.01
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        M = h.corpus_stats.calc_positive_PMI(N_xx, N_x)

        # Define an arbitrary f_delta
        # First make a non-one-sided embedder.
        embedder = h.embedder.HilbertEmbedder(
            M, d, h.f_delta.f_mse, learning_rate,
            constrainer=h.constrainer.glove_constrainer
        )

        old_V = embedder.V.copy()
        old_W = embedder.W.copy()

        embedder.cycle(print_badness=False)

        # Check that the update was performed, and constraints applied.
        new_V = old_V + learning_rate * np.dot(old_W.T, M - embedder.M_hat)
        new_W = old_W + learning_rate * np.dot(M - embedder.M_hat, old_V.T)
        h.constrainer.glove_constrainer(new_W, new_V)
        self.assertTrue(np.allclose(embedder.V, new_V))
        self.assertTrue(np.allclose(embedder.W, new_W))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        embedder.calc_badness()
        badness = np.sum(abs(M - np.dot(old_W, old_V))) / (d*d)
        self.assertEqual(badness, embedder.badness)



    def test_get_gradient(self):

        d = 11
        learning_rate = 0.01
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        M = h.corpus_stats.calc_positive_PMI(N_xx, N_x)

        embedder = h.embedder.HilbertEmbedder(
            M, d, h.f_delta.f_mse, learning_rate)

        W, V = embedder.W.copy(), embedder.V.copy()
        M_hat = np.dot(W,V)
        delta = M - M_hat
        expected_nabla_W = np.dot(delta, V.T)
        expected_nabla_V = np.dot(W.T, delta)

        nabla_W, nabla_V = embedder.get_gradient()

        self.assertTrue(np.allclose(nabla_W, expected_nabla_W))
        self.assertTrue(np.allclose(nabla_V, expected_nabla_V))


    def test_get_gradient_with_offsets(self):

        d = 11
        learning_rate = 0.01
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        M = h.corpus_stats.calc_positive_PMI(N_xx, N_x)
        offset_W = np.random.random(N_xx.shape)
        offset_V = np.random.random(N_xx.shape)
        
        embedder = h.embedder.HilbertEmbedder(
            M, d, h.f_delta.f_mse, learning_rate)

        original_W, original_V = embedder.W.copy(), embedder.V.copy()
        W, V =  original_W + offset_W,  original_V + offset_V
        M_hat = np.dot(W,V)
        delta = M - M_hat
        expected_nabla_W = np.dot(delta, V.T)
        expected_nabla_V = np.dot(W.T, delta)

        nabla_W, nabla_V = embedder.get_gradient(offsets=(offset_W, offset_V))

        self.assertTrue(np.allclose(nabla_W, expected_nabla_W))
        self.assertTrue(np.allclose(nabla_V, expected_nabla_V))

        # Verify that the embeddings were not altered by the offset
        self.assertTrue(np.allclose(original_W, embedder.W))
        self.assertTrue(np.allclose(original_V, embedder.V))


    def test_get_gradient_one_sided(self):

        d = 11
        learning_rate = 0.01
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        M = h.corpus_stats.calc_positive_PMI(N_xx, N_x)
        
        embedder = h.embedder.HilbertEmbedder(
            M, d, h.f_delta.f_mse, learning_rate, one_sided=True)

        original_V = embedder.V.copy()
        V =  original_V
        M_hat = np.dot(V.T,V)
        delta = M - M_hat
        expected_nabla_V = np.dot(V, delta)

        nabla_W, nabla_V = embedder.get_gradient()

        self.assertTrue(np.allclose(nabla_V, expected_nabla_V))
        self.assertTrue(np.allclose(nabla_W, expected_nabla_V.T))

        # Verify that the embeddings were not altered by the offset
        self.assertTrue(np.allclose(original_V.T, embedder.W))
        self.assertTrue(np.allclose(original_V, embedder.V))


    def test_get_gradient_one_sided_with_offset(self):

        d = 11
        learning_rate = 0.01
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        M = h.corpus_stats.calc_positive_PMI(N_xx, N_x)
        offset_V = np.random.random(N_xx.shape)
        
        embedder = h.embedder.HilbertEmbedder(
            M, d, h.f_delta.f_mse, learning_rate, one_sided=True)

        original_V = embedder.V.copy()
        V =  original_V + offset_V
        M_hat = np.dot(V.T,V)
        delta = M - M_hat
        expected_nabla_V = np.dot(V, delta)

        nabla_W, nabla_V = embedder.get_gradient(offsets=offset_V)

        self.assertTrue(np.allclose(nabla_V, expected_nabla_V))
        self.assertTrue(np.allclose(nabla_W, expected_nabla_V.T))

        # Verify that the embeddings were not altered by the offset
        self.assertTrue(np.allclose(original_V.T, embedder.W))
        self.assertTrue(np.allclose(original_V, embedder.V))



    def test_integration_with_f_delta(self):

        d = 11
        learning_rate = 0.01
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        M = h.corpus_stats.calc_positive_PMI(N_xx, N_x)
        pass_args = {'a':True, 'b':False}

        def mock_f_delta(M_, M_hat_, delta_, **kwargs):
            self.assertTrue(M_ is M)
            self.assertEqual(kwargs, {'a':True, 'b':False})
            return np.subtract(M_, M_hat_, delta_)

        embedder = h.embedder.HilbertEmbedder(
            M, d, mock_f_delta, learning_rate, pass_args=pass_args)

        self.assertEqual(embedder.learning_rate, learning_rate)
        self.assertEqual(embedder.d, d)
        self.assertTrue(embedder.M is M)
        self.assertEqual(embedder.f_delta, mock_f_delta)

        old_W, old_V = embedder.W.copy(), embedder.V.copy()

        embedder.cycle(pass_args=pass_args, print_badness=False)

        # Check that the update was performed
        new_V = old_V + learning_rate * np.dot(old_W.T, embedder.delta)
        new_W = old_W + learning_rate * np.dot(embedder.delta, old_V.T)
        self.assertTrue(np.allclose(embedder.V, new_V))
        self.assertTrue(np.allclose(embedder.W, new_W))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        embedder.calc_badness()
        badness = np.sum(abs(M - np.dot(old_W, old_V))) / (d*d)
        self.assertEqual(badness, embedder.badness)


    def test_arbitrary_f_delta(self):
        d = 11
        learning_rate = 0.01
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        M = h.corpus_stats.calc_positive_PMI(N_xx, N_x)

        # Define an arbitrary f_delta
        delta_amount = 0.1
        delta_always = np.zeros(M.shape) + delta_amount
        def f_delta(M, M_hat, delta):
            delta[:,:] = delta_amount
            return delta

        # First make a non-one-sided embedder.
        embedder = h.embedder.HilbertEmbedder(M, d, f_delta, learning_rate)

        old_V = embedder.V.copy()
        old_W = embedder.W.copy()

        embedder.cycle(print_badness=False)

        # Check that the update was performed.  Notice that the update of W
        # uses the old value of V, hence a synchronous update.
        new_V = old_V + learning_rate * np.dot(old_W.T, delta_always)
        new_W = old_W + learning_rate * np.dot(delta_always, old_V.T)
        self.assertTrue(np.allclose(embedder.V, new_V))
        self.assertTrue(np.allclose(embedder.W, new_W))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        embedder.calc_badness()
        badness = np.sum(delta_always) / (d*d)
        self.assertEqual(badness, embedder.badness)


    def test_one_sided(self):
        d = 11
        learning_rate = 0.01
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        M = h.corpus_stats.calc_positive_PMI(N_xx, N_x)

        # First make a non-one-sided embedder.
        embedder = h.embedder.HilbertEmbedder(
            M, d, h.f_delta.f_mse, learning_rate
        )

        # The covectors and vectors are not the same.
        self.assertFalse(np.allclose(embedder.W, embedder.V.T))

        # Now make a one-sided embedder.
        embedder = h.embedder.HilbertEmbedder(
            M, d, h.f_delta.f_mse, learning_rate, one_sided=True
        )

        # The covectors and vectors are the same.
        self.assertTrue(np.allclose(embedder.W, embedder.V.T))

        old_V = embedder.V.copy()
        embedder.cycle(print_badness=False)

        # Check that the update was performed.
        new_V = old_V + learning_rate * np.dot(old_V, embedder.delta)
        self.assertTrue(np.allclose(embedder.V, new_V))

        # Check that the vectors and covectors are still identical after the
        # update.
        self.assertTrue(np.allclose(embedder.W, embedder.V.T))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        embedder.calc_badness()
        badness = np.sum(abs(M - np.dot(old_V.T, old_V))) / (d*d)
        self.assertEqual(badness, embedder.badness)



    #def test_synchronous(self):

    #    d = 11
    #    learning_rate = 0.01
    #    dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
    #    M = h.corpus_stats.calc_positive_PMI(N_xx, N_x)

    #    embedder = h.embedder.HilbertEmbedder(
    #        M, d, h.f_delta.f_mse, learning_rate, synchronous=True
    #    )

    #    old_W, old_V = embedder.W.copy(), embedder.V.copy()
    #    embedder.cycle(print_badness=False)

    #    # Check that the update was performed.  Notice that the update of W
    #    # uses the old value of V, hence a synchronous update.
    #    new_V = old_V + learning_rate * np.dot(old_W.T, embedder.delta)
    #    new_W = old_W + learning_rate * np.dot(embedder.delta, old_V.T)
    #    self.assertTrue(np.allclose(embedder.V, new_V))
    #    self.assertTrue(np.allclose(embedder.W, new_W))

    #    # Check that the badness is correct 
    #    # (badness is based on the error before last update)
    #    badness = np.sum(abs(M - np.dot(old_W, old_V))) / (d*d)
    #    self.assertEqual(badness, embedder.badness)



    def test_mse_embedder(self):
        d = 11
        learning_rate = 0.01
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        M = h.corpus_stats.calc_positive_PMI(N_xx, N_x)

        mse_embedder = h.embedder.HilbertEmbedder(
            M, d, h.f_delta.f_mse, learning_rate)
        mse_embedder.cycle(100000, print_badness=False)

        self.assertEqual(mse_embedder.V.shape, (M.shape[1],d))
        self.assertEqual(mse_embedder.W.shape, (d,M.shape[0]))

        delta = np.zeros(M.shape, dtype='float64')
        residual = h.f_delta.f_mse(M, mse_embedder.M_hat, delta)

        self.assertTrue(np.allclose(
            residual, np.zeros(M.shape,dtype='float64')))
        

    def test_update(self):

        d = 11
        learning_rate = 0.01
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        M = h.corpus_stats.calc_positive_PMI(N_xx, N_x)

        embedder = h.embedder.HilbertEmbedder(
            M, d, h.f_delta.f_mse, learning_rate)

        old_W, old_V = embedder.W.copy(), embedder.V.copy()

        delta_V = np.random.random(M.shape)
        delta_W = np.random.random(M.shape)
        updates = delta_V, delta_W
        embedder.update(*updates)
        self.assertTrue(np.allclose(old_W + delta_W, embedder.W))
        self.assertTrue(np.allclose(old_V + delta_V, embedder.V))


    def test_update_with_constraints(self):

        d = 11
        learning_rate = 0.01
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        M = h.corpus_stats.calc_positive_PMI(N_xx, N_x)

        embedder = h.embedder.HilbertEmbedder(
            M, d, h.f_delta.f_mse, learning_rate,
            constrainer=h.constrainer.glove_constrainer
        )

        old_W, old_V = embedder.W.copy(), embedder.V.copy()

        delta_V = np.random.random(M.shape)
        delta_W = np.random.random(M.shape)
        updates = delta_V, delta_W
        embedder.update(*updates)

        expected_updated_W = old_W + delta_W
        expected_updated_V = old_V + delta_V
        h.constrainer.glove_constrainer(expected_updated_W, expected_updated_V)

        self.assertTrue(np.allclose(expected_updated_W, embedder.W))
        self.assertTrue(np.allclose(expected_updated_V, embedder.V))



    def test_update_one_sided_rejects_delta_W(self):

        d = 11
        learning_rate = 0.01
        dictionary, N_xx, N_x = h.corpus_stats.get_test_stats(2)
        M = h.corpus_stats.calc_positive_PMI(N_xx, N_x)

        embedder = h.embedder.HilbertEmbedder(
            M, d, h.f_delta.f_mse, learning_rate)

        # Show that we can update covector embeddings for a non-one-sided model
        delta_W = np.ones(M.shape)
        embedder.update(delta_W=delta_W)

        # Now show that a one-sided embedder rejects updates to covectors
        embedder = h.embedder.HilbertEmbedder(
            M, d, h.f_delta.f_mse, learning_rate, one_sided=True)
        delta_W = np.ones(M.shape)
        with self.assertRaises(ValueError):
            embedder.update(delta_W=delta_W)




if __name__ == '__main__':
    main()

