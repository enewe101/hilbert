import numpy as np


def f_mse(M, M_hat, delta):
    return np.subtract(M, M_hat, delta)


def calc_N_neg_xx(k, N_x, N=None):
    N = N if N is not None else np.sum(N_x)
    return float(k) * N_x * N_x.T / N


def get_f_w2v(N_xx, N_x, k):
    N_x = N_x.reshape((-1,1))
    N_neg_xx = calc_N_neg_xx(k, N_x)
    multiplier = N_xx + N_neg_xx
    def f_w2v(M, M_hat, delta):
        np.subtract(sigmoid(M), sigmoid(M_hat), delta)
        return np.multiply(multiplier, delta, delta)
    return f_w2v


def sigmoid(a):
    return 1 / (1 + np.e**(-a))
    

@np.vectorize
def g_glove(N_ij, X_max=100):
    return min(1, (float(N_ij) / X_max)**0.75)


def get_f_glove(N_xx,X_max=100):
    def f_glove(M, M_hat, delta):
        return np.array([
            [ 
                2 * g_glove(N_xx[i,j], X_max) * (M[i,j] - M_hat[i,j]) 
                if N_xx[i,j] > 0 else 0
                for j in range(N_xx.shape[1])
            ]
            for i in range(N_xx.shape[0])
        ])
    return f_glove


def get_f_MLE(N_xx, N_x):

    N_x = N_x.reshape((-1,1))
    N_xx_ind = N_x * N_x.T
    N_xx_ind_max = np.max(N_xx_ind)

    def f_MLE(M, M_hat, delta, t=1):

        multiplier = (N_xx_ind / float(N_xx_ind_max))**(1/float(t))
        difference = (np.e**M - np.e**M_hat)
        return multiplier * difference

    return f_MLE


def get_f_MLE_fast(N_xx, N_x, delta):

    N_x = N_x.reshape((-1,1)).astype('float64')
    N_xx_ind = N_x * N_x.T
    N_xx_ind_max = np.max(N_xx_ind)
    np.divide(N_xx_ind, N_xx_ind_max, N_xx_ind)

    tempered_N_xx_ind = np.zeros(N_xx.shape)
    exp_M = np.zeros(N_xx.shape)
    exp_M_hat = np.zeros(N_xx.shape)

    def f_MLE(M, M_hat, delta, t=1):

        np.power(N_xx_ind, 1.0/t, tempered_N_xx_ind)
        np.power(np.e, M, exp_M)
        np.power(np.e, M_hat, exp_M_hat)
        np.subtract(exp_M, exp_M_hat, delta)
        np.multiply(tempered_N_xx_ind, delta, delta)

        return delta

    return f_MLE


def calc_M_swivel(N_xx):

    with np.errstate(divide='ignore'):
        log_N_xx = np.log(N_xx)
        log_N_x = np.log(np.sum(N_xx, axis=1))
        log_N = np.log(np.sum(N_xx))

    return np.array([
        [
            log_N + log_N_xx[i,j] - log_N_x[i] - log_N_x[j]
            if N_xx[i,j] > 0 else log_N - log_N_x[i] - log_N_x[j]
            for j in range(N_xx.shape[1])
        ]
        for i in range(N_xx.shape[0])
    ])

# TODO: Why is this here?  Redundant with calc_M_swivel?
def get_f_swivel(N_xx, N_x):
    #N_x = np.sum(N_xx, axis=1)

    with np.errstate(divide='ignore'):
        log_N_x = np.log(N_x)
        log_N = np.log(np.sum(N_x))

    def f_swivel(M, M_hat, delta):
        return np.array([
            [
                np.sqrt(N_xx[i,j]) * (M[i,j] - M_hat[i,j])
                if N_xx[i,j] > 0 else
                (np.e**(M[i,j] - M_hat[i,j]) / 
                    (1 + np.e**(M[i,j] - M_hat[i,j])))
                for j in range(M.shape[1])
            ]
            for i in range(M.shape[0])
        ])
    return f_swivel


def glove_constrainer(W, V):
    W[:,1] = 1
    V[0,:] = 1
    return W,V


class HilbertEmbedder(object):

    def __init__(
        self,
        M,
        d=300,
        f_delta=f_mse,
        learning_rate=0.001,
        one_sided=False,
        constrainer=None,
        #synchronous=False,
        pass_args={}
    ):
        self.M = M
        self.d = d
        self.f_delta = f_delta
        self.learning_rate = learning_rate
        self.one_sided = one_sided
        self.constrainer = constrainer
        #self.synchronous = synchronous
        self.num_covecs, self.num_vecs = self.M.shape
        if self.one_sided and self.num_covecs != self.num_vecs:
            raise ValueError('M must be square for a A one-sided embedder.')
        self.reset()
        #self.measure(**pass_args)


    def reset(self):
        self.V = np.random.random((self.d, self.num_vecs)) * 2 - 1
        norms = np.linalg.norm(self.V, axis=1).reshape((-1,1))
        self.V = self.V / norms
        if self.one_sided:
            self.W = self.V.T
        else:
            self.W = np.random.random((self.num_covecs, self.d)) * 2 - 1
            norms = np.linalg.norm(self.W, axis=0).reshape((1,-1))
            self.W = self.W / norms
        self.M_hat = np.zeros(self.M.shape, dtype='float64')
        self.delta = np.zeros(self.M.shape, dtype='float64')
        self.badness = None


    def calc_badness(self):
        total_absolute_error = np.sum(abs(self.delta))
        num_cells = (self.M.shape[0] * self.M.shape[1])
        self.badness = total_absolute_error / num_cells
        return self.badness


    #def measure(self, **pass_args):
    #    """ Determine the current gradient """
    #    self.M_hat = np.dot(self.W, self.V)
    #    self.delta = self.f_delta(self.M, self.M_hat, **pass_args)
    #    self.calc_badness()


    # TODO: make this work for:
    #   - one_sided
    #   - synchronous
    #   - shard
    def get_gradient(self, offsets=None, pass_args=None):
        """ 
        Calculate and return the current gradient.  
            offsets: 
                Allowed values: None, (dW, dV)
                    where dW and dV are is a W.shape and V.shape numpy arrays
                Temporarily applies self.W += dW and self.V += dV before 
                calculating the gradient.
            pass_args:
                Allowed values: dict of keyword arguments.
                Supplies the keyword arguments to f_delta.
        """

        pass_args = pass_args or {}
        # Determine the prediction for current embeddings.  Allow an offset to
        # be specified for solvers like Nesterov Accelerated Gradient.
        if offsets is not None:
            dW, dV = offsets
            np.dot(self.W + dW, self.V + dV, out=self.M_hat)
        else:
            np.dot(self.W, self.V, out=self.M_hat)

        # Determine the errors.
        self.f_delta(self.M, self.M_hat, self.delta, **pass_args)

        # Determine the gradient
        nabla_W = np.dot(self.delta, self.V.T)
        nabla_V = np.dot(self.W.T, self.delta)

        return nabla_W, nabla_V



    def update(self, pass_args=None):
        nabla_W, nabla_V = self.get_gradient(pass_args=pass_args)
        self.V += self.learning_rate * nabla_V
        if not self.one_sided:
            self.W += self.learning_rate * nabla_W


    def apply_constraints(self):
        if self.constrainer is not None:
            self.W, self.V = self.constrainer(self.W,self.V)


    def cycle(self, times=1, print_badness=True, pass_args=None):
        pass_args = pass_args or {}
        for i in range(times):
            self.update(pass_args)
            self.apply_constraints()
            if print_badness:
                print(self.calc_badness())


    def project(self, new_d):

        delta_dim = abs(self.d - new_d)
        if delta_dim == 0:
            print('warning: no change during projection.')
            return

        elif new_d < self.d:
            mass = 1.0 / new_d
            random_projector = np.random.random((delta_dim, new_d)) * mass
            downsampler = np.append(np.eye(new_d), random_projector, axis=0)
            self.W = np.dot(self.W, downsampler)
            self.V = np.dot(downsampler.T, self.V)

        else:
            old_mass = float(self.d) / new_d
            new_mass = float(delta_dim) / new_d
            covector_extension = (np.random.random((
                self.num_covecs, delta_dim)) * 2 - 1) * new_mass
            self.W = np.append(self.W * old_mass, covector_extension, axis=1)
            vector_extension = (np.random.random((
                delta_dim, self.num_vecs)) * 2 - 1) * new_mass
            self.V = np.append(self.V * old_mass, vector_extension, axis=0)



