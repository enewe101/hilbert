import numpy as np

def f_mse(M1, M2):
    return M1 - M2


def calc_N_neg_xx(k, N_x, N=None):
    N = N if N is not None else np.sum(N_x)
    return float(k) * N_x * N_x.T / N


def get_f_w2v(N_xx,k):
    N_x = np.sum(N_xx, axis=1).reshape((-1,1))
    N_neg_xx = calc_N_neg_xx(k, N_x)
    def f_w2v(M, M_hat):
        return (N_xx + N_neg_xx) * (sigmoid(M) - sigmoid(M_hat))
    return f_w2v


def sigmoid(a):
    return 1 / (1 + np.e**(-a))
    

@np.vectorize
def g_glove(N_ij, X_max=100):
    return min(1, (float(N_ij) / X_max)**0.75)


def get_f_glove(N_xx,X_max=100):
    def f_glove(M, M_hat):
        return np.array([
            [ 
                2 * g_glove(N_xx[i,j], X_max) * (M[i,j] - M_hat[i,j]) 
                if N_xx[i,j] > 0 else 0
                for j in range(N_xx.shape[1])
            ]
            for i in range(N_xx.shape[0])
        ])
    return f_glove


def get_f_MLE(N_xx):
    N_x = np.sum(N_xx, axis=1)
    N_xx_ind = N_x * N_x.T
    N_xx_ind_max = np.max(N_xx_ind)
    def f_MLE(M, M_hat, t=1):
        return (N_xx_ind / float(N_xx_ind_max))**t * (np.e**M - np.e**M_hat)
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


def get_f_swivel(N_xx):
    N_x = np.sum(N_xx, axis=1)

    with np.errstate(divide='ignore'):
        log_N_x = np.log(N_x)
        log_N = np.log(np.sum(N_x))

    def f_swivel(M, M_hat):
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


def glove_constrainer(W,V,update_complete):
    if update_complete:
        W[:,1] = 1
    else:
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
        synchronous=False,
        pass_args={}
    ):
        self.M = M
        self.d = d
        self.f_delta = f_delta
        self.learning_rate = learning_rate
        self.one_sided = one_sided
        self.constrainer = constrainer
        self.synchronous = synchronous
        self.num_covecs, self.num_vecs = self.M.shape
        if self.one_sided and self.num_covecs != self.num_vecs:
            raise ValueError('M must be square for a A one-sided embedder.')
        self.reset()
        self.measure(**pass_args)


    def reset(self):
        self.V = np.random.random((self.d, self.num_vecs)) * 2 - 1
        if self.one_sided:
            self.W = self.V.T
        else:
            self.W = np.random.random((self.num_covecs, self.d)) * 2 - 1
        self.M_hat = None
        self.delta = None
        self.badness = None


    def get_badness(self):
        total_absolute_error = np.sum(abs(self.delta))
        num_cells = (self.M.shape[0] * self.M.shape[1])
        return total_absolute_error / num_cells


    def measure(self, **pass_args):
        self.M_hat = np.dot(self.W, self.V)
        self.delta = self.f_delta(self.M, self.M_hat, **pass_args)
        self.badness = self.get_badness()


    def update(self):

        # For synchronous updates, W gets updated using old vectors.
        if self.synchronous:
            V_to_update_W = self.V.copy()
        else:
            V_to_update_W = self.V

        # Update V
        self.V += self.learning_rate * np.dot(self.W.T, self.delta)

        # Possibly update W.  Apply constraints first if defined.
        if not self.one_sided:
            if self.constrainer is not None:
                self.W, self.V = self.constrainer(
                    self.W,self.V,update_complete=False)
            self.W += self.learning_rate * np.dot(self.delta, V_to_update_W.T)

        # Apply any constraints
        if self.constrainer is not None:
            self.W, self.V = self.constrainer(
                self.W,self.V,update_complete=True)


    def cycle(self, times=1, print_badness=True, pass_args=None):
        pass_args = pass_args or {}
        for i in range(times):
            self.measure(**pass_args)
            self.update()
            if print_badness:
                print(self.badness)


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



