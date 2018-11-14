try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None

import hilbert as h


def get_solver(solver_type, objective, **solver_args):
    if solver_type == 'sgd':
        return objective
    elif solver_type == 'momentum':
        return MomentumSolver(objective, **solver_args)
    elif solver_type == 'nesterov':
        return NesterovSolver(objective, **solver_args)
    elif solver_type == 'nesterov_cautious':
        return NesterovSolverCautious(objective, **solver_args)
    else:
        raise ValueError('Unexpected solver type: {}'.format(solver_type))


class MomentumSolver(object):
    """
    Accepts an objective object, and finds a local minumum using stochastic
    gradient descent with momentum.

    The objective object should define a get_gradient method, which returns
    a tuple of parameter-gradients, where each parameter gradient is a scalar
    or a numpy array.  Even if there is only one parameter, the gradient must
    be enclosed in a tuple.

    The objective object should also define an update method, which accepts
    a tuple of parameter updates, having the same shape.
    """

    def __init__(
        self, 
        objective,
        learning_rate=0.001,
        momentum_decay=0.9,
        device=None,
        verbose=True
    ):
        self.objective = objective
        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.verbose = verbose
        self.device = device
        self.allocate()


    def allocate(self):
        self.momenta = []
        param_gradients = self.objective.get_gradient()
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        for param in param_gradients:
            self.momenta.append(torch.tensor(
                np.zeros(param.shape), dtype=torch.float32, device=device))


    def cycle(self, times=1, pass_args=None):
        pass_args = pass_args or {}
        for i in range(times):

            gradients = self.objective.get_gradient(pass_args=pass_args)
            for j in range(len(gradients)):
                self.momenta[j] *= self.momentum_decay
                self.momenta[j] += gradients[j] * self.learning_rate

            self.objective.update(*self.momenta)
            if self.verbose:
                print(self.objective.badness.item())



# TODO: Currently not compatible with sharding.  Can be used as long as
#   the objective object has a single shard, or no sharding.
class NesterovSolver(object):
    """
    Accepts an objective object, and finds a local minumum using stochastic
    gradient descent with the Nesterov Accellerated Gradient modification.

    The objective object should define a get_gradient method, which returns
    a tuple of parameter-gradients, where each parameter gradient is a scalar
    or a numpy array.  Even if there is only one parameter, the gradient must
    be enclosed in a tuple.

    The objective object should also define an update method, which accepts
    a tuple of parameter updates, having the same shape.
    """

    def __init__(
        self,
        objective,
        learning_rate=0.001,
        momentum_decay=0.9,
        verbose=True,
        device=None
    ):
        self.objective = objective
        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.device = device
        self.verbose = verbose
        self.allocate()


    def allocate(self):
        self.momenta = []
        param_gradients = self.objective.get_gradient()
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        for param in param_gradients:
            self.momenta.append(torch.zeros(
                param.shape, dtype=torch.float32, device=device))


    def clear_momenta(self):
        self.last_norm = None
        for j in range(len(self.momenta)):
            self.momenta[j][...] = 0


    def cycle(self, times=1, pass_args=None):
        pass_args = pass_args or {}

        
        for i in range(times):

            # First, decay the momentum
            for j in range(len(self.momenta)):
                self.momenta[j] = self.momenta[j] * self.momentum_decay

            # Use the decayed momentum to offset position while getting gradient
            gradients = self.objective.get_gradient(
                offsets=self.momenta, pass_args=pass_args)

            # Update the momenta using the gradient.
            for j in range(len(self.momenta)):
                self.momenta[j] += gradients[j] * self.learning_rate

            self.objective.update(*self.momenta)
            if self.verbose:
                print(self.objective.badness.item())



class NesterovSolverOptimized(object):
    """
    Accepts an objective object, and finds a local minumum using stochastic
    gradient descent with the Nesterov Accellerated Gradient modification.

    The objective object should define a get_gradient method, which returns
    a tuple of parameter-gradients, where each parameter gradient is a scalar
    or a numpy array.  Even if there is only one parameter, the gradient must
    be enclosed in a tuple.

    The objective object should also define an update method, which accepts
    a tuple of parameter updates, having the same shape.
    """

    def __init__(
        self,
        objective,
        learning_rate=0.001,
        momentum_decay=0.9,
        verbose=True,
        device=None
    ):
        self.objective = objective
        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.device = device
        self.verbose = verbose
        self.allocate()


    def allocate(self):
        self.momenta = []
        self.updates = []
        param_gradients = self.objective.get_gradient()
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        for param in param_gradients:
            self.momenta.append(torch.zeros(
                param.shape, dtype=torch.float32, device=device))
            self.updates.append(torch.zeros(
                param.shape, dtype=torch.float32, device=device))


    def cycle(self, times=1, pass_args=None):
        pass_args = pass_args or {}
        
        for i in range(times):

            # Calculate gradients at accellerated position
            gradients = self.objective.get_gradient(pass_args=pass_args)

            # Calculate update to momenta.
            for j in range(len(gradients)):
                self.momenta[j] *= self.momentum_decay
                self.momenta[j] += gradients[j] * self.learning_rate
                self.updates[j] = (
                    gradients[j] * self.learning_rate
                    + self.momenta[j] * self.momentum_decay
                )

            self.objective.update(*self.updates)
            if self.verbose:
                print(self.objective.badness.item())



#TODO: test
class NesterovSolverCautious(object):
    """
    Accepts an objective object, and finds a local minumum using stochastic
    gradient descent with the Nesterov Accellerated Gradient modification.

    The objective object should define a get_gradient method, which returns
    a tuple of parameter-gradients, where each parameter gradient is a scalar
    or a numpy array.  Even if there is only one parameter, the gradient must
    be enclosed in a tuple.

    The objective object should also define an update method, which accepts
    a tuple of parameter updates, having the same shape.
    """

    def __init__(
        self,
        objective,
        learning_rate=0.001,
        momentum_decay=0.9,
        verbose=False,
        device=None
    ):
        self.objective = objective
        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.device = device
        self.verbose = verbose
        self.allocate()


    def allocate(self):
        self.last_norm = None
        self.momenta = []
        self.updates = []
        self.last_gradient = []
        param_gradients = self.objective.get_gradient()
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        for param in param_gradients:
            self.momenta.append(torch.zeros(
                param.shape, dtype=torch.float32, device=device))
            self.updates.append(torch.zeros(
                param.shape, dtype=torch.float32, device=device))
            self.last_gradient.append(torch.zeros(
                param.shape, dtype=torch.float32, device=device))


    def clear_momenta(self):
        self.last_norm = None
        for j in range(len(self.momenta)):
            self.momenta[j][...] = 0
            self.last_gradient[j][...] = 0
            self.updates.append(np.zeros(param.shape))


    def cycle(self, times=1, pass_args=None):
        pass_args = pass_args or {}
        for i in range(times):

            # Calculate gradients at accellerated position
            gradients = self.objective.get_gradient(pass_args=pass_args)

            # Calculate alignment with last gradient
            product = 0.
            norm_squared = sum(
                h.utils.norm(gradients[j])**2
                for j in range(len(gradients))
            )
            product = sum( 
                h.utils.dot(gradients[j], self.last_gradient[j])
                for j in range(len(gradients))
            )

            norm = torch.sqrt(norm_squared)

            if self.last_norm is None:
                alignment = 1
                norms = None
            else:
                norms = self.last_norm * norm
                alignment = product / norms

            #print('\tnorm: ' + str(norm))
            #print('\tlast_norm: ' + str(self.last_norm))
            #print('\tnorms: ' + str(norms))
            #print('\tproduct: ' + str(product))
            #if self.last_norm is not None:
            #    print('\tproduct / norms: ' + str(product / norms))
            #print('\talignment: ' + str(alignment))
            self.last_norm =  norm

            self.last_gradient = [
                gradients[j].clone() for j in range(len(gradients))]

            use_momentum_decay = max(0, alignment) * self.momentum_decay

            # Calculate update to momenta.

            for j in range(len(gradients)):
                self.momenta[j] *= use_momentum_decay
                self.momenta[j] += gradients[j] * self.learning_rate

            for j in range(len(gradients)):
                self.updates[j] = (
                    gradients[j] * self.learning_rate
                    + self.momenta[j] * use_momentum_decay
                )

            self.objective.update(*self.updates)
            if self.verbose:
                print(self.objective.badness.item())





