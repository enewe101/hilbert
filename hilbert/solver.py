try:
    import numpy as np
except ImportError:
    np = None

import torch
import hilbert as h

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
        self, objective, learning_rate=0.001, momentum_decay=0.9,
        implementation='torch', device='cuda'
    ):
        self.objective = objective
        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.implementation = implementation
        self.device = device
        h.utils.ensure_implementation_valid(implementation)
        self.allocate()


    def allocate(self):
        self.momenta = []
        param_gradients = self.objective.get_gradient()
        for param in param_gradients:
            if self.implementation == 'torch':
                self.momenta.append(
                    torch.tensor(np.zeros(param.shape), dtype=torch.float32)
                )
            else:
                self.momenta.append(np.zeros(param.shape))


    def cycle(self, times=1, pass_args=None):
        pass_args = pass_args or {}
        for i in range(times):

            gradients = self.objective.get_gradient(pass_args=pass_args)
            for j in range(len(gradients)):
                self.momenta[j] *= self.momentum_decay
                self.momenta[j] += gradients[j] * self.learning_rate

            self.objective.update(*self.momenta)



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
        self, objective, learning_rate=0.001, momentum_decay=0.9,
        implementation='torch', device='cuda'
    ):
        self.objective = objective
        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.implementation = implementation
        self.device = device
        h.utils.ensure_implementation_valid(implementation)
        self.allocate()


    def allocate(self):
        self.momenta = []
        param_gradients = self.objective.get_gradient()
        for param in param_gradients:
            if self.implementation == 'torch':
                self.momenta.append(
                    torch.zeros(param.shape, device=self.device))
            else:
                self.momenta.append(np.zeros(param.shape))


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
        self, objective, learning_rate=0.001, momentum_decay=0.9,
        implementation='torch', device='cuda'
    ):
        self.objective = objective
        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.implementation = implementation
        self.device = device
        h.utils.ensure_implementation_valid(implementation)
        self.allocate()


    def allocate(self):
        self.momenta = []
        self.updates = []
        param_gradients = self.objective.get_gradient()
        for param in param_gradients:
            if self.implementation == 'torch':
                self.momenta.append(
                    torch.zeros(param.shape, device=self.device))
                self.updates.append(
                    torch.zeros(param.shape, device=self.device))
            else:
                self.momenta.append(np.zeros(param.shape))
                self.updates.append(np.zeros(param.shape))


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
        self, objective, learning_rate=0.001, momentum_decay=0.9,
        implementation='torch', device='cuda'
    ):
        self.objective = objective
        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.implementation = implementation
        self.device = device
        h.utils.ensure_implementation_valid(implementation)
        self.allocate()


    def allocate(self):
        self.momenta = []
        self.updates = []
        self.last_gradient = []
        param_gradients = self.objective.get_gradient()
        for param in param_gradients:
            if self.implementation == 'torch':
                self.momenta.append(
                    torch.zeros(param.shape, device=self.device))
                self.updates.append(
                    torch.zeros(param.shape, device=self.device))
                self.last_gradient.append(
                    torch.zeros(param.shape, device=self.device))
            else:
                self.momenta.append(np.zeros(param.shape))
                self.updates.append(np.zeros(param.shape))
                self.last_gradient.append(np.zeros(param.shape))


    def clear_momenta(self):
        for j in range(len(self.momenta)):
            self.momenta[j][...] = 0


    def cycle(self, times=1, pass_args=None):
        pass_args = pass_args or {}
        
        for i in range(times):

            # Calculate gradients at accellerated position
            gradients = self.objective.get_gradient(pass_args=pass_args)

            # Calculate alignment with last gradient
            norm_squared = 0
            last_norm_squared = 0
            product = 0
            for j in range(len(self.last_gradient)):
                # TODO: handle non-matrix values (scalar and vector)
                last_norm_squared += torch.sum(
                    torch.mm(self.last_gradient[j].t(), self.last_gradient[j]))
                norm_squared += torch.sum(
                    torch.mm(gradients[j].t(), gradients[j]))
                product += torch.sum(
                    torch.mm(gradients[j].t(), self.last_gradient[j]))
            norms = torch.sqrt(norm_squared) * torch.sqrt(last_norm_squared)
            if norms == 0:
                alignment = 1
            else:
                alignment = product / norms

            self.last_gradient = [
                gradients[j].clone() for j in range(len(gradients))]
            print('alignment: %.2f %%' % (alignment * 100))
            use_momentum_decay = max(0, alignment) * self.momentum_decay

            # Calculate update to momenta.
            for j in range(len(gradients)):
                self.momenta[j] *= use_momentum_decay
                self.momenta[j] += gradients[j] * self.learning_rate
                self.updates[j] = (
                    gradients[j] * self.learning_rate
                    + self.momenta[j] * use_momentum_decay
                )

            self.objective.update(*self.updates)





