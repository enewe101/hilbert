try:
    import numpy as np
except ImportError:
    np = None


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

    def __init__(self, objective, learning_rate=0.001, momentum_decay=0.9):
        self.objective = objective
        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.allocate()


    def allocate(self):
        self.momenta = []
        self.gradient_steps = []
        param_gradients = self.objective.get_gradient()
        for param in param_gradients:
            self.momenta.append(np.zeros(param.shape))
            self.gradient_steps.append(np.zeros(param.shape))


    def cycle(self, times=1, pass_args=None):
        pass_args = pass_args or {}
        for i in range(times):
            gradients = self.objective.get_gradient(pass_args=pass_args)
            for j in range(len(self.gradient_steps)):
                np.multiply(
                    gradients[j], self.learning_rate,
                    self.gradient_steps[j]
                )
                np.multiply(
                    self.momenta[j], self.momentum_decay,
                    self.momenta[j]
                )
                np.add(
                    self.momenta[j], self.gradient_steps[j],
                    self.momenta[j]
                )
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

    def __init__(self, objective, learning_rate=0.001, momentum_decay=0.9):
        self.objective = objective
        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.allocate()


    def allocate(self):

        self.momenta = []
        self.gradient_steps = []

        param_gradients = self.objective.get_gradient()
        for param in param_gradients:
            self.momenta.append(np.zeros(param.shape))
            self.gradient_steps.append(np.zeros(param.shape))


    def cycle(self, times=1, pass_args=None):
        pass_args = pass_args or {}

        
        for i in range(times):
            # First, decay the momentum, we can then use it as an offset to
            # get the gradient at the current parameters offset by the momentum
            for j in range(len(self.momenta)):
                np.multiply(
                    self.momentum_decay, self.momenta[j], self.momenta[j])

            # Get the gradients offset by the decayed momentum
            gradients = self.objective.get_gradient(
                offsets=self.momenta, pass_args=pass_args)

            # Update the momenta using the gradient.  Note that we have already 
            # applied decay to the momenta.
            for j in range(len(self.gradient_steps)):
                np.multiply(
                    gradients[j], self.learning_rate,
                    self.gradient_steps[j]
                )
                np.add(
                    self.momenta[j], self.gradient_steps[j],
                    self.momenta[j]
                )

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

    def __init__(self, objective, learning_rate=0.001, momentum_decay=0.9):
        self.objective = objective
        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.allocate()


    def allocate(self):

        self.momenta = []
        self.gradient_steps = []
        self.updates = []

        param_gradients = self.objective.get_gradient()
        for param in param_gradients:
            self.momenta.append(np.zeros(param.shape))
            self.gradient_steps.append(np.zeros(param.shape))
            self.updates.append(np.zeros(param.shape))


    def cycle(self, times=1, pass_args=None):
        pass_args = pass_args or {}
        
        for i in range(times):

            # Calculate gradients at accellerated position
            gradients = self.objective.get_gradient(pass_args=pass_args)

            # Calculate update to momenta.
            for j in range(len(self.gradient_steps)):
                np.multiply(
                    self.momenta[j], self.momentum_decay, self.momenta[j])
                np.multiply(
                    gradients[j], self.learning_rate,
                    self.gradient_steps[j]
                )
                np.add(
                    self.momenta[j], self.gradient_steps[j],
                    self.momenta[j]
                )

            # Calculate the update to the accellerated position
            # We need to add the gradient_step and the decayed momentum.
            for j in range(len(self.momenta)):
                self.updates[j] = (
                    self.gradient_steps[j] 
                    + self.momenta[j] * self.momentum_decay
                )

            self.objective.update(*self.updates)









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

    def __init__(self, objective, learning_rate=0.001, momentum_decay=0.9):
        self.objective = objective
        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.allocate()


    def allocate(self):

        self.momenta = []
        self.gradient_steps = []
        self.updates = []
        self.last_gradient = []

        param_gradients = self.objective.get_gradient()
        for param in param_gradients:
            self.momenta.append(np.zeros(param.shape))
            self.gradient_steps.append(np.zeros(param.shape))
            self.updates.append(np.zeros(param.shape))
            self.last_gradient.append(np.zeros(param.shape))


    def clear_momenta(self):
        for j in range(len(self.momenta)):
            np.multiply(self.momenta[j], 0, self.momenta[j])


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
                last_norm_squared += np.sum(np.dot(
                    self.last_gradient[j].T, self.last_gradient[j]))
                norm_squared += np.sum(np.dot(gradients[j].T, gradients[j]))
                product += np.sum(np.dot(gradients[j].T, self.last_gradient[j]))
            norms = np.sqrt(norm_squared) * np.sqrt(last_norm_squared)
            if norms == 0:
                alignment = 1
            else:
                alignment = product / norms
            self.last_gradient = [
                gradients[j].copy() for j in range(len(gradients))]
            print('alignment: %.2f %%' % (alignment * 100))

            use_momentum_decay = max(0, alignment) * self.momentum_decay

            # Calculate update to momenta.
            for j in range(len(self.gradient_steps)):
                np.multiply(
                    self.momenta[j], use_momentum_decay, self.momenta[j])
                np.multiply(
                    gradients[j], self.learning_rate,
                    self.gradient_steps[j]
                )
                np.add(
                    self.momenta[j], self.gradient_steps[j],
                    self.momenta[j]
                )

            # Calculate the update to the accellerated position
            # We need to add the gradient_step and the decayed momentum.
            for j in range(len(self.momenta)):
                self.updates[j] = (
                    self.gradient_steps[j] 
                    + self.momenta[j] * use_momentum_decay
                )

            self.objective.update(*self.updates)

