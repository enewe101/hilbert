try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None

import hilbert as h


def get_solver(solver_type, objective, **solver_args):
    s = {
            'sgd': SgdSolver,
            'momentum': MomentumSolver,
            'nesterov': NesterovSolver,
            'nesterov_cautious': NesterovSolverCautious,
            'adagrad': AdagradSolver,
            'rmsprop': RmspropSolver,
            'adadelta': AdadeltaSolver,
        }

    try: 
        return s[solver_type](objective, **solver_args)
    except KeyError:
        raise ValueError('Unexpected solver type: {}'.format(solver_type))
        


class HilbertSolver(object):
    """
    An (implicitly) abstract class for all Hilbert-based solvers.

    Accepts an objective object, and finds a local minumum using stochastic
    gradient descent with whatever the extended solver is set as.

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
        device=None,
        verbose=True
    ):
        self.objective = objective
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.device = device
        self.crt_pass_args = None # in case gradient needs special args
        self.allocate()


    def allocate(self):
        raise NotImplementedError('This is an interface! Must implement!')


    def get_gradient_update(self):
        raise NotImplementedError('This is an interface! Must implement!')


    def request_gradient(self, offsets=None):
        return self.objective.get_gradient(
            offsets=offsets, pass_args=self.crt_pass_args)


    def _default_allocate(self, attr_names, overwrite=False):
        """
        A default allocation function for a solver that stores parameter-
        level gradient data. Stored as a list of torch zero tensors: self.gdata

        :param: attr_names: list of strings for the variable names
            of the each gradient storage attribute in the inheriting class
        """
        # shouldn't ever call this more than once
        try:
            if self.attr_names and not overwrite:
                raise MemoryError('Error, calling allocate more than once.')
        except AttributeError:
            pass

        # easy check for if someone made a mistake
        if type(attr_names) == str:
            attr_names = [attr_names]
        self.attr_names = attr_names
    
        # set an attribute for each string in the list of names with a list
        for name in attr_names:
            setattr(self, name, [])

        # allocate the memory for each attribute
        param_gradients = self.objective.get_gradient()
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        for param in param_gradients:
            for name in attr_names:
                getattr(self, name).append(torch.zeros(
                    param.shape, dtype=torch.float32, device=device))

    
    def reset(self):
        try:
            for attr in self.attr_names:
                tensor_list = getattr(self, attr)
                for i in range(len(tensor_list)):
                    tensor_list[i][...] = 0. 

        except AttributeError:
            raise AttributeError('Error, no attributes!')


    def cycle(self, times=1, pass_args=None):
        """
        Cycle for some number of times in order to update the same set of
        parameters multiple times in a row. Use pass_args to set any options
        that may need to be used by the get_gradient function.
        """
        pass_args = pass_args or {}
        for _ in range(times):

            # may be used when the inheriting class uses request_gradient()
            self.crt_pass_args = pass_args

            # get whatever updates the inheritor will make, then pass them
            updates = self.get_gradient_update()
            self.objective.update(*updates)

            if self.verbose:
                print(self.objective.badness.item())



class SgdSolver(HilbertSolver):
    """
    Most simple HilbertSolver class, basically just a wrapper class over
    the objective's implicit SGD optimization.
    """

    def __init__(
        self,
        objective,
        learning_rate=0.001,
        device=None,
        verbose=True
    ):
        super(SgdSolver, self).__init__(objective,
            learning_rate=learning_rate, device=device, verbose=verbose)


    def allocate(self, overwrite=False):
        # to integrate with the rest of the solver design pattern
        try:
            getattr(self, 'attr_names')
            raise MemoryError('Allocation has already occurred!')
        except AttributeError:
            pass

        # nothing to allocate for sgd!
        self.attr_names = []


    def get_gradient_update(self):
        gradients = self.request_gradient()
        updates = []
        for j in range(len(gradients)):
            updates.append(gradients[j] * self.learning_rate)
        return updates



class MomentumSolver(HilbertSolver):
    """
    Uses basic momentum to update the gradients.
    """

    def __init__(
        self,
        objective,
        learning_rate=0.001,
        momentum_decay=0.9,
        device=None,
        verbose=True
    ):
        super(MomentumSolver, self).__init__(objective,
            learning_rate=learning_rate, device=device, verbose=verbose)
        self.momentum_decay = momentum_decay


    def allocate(self):
        self._default_allocate(['momenta'])


    def get_gradient_update(self):
        gradients = self.request_gradient()
        for j in range(len(gradients)):
            self.momenta[j] *= self.momentum_decay
            self.momenta[j] += gradients[j] * self.learning_rate
        return self.momenta



class AdagradSolver(HilbertSolver):
    """
    Finds a local minumum using stochastic gradient descent with AdaGrad.
    See the following link:
    https://en.wikipedia.org/wiki/Stochastic_gradient_descent#AdaGrad

    AdaGrad uses O(n) memory, where n is the size of the gradient. It is
    efficient as it replicates using a O(n**2) matrix by storing the diagonal
    of the outer product gradient matrix, which is maintained over time.

    Note that AdaGrad involves a division of potentially zero-valued items;
    thus, just as PyTorch does, we add an epsilon (1e-10) to the denominator
    of the fraction in order to avoid any divide-by-zero errors. See the
    PyTorch implementation of AdaGrad to confirm that epsilon is added:
    https://github.com/pytorch/pytorch/blob/master/torch/optim/adagrad.py
    """

    def __init__(
        self,
        objective,
        learning_rate=0.01,
        device=None,
        verbose=True
    ):
        super(AdagradSolver, self).__init__(objective,
            learning_rate=learning_rate, device=device, verbose=verbose)
        self.epsilon = 1e-10


    def allocate(self):
        self._default_allocate(['adagrad'])


    def get_gradient_update(self):
        # Update the normed gradient holder vector, adag.
        # It is the diagonal of the outer product gradient matrix.
        gradients = self.request_gradient()
        updates = []
        for j in range(len(gradients)):
            self.adagrad[j] += gradients[j] ** 2

            # note that the multiplier is a vector and that we are doing a
            # component-wise multiplication
            rootinvdiag = 1. / (self.adagrad[j] + self.epsilon).sqrt() 
            updates.append(gradients[j] * rootinvdiag * self.learning_rate)
        return updates



class RmspropSolver(HilbertSolver):
    def __init__(
        self,
        objective,
        learning_rate=0.01,
        gamma=0.9,
        device=None,
        verbose=True
    ):
        super(RmspropSolver, self).__init__(objective,
            learning_rate=learning_rate, device=device, verbose=verbose)
        self.gamma = gamma
        self.epsilon = 1e-10


    def allocate(self):
        self._default_allocate(['adadecay'])


    def get_gradient_update(self):
        # Update the normed gradient holder vector, adag.
        # It is the diagonal of the outer product gradient matrix.
        gradients = self.request_gradient()
        updates = []
        for j in range(len(gradients)):
            self.adadecay[j] = (
                    (self.gamma * self.adadecay[j] ) + 
                    ((1. - self.gamma) * (gradients[j] ** 2) )
                )

            # note that the multiplier is a vector and that we are doing a
            # component-wise multiplication
            rootinvdiag = 1. / (self.adadecay[j] + self.epsilon).sqrt()
            updates.append(gradients[j] * rootinvdiag * self.learning_rate)
        return updates



class AdadeltaSolver(HilbertSolver):
    def __init__(
        self,
        objective,
        learning_rate=1,
        gamma=0.9,
        device=None,
        verbose=True
    ):
        super(AdadeltaSolver, self).__init__(objective,
            learning_rate=learning_rate, device=device, verbose=verbose)
        self.gamma = gamma
        self.epsilon = 1e-10


    def allocate(self):
        self._default_allocate(['adadecay', 'updatedecay'])


    def get_gradient_update(self):
        # Update the normed gradient holder vector, adag.
        # It is the diagonal of the outer product gradient matrix.
        gradients = self.request_gradient()
        updates = []
        for j in range(len(gradients)):
            
            # first get adadecay, like in RMS prop
            self.adadecay[j] = (
                    (self.gamma * self.adadecay[j] ) + 
                    ((1. - self.gamma) * (gradients[j] ** 2) )
                )
            rootinv_ada = 1. / (self.adadecay[j] + self.epsilon).sqrt()
            
            # now we build up the update vector
            prev_update = self.updatedecay[j]
            crt_update = (
                    (prev_update + self.epsilon).sqrt() 
                    * rootinv_ada * gradients[j]
                )
            updates.append(crt_update)
             
            # nowwww update the updatedecay!
            self.updatedecay[j] = (
                    (self.gamma * prev_update) +
                    ((1. - self.gamma) * (crt_update ** 2))
                )

        return updates





# TODO: Currently not compatible with sharding.  Can be used as long as
#   the objective object has a single shard, or no sharding.

class NesterovSolver(HilbertSolver):
    """
    Accepts an objective object, and finds a local minumum using stochastic
    gradient descent with the Nesterov Accellerated Gradient modification.
    """

    def __init__(
        self,
        objective,
        learning_rate=0.001,
        momentum_decay=0.9,
        verbose=True,
        device=None
    ):
        super(NesterovSolver, self).__init__(objective,
            learning_rate=learning_rate, device=device, verbose=verbose)
        self.momentum_decay = momentum_decay


    def allocate(self):
        self._default_allocate(['momenta'])


    def clear_momenta(self):
        self.last_norm = None
        self.reset()


    def get_gradient_update(self):
        # First, decay the momentum
        for j in range(len(self.momenta)):
            self.momenta[j] = self.momenta[j] * self.momentum_decay

        # Use the decayed momentum to offset position while getting gradient
        gradients = self.request_gradient(offsets=self.momenta)

        # Update the momenta using the gradient.
        for j in range(len(self.momenta)):
            self.momenta[j] += gradients[j] * self.learning_rate

        return self.momenta



class NesterovSolverOptimized(HilbertSolver):
    """
    Accepts an objective object, and finds a local minumum using stochastic
    gradient descent with the Nesterov Accellerated Gradient modification.
    """

    def __init__(
        self,
        objective,
        learning_rate=0.001,
        momentum_decay=0.9,
        verbose=True,
        device=None
    ):
        super(NesterovSolverOptimized, self).__init__(objective,
            learning_rate=learning_rate, device=device, verbose=verbose)
        self.momentum_decay = momentum_decay


    def allocate(self):
        self._default_allocate(['momenta', 'updates'])


    def get_gradient_update(self):
        gradients = self.request_gradient()

        # Calculate update to momenta and yield those.
        for j in range(len(gradients)):
            self.momenta[j] *= self.momentum_decay
            self.momenta[j] += gradients[j] * self.learning_rate
            self.updates[j] = (
                gradients[j] * self.learning_rate
                + self.momenta[j] * self.momentum_decay
            )
        return self.updates



#TODO: test
class NesterovSolverCautious(HilbertSolver):
    """
    Accepts an objective object, and finds a local minumum using stochastic
    gradient descent with the Nesterov Accellerated Gradient modification.
    """

    def __init__(
        self,
        objective,
        learning_rate=0.001,
        momentum_decay=0.9,
        verbose=False,
        device=None
    ):
        super(NesterovSolverCautious, self).__init__(objective,
            learning_rate=learning_rate, device=device, verbose=verbose)
        self.momentum_decay = momentum_decay


    def allocate(self):
        self.last_norm = None
        self._default_allocate(['momenta', 'updates', 'last_gradient'])


    def clear_momenta(self):
        self.last_norm = None
        self.reset()


    def get_gradient_update(self):
        # Calculate gradients at accellerated position
        gradients = self.request_gradient()

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

        return self.updates
