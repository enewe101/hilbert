class TempScheduler:

    def __init__(self, tempered_loss, milestones, temperatures):
        """
        Accepts a ``TemperedLoss``, a list of milestones, and a list of
        temperatures, which should both be the same length.  The milestones are
        epoch numbers; when a given milestone is reached, the corresponding
        temperature will be applied
        """
        self.loss = tempered_loss
        self.milestones = milestones
        self.temperatures = temperatures
        self.pointer = 0
        self.cur_epoch = -1
        self.step()

    def step(self):
        self.cur_epoch += 1

        # After reaching all milestones, do nothing.
        if self.pointer == len(self.milestones):
            return

        # On reaching milestone, update temp, and point to new milestone.
        if self.milestones[self.pointer] <= self.cur_epoch:
            self.loss.temperature = self.temperatures[self.pointer]
            self.pointer += 1


class LearningRateScheduler:
    def __init__(self,
                 optimizer,
                 start_lr,
                 num_epochs,
                 end_lr=0):
        """
        Base class for all learning rate schedulers.

        :param optimizer: Optimizer used for training objective.
        :param start_lr: Initial learning rate.
        :param num_epochs: Number of epochs
        :param end_lr: Stationary state learning rate
        """

        self.opt = optimizer
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_epochs = num_epochs
        self.cur_epoch = -1
        self.step()

    def step(self):
        raise NotImplementedError('Not implemented!')


class LinearLRScheduler(LearningRateScheduler):

    def __init__(self, optimizer, start_lr, num_epochs, end_lr=0):
        '''

        :param optimizer: Optimizer used for training objective.
        :param start_lr: Initial learning rate.
        :param num_epochs: Number of epochs
        :param end_lr: Stationary state learning rate
        '''
        super(LinearLRScheduler, self).__init__(
            optimizer,
            start_lr,
            num_epochs,
            end_lr)

    def step(self):
        self.cur_epoch += 1
        if self.cur_epoch < self.num_epochs:
            fraction_left = 1 - self.cur_epoch / self.num_epochs
            cur_lr = self.end_lr + (self.start_lr - self.end_lr) * fraction_left
        else:
            cur_lr = self.end_lr

        for param_group in self.opt.param_groups:
            param_group['lr'] = cur_lr


class InverseLRScheduler(LearningRateScheduler):
    def __init__(self, optimizer, start_lr, num_epochs):
        """
        Based on the learning rate scheduler O(1/t) from Bergstra and Bengio (2012), where keeping learning rate
        constant for the first [num_epochs] updates.
        cur_lr = start_lr * num_epoch/max(t, num_epoch)

        :param optimizer: Optimizer used for training objective.
        :param start_LR: Initial learning rate.
        :param num_epoch: Number of epochs keeping learning rate constant
        """

        super(InverseLRScheduler, self).__init__(
            optimizer,
            start_lr,
            num_epochs)

    def step(self):
        self.cur_epoch += 1
        cur_lr = self.start_lr * self.num_epochs / max(self.cur_epoch,
                                                       self.num_epochs)
        for param_group in self.opt.param_groups:
            param_group['lr'] = cur_lr
