
"""
This module provides classes that schedule a varying learning rate during
training.  The BaseScheduler provides a constant learning rate, and can be used
when it is necessary to support the interface of a learning rate scheduler while
using a constant learning rate.
"""

from collections import deque
import numpy as np


class BaseScheduler:
    """
    Fulfills Scheduler interface, but provides a constant learning rate.
    """
    def get_rate(self, badness):
        raise NotImplementedError


class ConstantScheduler(BaseScheduler):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def get_rate(self, badness):
        return self.learning_rate


class PlateauScheduler(BaseScheduler):

    def __init__(
        self, initial_rate, plateau_threshold=0.00001, maxlen=50, wait_time=1000
    ):
        self.initial_rate = initial_rate
        self.plateau_threshold = plateau_threshold
        self.current_rate = initial_rate
        self.maxlen = maxlen
        self.relative_changes = deque(maxlen=maxlen)
        self.last_badness = None
        self.waiting_iters=0
        self.wait_time = wait_time


    def get_rate(self, badness):

        self.waiting_iters += 1

        if badness is None:
            return self.current_rate

        if self.last_badness is None:
            self.last_badness = badness
            return self.current_rate

        self.relative_changes.append((self.last_badness - badness) / badness)
        self.last_badness = badness

        if len(self.relative_changes) < self.maxlen:
            return self.current_rate

        if self.waiting_iters < self.wait_time:
            return self.current_rate
        if self.waiting_iters == self.wait_time:
            print('Done waiting')

        average_relative_change = sum(self.relative_changes) / self.maxlen
        if average_relative_change < self.plateau_threshold:
            self.current_rate = self.current_rate / 10
            self.waiting_iters=0
            self.relative_changes.clear()
            print('average_relative_change', average_relative_change)
            print('new rate', self.current_rate)

        return self.current_rate
        



class JostleScheduler(BaseScheduler):
    """
    Scans learning rate from a high value, to a low value, repeatedly.  Each
    scan is a "sprint".  The first sprint starts with ``initial_rate`` and
    linearly decreases it toward zero over ``sprint_length`` iterations.  Each
    successive sprint cuts the initial rate in half, and doubles the sprint
    length.  The final sprint provides an exponentially decaying rate that
    plateaus to ``final_rate``.
    """

    def __init__(
        self, initial_rate, sprint_length, num_sprints=5, final_rate=None
    ):
        self.initial_rate = initial_rate
        self.sprint_rate = initial_rate
        self.sprint_length = sprint_length
        self.num_sprints = num_sprints
        self.sprint = 1
        self.cycles = 0

        self.final_rate = final_rate
        if self.final_rate is None:
            self.final_rate = 1e-3 * self.initial_rate


    def get_rate(self, badness):
        self.cycles += 1

        if self.sprint < self.num_sprints:
            fraction = (self.sprint_length - self.cycles) / self.sprint_length
            if self.cycles == self.sprint_length:
                self.sprint += 1
                self.sprint_length *= 2
                self.sprint_rate /= 2
                self.cycles = 0

            return (
                self.sprint_rate - self.final_rate) * fraction + self.final_rate

        fraction = np.e**(-2*self.cycles/self.sprint_length)
        return self.sprint_rate * fraction + self.final_rate



