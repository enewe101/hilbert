import abc

"""
This file is used to store some generic interfaces that
most classes end up extending. This simply centralizes
the design patterns used in this codebase.
"""


class Describable(abc.ABC):

    @abc.abstractmethod
    def describe(self):
        return

