"""This file defies specific implementations of devices when using numpy as NDArray backend.
"""
import numpy
import functools


class Device:
    """Baseclass of all device"""


class CPUDevice(Device):
    """Represents data that sits in CPU"""

    @staticmethod
    def Array(size):
        return numpy.empty(size)
    
    @staticmethod
    def from_numpy(array : numpy.ndarray, handle : numpy.ndarray):
        size1 = functools.reduce(lambda x, y: x * y, array.shape)
        size2 = functools.reduce(lambda x, y: x * y, handle.shape)
        assert size1 == size2, "Size mismatch"
        numpy.copyto(handle, array.reshape(handle.shape))

    def __repr__(self):
        return "needle.cpu()"

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CPUDevice)
    

    def enabled(self):
        return True

    def zeros(self, *shape, dtype="float32"):
        return numpy.zeros(shape, dtype=dtype)

    def ones(self, *shape, dtype="float32"):
        return numpy.ones(shape, dtype=dtype)

    def randn(self, *shape):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return numpy.random.randn(*shape)

    def rand(self, *shape):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return numpy.random.rand(*shape)

    def one_hot(self, n, i, dtype="float32"):
        return numpy.eye(n, dtype=dtype)[i]

    def empty(self, shape, dtype="float32"):
        return numpy.empty(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype="float32"):
        return numpy.full(shape, fill_value, dtype=dtype)


def cpu():
    """Return cpu device"""
    return CPUDevice()


def default_device():
    return cpu()


def all_devices():
    """return a list of all available devices"""
    return [cpu()]
