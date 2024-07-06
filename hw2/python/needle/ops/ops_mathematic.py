"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        return out_grad * self.scalar * (node.inputs[0] ** (self.scalar - 1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return array_api.divide(a, b)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[1], -out_grad * node.inputs[0] / (node.inputs[1] ** 2)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return array_api.divide(a, self.scalar)

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if a.ndim <= 1:
            return a

        if self.axes is None:
            dim = a.ndim
            axes = (dim - 2, dim -1)
        else:
            axes = self.axes

        first = axes[0]
        second = axes[1]
        axes = list(range(a.ndim))
        axes[first], axes[second] = axes[second], axes[first]

        return array_api.transpose(a, axes)

    def gradient(self, out_grad, node):
        return transpose(out_grad, axes=self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad : Tensor, node):
        return reshape(out_grad, node.inputs[0].shape)

def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape
        self.axes = []


    def compute(self, a):
        a = array_api.squeeze(a)
        front_append = 0
        back_append = 0
        if len(a.shape) > 0:
            size = a.shape[len(a.shape)-1]
            i = 0
            for i in reversed(range(len(self.shape))):
                if self.shape[i] == size:
                    break
            back_append = len(self.shape) - i - 1
            front_append = i - len(a.shape) + 1
            for i in range(front_append):
                a = array_api.expand_dims(a, 0)
            for i in range(back_append):
                a = array_api.expand_dims(a, len(a.shape))
        else:
            front_append = len(self.shape)

        self.axes = [i for i in range(front_append)]
        self.axes.extend([a.ndim - i - 1 for i in reversed(range(back_append))])
        # Now, broadcast to the target shape
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ret = out_grad
        cutted = 0
        for axis in self.axes:
            ret = summation(ret, axes = axis - cutted)
            cutted += 1
        ret = reshape(ret, node.inputs[0].shape)
        return ret


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis = self.axes)

    def gradient(self, out_grad, node):
        return broadcast_to(out_grad, node.inputs[0].shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        node0, node1 = node.inputs
        tr1 = transpose(node.inputs[1])
        tr0 = transpose(node.inputs[0])
        ret0 = matmul(out_grad, tr1)
        ret1 = matmul(tr0, out_grad)
        while ret0.shape.__len__() > node0.shape.__len__():
            ret0 = summation(ret0, axes=0)
        while ret1.shape.__len__() > node1.shape.__len__():
            ret1 = summation(ret1, axes=0)

        return ret0, ret1


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        return negate(out_grad)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        epsilon = 1e-8
        return out_grad / (node.inputs[0] + epsilon)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.multiply(a, array_api.greater_equal(a, 0))

    def gradient(self, out_grad, node):

        cache_input = node.inputs[0].cached_data

        return out_grad * Tensor(array_api.greater_equal(cache_input, 0))


def relu(a):
    return ReLU()(a)

class Greater_or_equal(TensorOp):
    def compute(self, a):
        return array_api.greater_equal(a, 0)

    def gradient(self, out_grad, node):
        return out_grad * 0
