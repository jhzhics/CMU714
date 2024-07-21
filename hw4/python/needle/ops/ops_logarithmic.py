from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = axes

    def compute(self, Z):
        maxz = Z.max(axis = self.axes)
        maxz_broadcasted = array_api.broadcast_to(maxz, Z.shape)
        Z1 = Z - maxz_broadcasted
        expz = array_api.exp(Z1)
        sumexpz = array_api.sum(expz, axis = self.axes)
        logsumexpz = array_api.log(sumexpz)
        out = logsumexpz + maxz
        if self.axes is None:
            return out
        new_shape = []
        for i in range(len(Z.shape)):
            if i not in self.axes:
                new_shape.append(Z.shape[i])
        out = array_api.reshape(out, tuple(new_shape))
        return out

    def gradient(self, out_grad, node):
        input = node.inputs[0].cached_data
        maxz = input.max(axis = self.axes)
        maxz_broadcasted = array_api.broadcast_to(maxz, input.shape)
        maxz_broadcasted = array_api.reshape(maxz_broadcasted, input.shape)
        input = input - maxz_broadcasted
        input = Tensor(input)
        expinput = exp(input)
        sumexpinput = summation(expinput, axes = self.axes)
        old_shape = input.shape
        new_shape = list(old_shape)
        axes = []
        if self.axes is None:
            self.axes = tuple(range(len(old_shape)))
        elif isinstance(self.axes, int):
            self.axes = [self.axes]
        else:
            self.axes = list(self.axes)
        for i in self.axes:
            new_shape[i] = 1
        out_grad = reshape(out_grad, tuple(new_shape))
        sumexpinput = reshape(sumexpinput, tuple(new_shape))
        out_grad = broadcast_to(out_grad, node.inputs[0].shape)
        sumexpinput = broadcast_to(sumexpinput, input.shape)
        ret = multiply(out_grad, expinput)
        ret = divide(ret, sumexpinput)
        return ret



def logsumexp(a, axes=None):

    return LogSumExp(axes=axes)(a)


