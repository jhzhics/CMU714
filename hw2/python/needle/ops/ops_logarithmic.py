from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

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
        self.axes = axes

    def compute(self, Z):
        print(Z)
        maxz = array_api.max(Z, axis = self.axes, keepdims = True)
        maxz_broadcasted = array_api.broadcast_to(maxz, Z.shape)
        Z1 = Z - maxz_broadcasted
        expz = array_api.exp(Z1)
        sumexpz = array_api.sum(expz, axis = self.axes)
        logsumexpz = array_api.log(sumexpz)
        maxz = array_api.squeeze(maxz, axis = self.axes)
        return logsumexpz + maxz

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

