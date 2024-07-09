"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x
    
# class Linear(Module):
#     def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features

#         ### BEGIN YOUR SOLUTION
#         self.weight = Parameter(init.kaiming_uniform(in_features, out_features, requires_grad=True))
#         self.bias = Parameter(init.kaiming_uniform(out_features, 1, requires_grad=True).transpose()) if bias else None
#         ### END YOUR SOLUTION

#     def forward(self, X: Tensor) -> Tensor:
#         ### BEGIN YOUR SOLUTION
#         # X: N x in_feat, weight: in_feat x out_feat
#         out = X.matmul(self.weight)
#         if self.bias:
#             out += self.bias.broadcast_to(out.shape)
#         return out
#         ### END YOUR SOLUTION

class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1).transpose())

    def forward(self, X: Tensor) -> Tensor:
        if self.bias is not None:
            z = ops.matmul(X, self.weight)
            bias_after_broadcast = ops.broadcast_to(self.bias, z.shape)
            z = ops.add(z, bias_after_broadcast)
            return z
        else:
            return ops.matmul(X, self.weight)

class Flatten(Module):
    def forward(self, X):
        return ops.reshape(X, (X.shape[0], -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        logsumexp = ops.logsumexp(logits, axes = (len(logits.shape) - 1))
        onehot = init.one_hot(logits.shape[-1], y, dtype=logits.dtype)
        zy = ops.multiply(logits, onehot)
        zy = ops.summation(zy, axes = (len(logits.shape) - 1))
        ret = logsumexp - zy
        loss = ops.summation(ret)
        batch_size = 1
        for i in range(len(logits.shape) - 1):
            batch_size *= logits.shape[i]
        factor = np.array([1.0 / batch_size], dtype=loss.dtype)
        loss = ops.mul_scalar(loss,  factor)
        return loss


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.dim
        shape = list(x.shape)
        shape[0] = 1
        mean1 = ops.summation(x, axes = 0) / x.shape[0]
        
        mean = ops.reshape(mean1, shape)
        mean = ops.broadcast_to(mean, x.shape)
        x1 = x - mean
        x2 = x1 * x1
        var1 = ops.summation(x2, axes = 0) / x.shape[0]
        var = ops.reshape(var1, shape)
        var = ops.broadcast_to(var, x.shape)
        std = (var + self.eps) ** 0.5
        x_hat = x1 / std

        self.running_mean = (self.momentum * mean1
                             + (1 - self.momentum) * self.running_mean).detach()
    
        self.running_var = (self.momentum * var1
                            + (1 - self.momentum) * self.running_var).detach()

        if self.training:
            y = self.weight * x_hat + self.bias
        else:
            y = self.weight * (x - self.running_mean) / (self.running_var + self.eps) ** 0.5 + self.bias
        return y


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.beta = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.dim 
        shape = list(x.shape)
        shape[-1] = 1
        mean = ops.summation(x, axes = len(x.shape) - 1) / x.shape[-1]
        mean = ops.reshape(mean, shape)
        mean = ops.broadcast_to(mean, x.shape)
        x1 = x - mean
        x2 = x1 * x1
        var = ops.summation(x2, axes = len(x.shape) - 1) / x.shape[-1]
        var = ops.reshape(var, shape)
        var = ops.broadcast_to(var, x.shape)
        std = (var + self.eps) ** 0.5
        x_hat = x1 / std
        y = self.gamma * x_hat + self.beta
        return y


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = np.random.rand(*x.shape) <= 1 - self.p
            mask = mask.astype(x.dtype)
            mask = mask / (1 - self.p)
            mask = Tensor(mask)
            return x * mask
        else:
            return x

class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)

