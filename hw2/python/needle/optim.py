"""Optimization module"""
import needle as ndl
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params : list[ndl.Tensor] = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for w in self.params:
            if not w in self.u:
                self.u[w] = ndl.zeros_like(w)
            if self.weight_decay > 0:
                grad = w.grad.data + self.weight_decay * w.data
            else:
                grad = w.grad.data

            self.u[w] = self.momentum * self.u[w] + (1 - self.momentum) * grad
            w.data = w.data - self.lr * self.u[w]


            



    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m : dict[ndl.Tensor, ndl.Tensor]= {}
        self.v : dict[ndl.Tensor, ndl.Tensor]= {}

    def step(self):
        self.t += 1
        for w in self.params:
            grad = (w.grad + self.weight_decay * w.data).detach()
            if not w in self.m:
                self.m[w] = ndl.zeros_like(w)
                self.v[w] = ndl.zeros_like(w)

            self.m[w] =(self.beta1*self.m[w] + (1-self.beta1)*grad).detach()
            self.v[w] = (self.beta2*self.v[w] + (1-self.beta2)*grad**2).detach()
            u_hat = self.m[w] / (1-self.beta1**self.t)
            v_hat = self.v[w] / (1-self.beta2**self.t)
            w.data -= (self.lr * u_hat / (v_hat**0.5 + self.eps)).detach()
