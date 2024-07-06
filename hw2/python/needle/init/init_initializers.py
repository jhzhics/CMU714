import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low = -a, high= a, **kwargs) 


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return randn(fan_in, fan_out, std = std, **kwargs)



def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    if nonlinearity == "relu":
        a = math.sqrt(6.0 / fan_in)
    else:
        raise ValueError("Unsupported nonlinearity")
    return rand(fan_in, fan_out, low = -a, high= a, **kwargs)


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    if nonlinearity == "relu":
        std = math.sqrt(2.0 / fan_in)
    else:
        raise ValueError("Unsupported nonlinearity")
    return randn(fan_in, fan_out, std = std, **kwargs)