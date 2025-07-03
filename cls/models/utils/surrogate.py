import torch
from braincog.base.strategy.surrogate import *


class HeavisideSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sigmoid_grad = torch.sigmoid(input) * (1 - torch.sigmoid(input))
        return grad_output * sigmoid_grad



class HeavisideParametricSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, v_threshold):
        ctx.save_for_backward(input)
        ctx.k = k
        ctx.v_threshold = v_threshold
        return (input >= v_threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        k = ctx.k
        v_threshold = ctx.v_threshold
        sigmoid_grad = k * torch.sigmoid(k * (input - v_threshold)) * (1 - torch.sigmoid(k * (input - v_threshold)))
        return grad_output * sigmoid_grad, None, None


class Sigmoid_Grad(SurrogateFunctionBase):
    """
    Sigmoid activation function with gradient
    Overwrite sigmoid function in BrainCog
    """

    def __init__(self, alpha=4., requires_grad=False):
        super().__init__(alpha=alpha, requires_grad=requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return sigmoid.apply(x, alpha)


class QGate_Grad(SurrogateFunctionBase):

    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha=alpha, requires_grad=requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return quadratic_gate.apply(x, alpha)

class rectangle(torch.autograd.Function):
    """
    CLIF act func
    """

    @staticmethod
    def forward(ctx, x, vth):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.vth = vth
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            x = ctx.saved_tensors[0]
            mask1 = (x.abs() > ctx.vth / 2)
            mask_ = mask1.logical_not()
            grad_x = grad_output * x.masked_fill(
                mask_, 1. / ctx.vth).masked_fill(mask1, 0.)
        return grad_x, None


class Rectangle(SurrogateFunctionBase):
    """
    CLIF act func class
    """

    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return rectangle.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return torch.min(torch.max(1. / alpha * x, 0.5), -0.5)


class SpikeAct_extended(torch.autograd.Function):
    '''
    GLIF act function
    solving the non-differentiable term of the Heavisde function
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = torch.gt(input, 0.)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone()

        # hu is an approximate func of df/du in linear formulation
        hu = abs(input) < 0.5
        hu = hu.float()

        # arctan surrogate function
        # hu =  1 / ((input * torch.pi) ** 2 + 1)

        # triangles
        # hu = (1 / gamma_SG) * (1 / gamma_SG) * ((gamma_SG - input.abs()).clamp(min=0))

        return grad_input * hu