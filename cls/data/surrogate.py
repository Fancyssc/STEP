import torch

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
