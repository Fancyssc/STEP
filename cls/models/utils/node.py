from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.strategy.surrogate import *

# for debugging
from spikingjelly.clock_driven.neuron import LIFNode as sj_LIFNode
from spikingjelly.clock_driven import neuron_kernel
from spikingjelly.clock_driven import surrogate


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

class BaseNode_Torch(BaseNode):
    """
    Base Node for Layer by Layer forward propagation
    New OP added to adapt specific dim needed by Spiking Transformers
    :param threshold: The threshold that a neuron needs to reach in order to fire an action potential.
    :param step: The number of time steps that the neuron will be simulated for.
    """
    def __init__(self, threshold=1.0, step=10, layer_by_layer=True, mem_detach=True):
        super().__init__(threshold=threshold, step=step, layer_by_layer=layer_by_layer, mem_detach=mem_detach)

    def rearrange2node(self, inputs):
        if self.groups != 1:
            if len(inputs.shape) == 4:
                outputs = rearrange(inputs, 'b (c t) w h -> t b c w h', t=self.step)
            elif len(inputs.shape) == 2:
                outputs = rearrange(inputs, 'b (c t) -> t b c', t=self.step)
            else:
                raise NotImplementedError

        elif self.layer_by_layer:
            if len(inputs.shape) == 4:
                outputs = rearrange(inputs, '(t b) c w h -> t b c w h', t=self.step)
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs, '(t b) n c -> t b n c', t=self.step)
            elif len(inputs.shape) == 2:
                outputs = rearrange(inputs, '(t b) c -> t b c', t=self.step)
            else:
                raise NotImplementedError


        else:
            outputs = inputs

        return outputs

    def rearrange2op(self, inputs):
        if self.groups != 1:
            if len(inputs.shape) == 5:
                outputs = rearrange(inputs, 't b c w h -> b (c t) w h')
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs, ' t b c -> b (c t)')
            else:
                raise NotImplementedError
        elif self.layer_by_layer:
            if len(inputs.shape) == 5:
                outputs = rearrange(inputs, 't b c w h -> (t b) c w h')

            # adapt Spikformer
            elif len(inputs.shape) == 4:
                outputs = rearrange(inputs, ' t b n c -> (t b) n c')
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs, ' t b c -> (t b) c')
            else:
                raise NotImplementedError

        else:
            outputs = inputs
        return outputs

class LIFNode(BaseNode_Torch):
    """
    Leaky Integrate-and-Fire (LIF) neuron model
    :param threshold: The threshold that a neuron needs to reach in order to fire an action potential.
    :param step: The number of time steps that the neuron will be simulated for.
    :param tau: The time constant of the neuron.
    :param act_fun: The activation function of the neuron.
    """
    def __init__(self, threshold=1.0, step=4, layer_by_layer=True, tau=2., act_fun=Sigmoid_Grad,mem_detach=False, *args,
                 **kwargs):
        super().__init__(threshold=threshold, step=step, layer_by_layer=layer_by_layer, mem_detach=mem_detach)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun()

    def integral(self, inputs):
        self.mem = self.mem + (inputs - self.mem) / self.tau

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach())

# SpikingJelly Node
# # for debugging
class LIFNode_Spikingjelly(sj_LIFNode):
    def __init__(self, step=4, threshold=1., tau=2.,detach_reset=True, backend='torch', **kwargs):
        super().__init__(v_threshold=threshold, tau=tau, detach_reset=detach_reset, v_reset=0. , surrogate_function=surrogate.Sigmoid())
        # self.register_memory('v_seq', None)
        self.step = step
        self.backend = backend
    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        x_shape = x_seq.shape

        if self.backend == 'torch':
            #方便debug 多封装一层
            x_seq = x_seq.reshape(self.step,-1, *x_shape[1:]).contiguous() #适配braincog框架的输入

            spike_seq = []
            self.v_seq = []
            for t in range(x_seq.shape[0]):
                # print(self)
                spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
                self.v_seq.append(self.v.unsqueeze(0))

            spike_seq = torch.cat(spike_seq, 0)
            self.v_seq = torch.cat(self.v_seq, 0)

            return spike_seq.flatten(0,1)

        else:
            raise NotImplementedError(self.backend)


# # for debugging
class LIFNode_Spikingjelly_Cupy(sj_LIFNode):
    def __init__(self, step=4, threshold=1., tau=2.,detach_reset=True, backend='cupy', **kwargs):
        super().__init__(v_threshold=threshold, tau=tau, detach_reset=detach_reset, v_reset=0. , surrogate_function=surrogate.Sigmoid())
        self.register_memory('v_seq', None)
        self.step = step
        self.backend = backend
    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        x_shape = x_seq.shape

        if self.backend == 'cupy':
            #方便debug 多封装一层
            x_seq = x_seq.reshape(self.step,-1, *x_shape[1:]).contiguous() #适配braincog框架的输入

            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)

            spike_seq, self.v_seq = neuron_kernel.MultiStepLIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.decay_input, self.tau, self.v_threshold, self.v_reset,
                self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            self.v_seq = self.v_seq.reshape(x_seq.shape)

            self.v = self.v_seq[-1].clone()

            return spike_seq.flatten(0,1)


        else:
            raise NotImplementedError(self.backend)