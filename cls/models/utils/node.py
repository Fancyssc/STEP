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

    def __init__(self,
                 threshold=1.0,
                 step=10,
                 layer_by_layer=True,
                 mem_detach=True):
        super().__init__(threshold=threshold,
                         step=step,
                         layer_by_layer=layer_by_layer,
                         mem_detach=mem_detach)

    def rearrange2node(self, inputs):
        if self.groups != 1:
            if len(inputs.shape) == 4:
                outputs = rearrange(inputs,
                                    'b (c t) w h -> t b c w h',
                                    t=self.step)
            elif len(inputs.shape) == 2:
                outputs = rearrange(inputs, 'b (c t) -> t b c', t=self.step)
            else:
                raise NotImplementedError

        elif self.layer_by_layer:
            if len(inputs.shape) == 4:
                outputs = rearrange(inputs,
                                    '(t b) c w h -> t b c w h',
                                    t=self.step)
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs,
                                    '(t b) n c -> t b n c',
                                    t=self.step)
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

    def __init__(self,
                 threshold=1.0,
                 step=4,
                 layer_by_layer=True,
                 tau=2.,
                 act_fun=Sigmoid_Grad,
                 mem_detach=False,
                 *args,
                 **kwargs):
        super().__init__(threshold=threshold,
                         step=step,
                         layer_by_layer=layer_by_layer,
                         mem_detach=mem_detach)
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

    def __init__(self,
                 step=4,
                 threshold=1.,
                 tau=2.,
                 detach_reset=True,
                 backend='torch',
                 **kwargs):
        super().__init__(v_threshold=threshold,
                         tau=tau,
                         detach_reset=detach_reset,
                         v_reset=0.,
                         surrogate_function=surrogate.Sigmoid())
        # self.register_memory('v_seq', None)
        self.step = step
        self.backend = backend

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        x_shape = x_seq.shape

        if self.backend == 'torch':
            #方便debug 多封装一层
            x_seq = x_seq.reshape(self.step, -1,
                                  *x_shape[1:]).contiguous()  #适配braincog框架的输入

            spike_seq = []
            self.v_seq = []
            for t in range(x_seq.shape[0]):
                # print(self)
                spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
                self.v_seq.append(self.v.unsqueeze(0))

            spike_seq = torch.cat(spike_seq, 0)
            self.v_seq = torch.cat(self.v_seq, 0)

            return spike_seq.flatten(0, 1)

        else:
            raise NotImplementedError(self.backend)


# # for debugging
class LIFNode_Spikingjelly_Cupy(sj_LIFNode):

    def __init__(self,
                 step=4,
                 threshold=1.,
                 tau=2.,
                 detach_reset=True,
                 backend='cupy',
                 **kwargs):
        super().__init__(v_threshold=threshold,
                         tau=tau,
                         detach_reset=detach_reset,
                         v_reset=0.,
                         surrogate_function=surrogate.Sigmoid())
        self.register_memory('v_seq', None)
        self.step = step
        self.backend = backend

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        x_shape = x_seq.shape

        if self.backend == 'cupy':
            #方便debug 多封装一层
            x_seq = x_seq.reshape(self.step, -1,
                                  *x_shape[1:]).contiguous()  #适配braincog框架的输入

            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)

            spike_seq, self.v_seq = neuron_kernel.MultiStepLIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.decay_input,
                self.tau, self.v_threshold, self.v_reset, self.detach_reset,
                self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            self.v_seq = self.v_seq.reshape(x_seq.shape)

            self.v = self.v_seq[-1].clone()

            return spike_seq.flatten(0, 1)

        else:
            raise NotImplementedError(self.backend)


class PLIFNode(BaseNode_Torch):
    """
    Parametric LIF， 其中的 ```tau``` 会被backward过程影响
    Reference：https://arxiv.org/abs/2007.05785
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self,
                 threshold=1.,
                 tau=2.,
                 act_fun=AtanGrad,
                 *args,
                 **kwargs):
        super().__init__(threshold, *args, **kwargs)
        init_w = -math.log(tau - 1.)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=True)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    def integral(self, inputs):
        self.mem = self.mem + (
            (inputs - self.mem) * self.w.sigmoid()) * self.dt

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.get_thres())
        self.mem = self.mem * (1 - self.spike.detach())


class PSNTorch(nn.Module):
    """
    Parallel Spiking Neuron (PSN)
    https://arxiv.org/abs/2304.12760

    MUST set layer_by_layer = True
    CANNOT use braincog BaseNode
    """

    def __init__(self,
                 threshold=1,
                 step=10,
                 layer_by_layer=True,
                 mem_detach=True,
                 act_func=AtanGrad):
        super().__init__()
        assert layer_by_layer == True, "PSN MUST set layer_by_layer=True"
        self.fc = nn.Linear(step, step)
        nn.init.constant_(self.fc.bias, -1)

        # set alpha of Atan has no grad
        self.act_func = act_func(alpha=2., requires_grad=False)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.weight,
                            x_seq.flatten(1))  # [T,T] @ [T,*]
        spike = self.act_func(h_seq)
        return spike.view(x_seq.shape)
