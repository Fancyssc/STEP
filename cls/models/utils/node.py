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

    CANNOT use braincog BaseNode. Because NOT feed timestep sequentially to neuron
    Using nn.Module instead
    """

    def __init__(self,
                 threshold=1,
                 step=10,
                 layer_by_layer=True,
                 mem_detach=True,
                 act_func=AtanGrad):
        super().__init__()
        assert layer_by_layer is True, "PSN MUST set layer_by_layer=True"
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
        input = ctx.saved_tensors
        grad_input = grad_output.clone()

        # hu is an approximate func of df/du in linear formulation
        hu = abs(input) < 0.5
        hu = hu.float()

        # arctan surrogate function
        # hu =  1 / ((input * torch.pi) ** 2 + 1)

        # triangles
        # hu = (1 / gamma_SG) * (1 / gamma_SG) * ((gamma_SG - input.abs()).clamp(min=0))

        return grad_input * hu


class GLIFTorch(nn.Module):
    '''
        GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks
        https://arxiv.org/abs/2210.13768

        This may convert to BrainCog BaseNode version ?
    '''

    def __init__(self,
                 step=10,
                 soft_mode=False,
                 static_gate=True,
                 static_param=False,
                 time_wise=True,
                 layer_by_layer=True,
                 gate=[0.6, 0.8, 0.6],
                 param=[0.25, 0.5, -1, 0.5]):

        # gate: [alpha, beta, gamma]
        # param: [tau, Vth, linear_decay, conduct]

        super().__init__()
        self.T = step
        self.soft_mode = soft_mode
        self.static_gate = static_gate
        self.static_param = static_param
        self.time_wise = time_wise
        self.gate = gate
        self.param = param

        assert layer_by_layer is True, "must set layer by layer=True for GLIFTorch"

        # set linear_decay for param
        linear_decay = param[1] / (
            self.T * 2)  #linear decay coefficient, set to Vth/(T*2)
        self.param[2] = linear_decay

        self.alpha, self.beta, self.gamma = [
            nn.Parameter(
                torch.tensor(math.log(1 / ((i - 0.5) * 0.5 + 0.5) - 1),
                             dtype=torch.float)) for i in self.gate
        ]
        if self.static_param:
            self.tau, self.Vth, self.leak, self.conduct = [
                torch.tensor(-math.log(1 / i - 1), dtype=torch.float)
                for i in self.param
            ]
            self.reVth = self.Vth
        else:
            if self.time_wise:
                self.tau, self.Vth, self.leak, self.conduct = [
                    nn.Parameter(-math.log(1 / i - 1) *
                                 torch.ones(self.T, dtype=torch.float))
                    for i in self.param
                ]
                self.reVth = nn.Parameter(
                    -math.log(1 / self.param[1] - 1) *
                    torch.ones(self.T, dtype=torch.float))

            else:
                self.tau, self.Vth, self.leak = [
                    nn.Parameter(
                        torch.tensor(-math.log(1 / i - 1), dtype=torch.float))
                    for i in self.param[:-1]
                ]
                self.reVth = nn.Parameter(
                    torch.tensor(-math.log(1 / self.param[1] - 1),
                                 dtype=torch.float))

                self.conduct = [
                    nn.Parameter(-math.log(1 / i - 1) *
                                 torch.ones(self.T, dtype=torch.float))
                    for i in self.param[3:]
                ][0]

    def forward(self, x):
        u = torch.zeros(x.shape[1:], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(self.T):
            u, out[step] = self.extended_state_update(
                u,
                out[max(step - 1, 0)],
                x[step],
                tau=self.tau[step].sigmoid()
                if self.time_wise else self.tau.sigmoid(),
                Vth=self.Vth[step].sigmoid()
                if self.time_wise else self.Vth.sigmoid(),
                leak=self.leak[step].sigmoid()
                if self.time_wise else self.leak.sigmoid(),
                conduct=self.conduct[step].sigmoid()
                if not self.static_param else self.conduct.sigmoid(),
                reVth=self.reVth[step].sigmoid()
                if self.time_wise else self.reVth.sigmoid())
        return out

    #
    # def state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau):
    #     u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t_n1
    #     o_t1_n1 = spikeAct(u_t1_n1)
    #     return u_t1_n1, o_t1_n1

    def extended_state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau, Vth,
                              leak, conduct, reVth):
        if self.static_gate:
            al, be, ga = self.alpha.clone().detach().gt(
                0.).float(), self.beta.clone().detach().gt(
                    0.).float(), self.gamma.clone().detach().gt(0.).float()
        else:
            al, be, ga = self.alpha.sigmoid(), self.beta.sigmoid(
            ), self.gamma.sigmoid()
        # I_t1 = W_mul_o_t_n1 + be * I_t0 * self.conduct.sigmoid()#原先
        I_t1 = W_mul_o_t_n1 * (1 - be * (1 - conduct))
        u_t1_n1 = ((1 - al * (1 - tau)) * u_t_n1 * (1 - ga * o_t_n1.clone()) -
                   (1 - al) * leak) + I_t1 - (1 - ga) * reVth * o_t_n1.clone()
        o_t1_n1 = SpikeAct_extended.apply(u_t1_n1 - Vth)
        return u_t1_n1, o_t1_n1

    def _initialize_params(self, **kwargs):
        self.mid_gate_mode = True
        self.tau.copy_(
            torch.tensor(-math.log(1 / kwargs['param'][0] - 1),
                         dtype=torch.float,
                         device=self.tau.device))
        self.Vth.copy_(
            torch.tensor(-math.log(1 / kwargs['param'][1] - 1),
                         dtype=torch.float,
                         device=self.Vth.device))
        self.reVth.copy_(
            torch.tensor(-math.log(1 / kwargs['param'][1] - 1),
                         dtype=torch.float,
                         device=self.reVth.device))

        self.leak.copy_(
            -math.log(1 / kwargs['param'][2] - 1) *
            torch.ones(self.T, dtype=torch.float, device=self.leak.device))
        self.conduct.copy_(
            -math.log(1 / kwargs['param'][3] - 1) *
            torch.ones(self.T, dtype=torch.float, device=self.conduct.device))


class KLIFNode(BaseNode_Torch):
    """
    KLIF: An optimized spiking neuron unit for tuning surrogate gradient slope and membrane potential
    https://arxiv.org/abs/2302.09238
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
        self.k = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def integral(self, inputs):
        self.mem = self.mem + (inputs - self.mem) / self.tau

        # klif
        self.mem *= self.k
        self.mem = nn.ReLU(self.mem)

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach())
