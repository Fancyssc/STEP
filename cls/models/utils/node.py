# from braincog.base.connection.layer import *
# from braincog.base.strategy.surrogate import *
import torch
from braincog.model_zoo.base_module import BaseModule
from braincog.base.node.node import BaseNode, rearrange
from .surrogate import *

# for debugging
from spikingjelly.clock_driven.neuron import LIFNode as sj_LIFNode
from spikingjelly.clock_driven import surrogate



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
                 mem_detach=True,
                 *args,**kwargs):
        super().__init__(threshold=threshold,
                         step=step,
                         layer_by_layer=layer_by_layer,
                         mem_detach=mem_detach,
                         *args, **kwargs)

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

class IFNode(BaseNode_Torch):
    """
    Integrate and Fire Neuron(IFNode)
    Re-implemented with BaseNode_Torch

    :param threshold: The threshold that a neuron needs to reach in order to fire an action potential.
    """

    def __init__(self, threshold=.5, act_fun=AtanGrad, *args, **kwargs):
        """
        :param threshold: The threshold that a neuron needs to reach in order to fire an action potential.
        :param act_fun: Surrogate Activation Function
        """
        super().__init__(threshold, *args, **kwargs)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)

    def integral(self, inputs):
        self.mem = self.mem + inputs * self.dt

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.get_thres())
        self.mem = self.mem * (1 - self.spike.detach())

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
                         mem_detach=mem_detach,
                         *args,**kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun()

    def integral(self, inputs):
        self.mem = self.mem + (inputs - self.mem) / self.tau

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach())


class PLIFNode(BaseNode_Torch):
    """
    Parametric LIF(PLIF Node) Re-implmented with BaseNode_Torch
    """

    def __init__(self,
                 step=4,
                 layer_by_layer=True,
                 mem_detach=False,
                 threshold=1.,
                 tau=2.,
                 act_fun=Sigmoid_Grad,
                 *args,
                 **kwargs):
        super().__init__(threshold=threshold,
                         step=step,
                         layer_by_layer=layer_by_layer,
                         mem_detach=mem_detach,
                         *args,
                         **kwargs
                         )
        init_w = -math.log(tau - 1.)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun()
        self.w = nn.Parameter(torch.as_tensor(init_w))

    def integral(self, inputs):
        self.mem = self.mem + (
            (inputs - self.mem) * self.w.sigmoid()) * self.dt

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.get_thres())
        self.mem = self.mem * (1 - self.spike.detach())


class EIFNode(BaseNode_Torch):
    """
        a simplified EIFNode implemented with BrainCog
        :param v_reset: in EIFNode, the reset
    """
    def __init__(self,
                 threshold: float = 1.0,
                 step = 4,
                 tau = 2.,
                 layer_by_layer = True,
                 mem_detach=False,
                 delta_T = 1.,
                 theta_rh =.8,
                 v_reset = -0.1, # defualt v_reset = -0.1
                 act_fun=Sigmoid_Grad,
                 *args, **kwargs):
        super().__init__(threshold=threshold,
                         step=step,
                         layer_by_layer=layer_by_layer,
                         mem_detach=mem_detach,
                         *args,
                         **kwargs
                         )

        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun()

        self.v_reset = v_reset
        self.register_buffer('delta_T', torch.tensor(delta_T))
        self.register_buffer('theta_rh', torch.tensor(theta_rh))
        self.register_buffer('tau', torch.tensor(tau)) # register as tensor for torch.exp()

    def integral(self, inputs):
        # self.mem = self.mem + (inputs - self.mem) / self.tau
        self.mem = self.mem + (inputs - self.mem + self.delta_T * torch.exp(
            (self.mem - self.theta_rh) / self.delta_T)) / self.tau

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach()) + self.spike.detach() * self.v_reset # v_reset is not zero



heaviside_sigmoid = HeavisideSigmoid.apply
heaviside_parametric_sigmoid = HeavisideParametricSigmoid.apply

class HDLIFNode(BaseModule):
    """
        AKA MoE-IFNode of Spiking Point Transformer(SPT, AAAI 2025)
    """
    def __init__(self,
                 input_dims,  # compulsory in HDLIFNode
                 threshold = 0.5,
                 step = 4,
                 tau = 2.,
                 layer_by_layer = True,
                 mem_detach=False,
                 act_fun=Sigmoid_Grad,
                 *args, **kwargs):
        super().__init__(step=step,encode_type='direct',layer_by_layer=layer_by_layer,
                         *args,
                         **kwargs
                         )

        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun()
        self.step = step
        self.embed_dims = input_dims
        self.tau = tau
        self.threshold = threshold

        # EIFNode param (fixed)
        self.v_reset_EIF = -0.1
        self.delta_T_EIF = 1.0
        self.theta_rh_EIF= 0.8

        # adding neurons
        self.experts = nn.ModuleList([])
        self.experts.append(IFNode(threshold=threshold, step=step, layer_by_layer=layer_by_layer,
                                   act_fun=act_fun, mem_detach=mem_detach,requires_mem=True)) # IFNode
        self.experts.append(LIFNode(threshold=threshold,step=step,layer_by_layer=layer_by_layer,tau=tau,
                                    act_fun=act_fun,mem_detach=mem_detach,requires_mem=True)) # LIFNode
        self.experts.append(PLIFNode(threshold=threshold,step=step,layer_by_layer=layer_by_layer,tau=tau,
                                    act_fun=act_fun,mem_detach=mem_detach,requires_mem=True)) # PLIFNode
        self.experts.append(EIFNode(threshold=threshold,step=step,layer_by_layer=layer_by_layer,tau=self.tau,
                                    act_fun=act_fun,mem_detach=mem_detach,v_reset=self.v_reset_EIF,
                                    delta_T=self.delta_T_EIF,theta_rh=self.theta_rh_EIF,requires_mem=True)) # EIFNode

        self.k_MoE = 4
        self.v_th_MoE = 0.2
        self.gate_MoE = nn.Conv1d(self.embed_dims*self.step, len(self.experts), 1)

    def forward(self, x):

        H, W = None, None
        transpose_flag = False

        # TB C H W should be reshaped to TB C N for MoE op
        if len(x.shape) == 4: # TB C H W -> TB C N
            TB, C, H, W = x.shape
            x = x.flatten(-2, -1)  # TB C M

        # TB N C should be transposed to TB C N for MoE op
        ## We assume C != N
        if x.shape[1] != self.embed_dims:
            x = x.transpose(-2, -1)
            transpose_flag = True

        if self.training:
            z = x.view(-1, self.step, *x.shape[1:]).flatten(1, 2) # B TC N
            gate = F.softmax(self.gate_MoE(z), dim=-2).repeat(self.step, 1, 1, 1) # T B E N
            spikes = torch.stack([expert(x) for expert in self.experts], dim=1)  # TB E C N # fire
            expert_outputs = torch.stack([torch.stack(expert.mem_collect, dim=0).flatten(0, 1)
                                          for expert in self.experts], dim=1).view(
                                          self.step, -1, len(self.experts),*x.shape[1:])  # T, B, E, C, N
            expert_outputs[expert_outputs == 0.0] = self.threshold
            output = torch.sum(gate.unsqueeze(3) * expert_outputs, dim=2)
            output = heaviside_parametric_sigmoid(output, self.k_MoE, self.v_th_MoE)

        else:
            z = x.view(-1, self.step, *x.shape[1:]).flatten(1, 2)
            gate = self.gate_MoE(z)
            topk_values, topk_indices = torch.topk(gate, 2, dim=-2)

            gate = torch.zeros_like(gate)
            gate.scatter_(-2, topk_indices, topk_values)
            gate_masked = gate.clone()
            gate_masked[gate == 0] = float('-inf')
            gate = F.softmax(gate_masked, dim=-2).repeat(self.step, 1, 1, 1)

            expert_spikes = torch.stack([expert(x) for expert in self.experts], dim=1)
            expert_outputs = torch.stack([torch.stack(expert.mem_collect,dim=0).flatten(0, 1)
                                          for expert in self.experts], dim=1).view(
                                          self.step, -1, len(self.experts),*x.shape[1:])
            expert_outputs[expert_outputs == 0.0] = self.threshold
            output = torch.sum(gate.unsqueeze(3) * expert_outputs, dim=2)
            output = heaviside_parametric_sigmoid(output, self.k_MoE, self.v_th_MoE)

        if H is None and W is None:
            if not transpose_flag:
               return output.flatten(0, 1) # TB C N
            else:
                return output.flatten(0, 1).transpose(-2, -1) #TB N C
        else:
            return output.flatten(0, 1).reshape(TB, C, H, W)


class ILIFNode(BaseNode_Torch):
    """
    Integer LIFNode (ILIFNode)
    """

    def __init__(self,
                 threshold=1.0,
                 step=4,
                 layer_by_layer=True,
                 tau=2.,
                 act_fun=Sigmoid_Grad,
                 mem_detach=False,
                 D = 2,
                 *args,
                 **kwargs):
        super().__init__(threshold=threshold,
                         step=step,
                         layer_by_layer=layer_by_layer,
                         mem_detach=mem_detach,
                         *args,**kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun()
        self.D = D

    def integral(self, inputs):
        self.mem = self.mem + (inputs - self.mem) / self.tau

    def calc_spike(self):
        # compute normalized potential
        u = self.mem / self.threshold
        u_clamped = torch.clamp(u, min=0, max=self.D)
        spikes_hard = torch.round(u_clamped)
        spikes = spikes_hard.detach() - u_clamped.detach() + u_clamped # for back propagation
        self.spike = spikes
        self.mem = self.mem - spikes_hard * self.threshold

class NILIFNode(BaseNode_Torch):
    """
    Normlized Integer LIFNode (NILIFNode)
    """

    def __init__(self,
                 threshold=1.0,
                 step=4,
                 layer_by_layer=True,
                 tau=2.,
                 act_fun=Sigmoid_Grad,
                 mem_detach=False,
                 D = 2,
                 *args,
                 **kwargs):
        super().__init__(threshold=threshold,
                         step=step,
                         layer_by_layer=layer_by_layer,
                         mem_detach=mem_detach,
                         *args,**kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun()
        self.D = D

    def integral(self, inputs):
        self.mem = self.mem + (inputs - self.mem) / self.tau

    def calc_spike(self):
        # compute normalized potential
        u = self.mem / self.threshold
        u_clamped = torch.clamp(u, min=0, max=self.D) / self.D
        spikes_hard = torch.round(u_clamped)
        spikes = spikes_hard.detach() - u_clamped.detach() + u_clamped # for back propagation
        self.spike = spikes
        self.mem = self.mem - spikes_hard * self.threshold

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
                 step=4,
                 layer_by_layer=True,
                 mem_detach=True,
                 act_fun=Sigmoid_Grad,tau=2.0,**kwargs):
        super().__init__()
        assert layer_by_layer is True, "PSN MUST set layer_by_layer=True"
        self.fc = nn.Linear(step, step)
        self.T = step
        nn.init.constant_(self.fc.bias, -1)

        # set alpha of Atan has no grad
        self.act_func = act_fun(alpha=2., requires_grad=False)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T*N, *]
        x_seq_t = x_seq.reshape(self.T,-1)
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.weight,
                            x_seq_t)  # [T,T] @ [T,*]
        spike = self.act_func(h_seq)
        return spike.view(x_seq.shape)



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
                 param=[0.25, 0.5, -1, 0.5],**kwargs):

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
        x_t = x.reshape(self.T,-1)
        u = torch.zeros(x_t.shape[1:], device=x.device)
        out = torch.zeros(x_t.shape, device=x.device)
        for step in range(self.T):
            u, out[step] = self.extended_state_update(
                u,
                out[max(step - 1, 0)],
                x_t[step],
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
        return out.reshape(x.shape)

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
        # print(u_t1_n1.shape)
        # print(Vth)
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
        self.relu = nn.ReLU()

    def integral(self, inputs):
        self.mem = self.mem + (inputs - self.mem) / self.tau

        # klif
        self.mem *= self.k
        self.mem = self.relu(self.mem)

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach())



class ComplementaryLIFNeuron(nn.Module):
    """
    CLIF: Complementary Leaky Integrate-and-Fire Neuron for Spiking Neural Networks
    https://arxiv.org/abs/2402.04663 
    """

    def __init__(self,
                 step=4,
                 tau: float = 2.,
                 decay_input: bool = False,
                 v_threshold: float = 1.,
                 v_reset: float = None,
                 surrogate_function=Rectangle(),
                 layer_by_layer=True,
                 **kwargs):
        super().__init__()

        self.tau = tau
        self.step=step
        self.decay_input = decay_input
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        if v_reset:
            self.v = v_reset  # membrane potential
        else:
            self.v = 0.0
        self.surrogate_function = surrogate_function.spiking_function
        self.lbl = layer_by_layer

        self.m = 0  # Complementary memory
        # self.register_memory('m', 0.)

    def forward(self, inputs: torch.tensor):
        if self.lbl:
            return self.forward_lbl(x_seq=inputs)
        else:
            return self.forward_timestep(x=inputs)

    def forward_timestep(self, x: torch.Tensor):
        self.neuronal_charge(x)  # LIF charging
        self.m = self.m * torch.sigmoid(self.v / self.tau)  # Forming
        spike = self.neuronal_fire()  # LIF fire
        self.m += spike  # Strengthen
        self.neuronal_reset(spike)  # LIF reset
        self.v = self.v - spike * torch.sigmoid(self.m)  # Reset
        return spike

    def forward_lbl(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1

        #print(x_seq.shape)
        if len(x_seq.shape) == 3:
            x_seq_t = rearrange(x_seq,'(t b) n c -> t b n c',t=self.step)
        elif len(x_seq.shape) == 4:
            x_seq_t = rearrange(x_seq,'(t b) n h w -> t b n h w',t=self.step)
        # print(x_seq_t.shape)
        # x_seq.shape = [T, *]
        spike_seq = []
        self.v_seq = []
        for t in range(x_seq_t.shape[0]):
            spike_seq.append(self.forward_timestep(x_seq_t[t]).unsqueeze(0))
            self.v_seq.append(self.v.unsqueeze(0))
        spike_seq = torch.cat(spike_seq, 0).reshape(x_seq.shape)
        self.v_seq = torch.cat(self.v_seq, 0)
        return spike_seq

    def neuronal_charge(self, x: torch.Tensor):
        self._charging_v(x)

    def neuronal_reset(self, spike: torch.Tensor):
        self._reset(spike)

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold,alpha=1.0)

    def _charging_v(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0:
            if type(self.v) is float:
                self.v = x
            else:
                self.v = self.v * (1 - 1. / self.tau) + x
        else:
            if type(self.v) is float:
                self.v = self.v_reset * (
                    1 - 1. / self.tau) + self.v_reset / self.tau + x
            else:
                self.v = self.v * (1 -
                                   1. / self.tau) + self.v_reset / self.tau + x

    def _reset(self, spike):
        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold
        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def n_reset(self):
        if self.v_reset:
            self.v = self.v_reset  # membrane potential
        else:
            self.v = 0.0

        self.m = 0  # Complementary memory




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

'''
   move to surrogate.py
'''

# class Sigmoid_Grad(SurrogateFunctionBase):
#     """
#     Sigmoid activation function with gradient
#     Overwrite sigmoid function in BrainCog
#     """
#
#     def __init__(self, alpha=4., requires_grad=False):
#         super().__init__(alpha=alpha, requires_grad=requires_grad)
#
#     @staticmethod
#     def act_fun(x, alpha):
#         return sigmoid.apply(x, alpha)
#
#
# class QGate_Grad(SurrogateFunctionBase):
#
#     def __init__(self, alpha=2., requires_grad=False):
#         super().__init__(alpha=alpha, requires_grad=requires_grad)
#
#     @staticmethod
#     def act_fun(x, alpha):
#         return quadratic_gate.apply(x, alpha)