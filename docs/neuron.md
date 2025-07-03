# Introduction
We have integrated almost all commonly used neuron types into the framework, though you can also customize your own neurons as needed. 

On this basis, we recommend developing new neuron types by inheriting from the neuron classes provided in BrainCog.

## Neuron Model Construction
Reference to the LIFNode, you can directly inherit BaseNode from BrainCog to develop new neurons. However, I highly recommend inheriting the rewritten BaseNode_PyTorch from `model/utils/node.py` in the framework. We have added operations in this neuron that are better adapted for Spiking Transformer.

Taking LIFNode as an example:
```angular2html
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
```

For the newly defined neuron, you must supply its intrinsic parameters—`Threshold` and `tau`— while also implementing membrane-potential accumulation and spike-release computation. 

Consequently, you need to rewrite the function `self.integral()` & `self.calc_spike()` as specified.

## Neuron Reset
Neurons in an SNN must reset between two unrelated successive inputs(basically happens between batches); the detailed procedure is as follows:

### Node Level 
A single neuron could be reset by:
```angular2html
lifnode = LIFNode(threshold=1.0, step=4, tau=2.0)
lifnode.reset()
```
This method is relatively cumbersome, so we recommend the following approach to perform the reset.

### Moudle Level
In the previous section we advised using a BaseModule. When neurons are incorporated inside a BaseModule, you can place the reset logic in the BaseModule’s forward function; this will reset the membrane potentials of all nodes.
```angular2html
class MyModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lifnode = LIFNode(threshold=1.0, step=4, tau=2.0)
        self.ifnode = IFNode(threshold=1.0, step=4,)
    def forward(self, inputs):
        self.reset()  # Reset all neurons in the module
        return self.lifnode(inputs)
```