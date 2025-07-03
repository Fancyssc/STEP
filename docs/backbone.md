# introduction
This framework launches model training via config files. In earlier versions, hyperparameters could be configured via terminal input, but we found this approach inconvenient for experiment management. Therefore, we recommend configuring and launching training using the same method as in the demo (i.e., through config files).

In this section, we use **classification (cls) as the primary example**. However, the backbone can be seamlessly adapted to segmentation (seg) and detection (det) tasks to accomplish similar workflows.

## Model Construction
By following `cls/models/spikformer_cifar.py`, you can learn how to build a model from scratch.

**Special Notes:**

**1. Transformer Design:**  
   - Inherit from `BaseModule` (BrainCog) instead of PyTorch's `nn.Module`  
   - *Rationale:* Better integration with neuromorphic computing features

**2. Layer-by-Layer Mode:**  
   - Set `layer_by_layer = True` in `BaseModule`  
   - *Advantage:* Enables gradual SNN parameter tuning

**3. Model Registration:**  
```angular2html
   # Register with timm library
   @register_model
   class YourModel(BaseModule):
       def __init__(self, ...):
           super().__init__(...)
           # Model initialization code
```

**4. Model Configuration**
Write and save your configuration file at:
`cls/configs/[your_model]/[dataset].yml`

In the config file, you can specify training hyperparameters, model size, neuron type, firing threshold, time constants, etc. However, you must ensure your custom model can properly receive these parameters. For highly specialized parameters, script modifications may be required.


<details>
<summary> Config File Example (Spikformer/cifar10.yml) </summary>
```angular2html
# dataset
data_dir: '/data/datasets/CIFAR10'
dataset: torch/cifar10
num_classes: 10
img_size: 32

#data augmentation
mean:
    - 0.4914
    - 0.4822
    - 0.4465
std:
    - 0.2470
    - 0.2435
    - 0.2616
crop_pct: 1.0
scale:
    - 1.0
    - 1.0
ratio: [1.0,1.0]
color_jitter: 0.
interpolation: bicubic
train_interpolation: bicubic
aa: rand-m9-n1-mstd0.4-inc1
epochs: 400   #epochs
mixup: 0.5
mixup_off_epoch: 200
mixup_prob: 1.0
mixup_mode: batch
mixup_switch_prob: 0.5
cutmix: 0.0
reprob: 0.25
remode: const

# model structure
model: "spikformer_cifar"
step: 4
patch_size: 4
in_channels: 3
embed_dim: 384
num_heads: 12
mlp_ratio: 4
attn_scale: 0.125
mlp_drop: 0.0
attn_drop: 0.0
depths: 4

#meta transformer layer
embed_layer: 'SPS'
attn_layer: 'SSA'


# node
tau: 2.0
threshold: 1.0
act_function: SigmoidGrad
node_type: LIFNode
alpha: 4.0

# train hyperparam
amp: True
batch_size: 128
val_batch_size: 128
lr: 5e-4
min_lr: 1e-5
sched: cosine
weight_decay: 6e-2
cooldown_epochs: 10
warmup_epochs: 20
warmup_lr: 0.00001
opt: adamw
smoothing: 0.1
workers: 16
seed: 42
log_interval: 200

# log dir
output: "/home/shensicheng/log/SpikingTransformerBenchmark/cls/Spikformer"

# device
device: 2
```
</details>