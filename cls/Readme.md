## Classification

### Implemented & To-implement Models
The listed results display the performance of models on different datasets. The results are based on the original papers and the corresponding datasets. The results are updated regularly. If you have any questions, please feel free to contact us.

To emphasize, if no special note, the default size of model is ```4-384``` for CIFAR10/100 and ```8-768``` for ImageNet-1K. The default step is ```4``` for both datasets.
The default size of model is ```2-512``` for CIFAR10-DVS and N-Cal101. The default step is ```10``` for both datasets.

|                Model                |                                                                                  Pub. Info.                                                                                  |             CIFAR10/100             |     ImageNet-1K      |         CIFAR10-DVS         | N-Cal101 |
|:-----------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------:|:--------------------:|:---------------------------:|:--------:|
|             Spikformer              |                                                                [ICLR 2023](https://arxiv.org/abs/2209.15425)                                                                 |            95.41/ 78.21             |        74.81         |            78.9             |    -     |
|             Spikformer v2              |                                                                [Arxiv](https://arxiv.org/abs/2401.02020)                                                                 |            - / -             |        80.38 ```(8-512)```         |            -             |    -     |
|              QKFormer               |                                                               [NeurIPS 2024](https://arxiv.org/abs/2403.16552)                                                               |            96.18/ 81.15             | 85.65 ```(10-786)``` |      84.0```(T=16)```       |    -     |
|            Spikingformer            |                                                                  [Arxiv](https://arxiv.org/abs/2304.11954)                                                                   |            95.81/ 79.21             |        75.85         |            79.9             |    -     |
|              SGLFormer              |                           [Frontiers in Neuroscience](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1371290/full)                            |            96.76/ 82.26             |        83.73         |            82.9             |    -     |         
|     Spiking Wavelet Transforemr     |                                                  [ECCV 2024](https://link.springer.com/chapter/10.1007/978-3-031-73116-7_2)                                                  |             96.1/ 79.3              | 75.34 ```(8-512)```  |            82.9             |  88.45   |
|  Spike-driven Transformer(SDT v1)   |                     [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ca0f5358dbadda74b3049711887e9ead-Abstract-Conference.html)                     |       95.6/ 78.4```(2-512)```       |        77.07         |      80.0 ```(T=16)```      |    -     |
|       Meta-SpikeFormer(SDT v2)      |                                                                [ICLR 2024](https://arxiv.org/abs/2404.03663)                                                                 |             - / -                   |        80.00        |      -      |    -     |
|        E-SpikeFormer(SDT v3)        |                                                                [TPAMI](https://arxiv.org/abs/2411.16061)                                                                     |             - / -                    |    86.20 ```(T=8)``` |      -      |    -     |
|                  MST                |                                                                [ICCV 2023](https://arxiv.org/abs/2210.01208)                                                                     |  97.27 / 86.91  ```(ANN-to-SNN)```  |  78.51 ```(ANN-to-SNN)``` |      88.12 ```(ANN-to-SNN)```      |    91.38 ```(ANN-to-SNN)```     |
|                 QSD                   |                                                                [ICLR 2025](https://arxiv.org/abs/2501.13492)                                                                 |  98.4 / 87.6    ```(Transfer Learning)``` |        80.3          |      89.8 ```(Transfer Learning)```      |    -     |
|         Spiking Transformer         |                                                                [CVPR 2025](https://arxiv.org/abs/2503.00226)                                                                 |           96.32 / 79.69             |        78.66 ```(10-512)```         |      -      |    -     |
|               SNN-ViT               |                                                                [ICLR 2025](https://openreview.net/forum?id=qzZsz6MuEq)                                                                 |           96.1 / 80.1             |        80.23         |      82.3      |    -     |
|               STSSA               |                                                                [ICASSP 2025](https://ieeexplore.ieee.org/document/10890026)                                                                 |           - / -             |        -         |      83.8      |    81.65     |
|          Spikformer + SEMM          |                                                          [NeurIPS 2024](https://openreview.net/forum?id=WcIeEtY3AG)                                                          |            95.78/ 79.04             | 75.93 ```(8-512)```  |            82.32            |    -     |
|          SpikingResformer           | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Shi_SpikingResformer_Bridging_ResNet_and_Vision_Transformer_in_Spiking_Neural_Networks_CVPR_2024_paper.html) | 97.40/ 85.98 ```(Transfer Learning)``` |        79.40         | 84.8 ```(Transfer Learning)``` |    -     |    
|                 TIM                 |                                                        [IJCAI 2024](https://www.ijcai.org/proceedings/2024/0347.pdf)                                                         |                  -                  |          -           |            81.6             |  79.00   |
More models are to be implemented soon...


### Experiment Results
The default neuron node used in spiking transformers are `LIFNode(tau=2.,thres=1.0,Sigmoid_Grad(alpha=4.))` and the models are in the mode of `layer by layer`. If any 
special conditions are considered, it will be noted in the supp. of the table.

Other hyper-param setting are following the original paper.
#### CIFAR 10
|       Model       | Batch-Size | Dataset | Step | Epoch | Result(Acc@1) |                        supp.                         |
|:-----------------:|:----------:|:-------:|:-------:|:-----:|:-------------:|:----------------------------------------------------:|
|    Spikformer     |    128     | CIFAR10 | 4 |  400  |     95.12     |                          -                           |
|                   |            |         ||       |
|        SDT        |    128     | CIFAR10 | 4 |  400  |     95.77     |                          -                           |
|                   |            |         ||       |
|     QKFormer      |    128     | CIFAR10 | 4 |  400  |     96.24     |                          -                           |
|                   |            |         ||       |
|   Spikingformer   |    128     | CIFAR10 | 4 |  400  |     95.53     |                          -                           |
|                   |            |         ||       |
| Spikformer + SEMM |    128     | CIFAR10 | 4 |  400  |     94.98     |                          -                           |
|                   |            |         ||       |
|  Spiking Wavelet  |    128     | CIFAR10 | 4 |  400  |     95.31     |                          -                           |
|                   |            |         ||       |
|     SGLFormer     |     16     | CIFAR10 | 4 |  400  |     95.88     |                          -                           |
|                   |            |         ||       |
| Spikingresformer  |    128     | CIFAR10 | 4 |  400  |     95.69     |          Transfer Learning Used Originally           |
|                   |            |         ||       |

#### CIFAR 100
|       Model       | Batch-Size | Dataset  | Step | Epoch | Result(Acc@1) |               supp.               |
|:-----------------:|:----------:|:--------:|:-------:|:-----:|:-------------:|:---------------------------------:|
|    Spikformer     |    128     | CIFAR100 | 4 |  400  |     77.37     |                 -                 |
|                   |            |          ||       |
|        SDT        |    128     | CIFAR10  | 4 |  400  |     78.29     |                 -                 |
|                   |            |          ||       |
|     QKFormer      |    128     | CIFAR100 | 4 |  400  |     79.72     |                 -                 |
|                   |            |          ||       |
|   Spikingformer   |    128     | CIFAR100 | 4 |  400  |     79.12     |                 -                 |
|                   |            |          ||       |
| Spikformer + SEMM |    128     | CIFAR100 | 4 |  400  |     77.59     |                 -                 |
|                   |            |          ||       |
|  Spiking Wavelet  |    128     | CIFAR100 | 4 |  400  |     76.99     |                 -                 |
|                   |            |          ||       |
|     SGLFormer     |     16     | CIFAR100 | 4 |  400  |     80.61     |                 -                 |
|                   |            |          ||       |
| Spikingresformer  |    128     | CIFAR100 | 4 |  400  |     79.45     | Transfer Learning Used Originally |
|                   |            |          ||       |

#### Sequential Image Classification

|   Model    | Batch-Size | Dataset | Step | Epoch | Result(Acc@1) |               supp.               |
|:----------:|:----------:|:-------:|:-------:|:-----:|:-------------:|:---------------------------------:|
| Spikformer |    128     | sMNIST  | 4 |  400  |     98.84     |                 -                 |
| Spikformer |    128     | psMNIST | 4 |  400  |     97.97     |                 -                 |
| Spikformer |    128     | sCIFAR10| 4 |  400  |     84.26     |                 -                 |
|            |            |          ||       |
|    SDT     |    128     | sMNIST  | 4 |  400  |     98.77     |                 -                 |
|    SDT     |    128     | psMNIST | 4 |  400  |     97.80     |                 -                 |
|    SDT     |    128     | sCIFAR10| 4 |  400  |     82.31     |                 -                 |
|            |            |          ||       |
| Spikformer + SEMM |    128     | sMNIST  | 4 |  400  |     99.33     |                 -                 |
| Spikformer + SEMM |    128     | psMNIST | 4 |  400  |     98.46     |                 -                 |
| Spikformer + SEMM |    128     | sCIFAR10| 4 |  400  |     85.61     |                 -                 |

#### Neuron Test
|   Model    | Batch-Size | Dataset  | Step | Epoch | Result(Acc@1) | Neuron |
|:----------:|:----------:|:--------:|:----:|:-----:|:-------------:|:------:|
| Spikformer |    128     | CIFAR10 |  4   |  400  |     95.38     |  CLIF  |
| Spikformer |    128     | CIFAR10 |  4   |  400  |     95.41     |  GLIF  |
| Spikformer |    128     | CIFAR10 |  4   |  400  |     95.85     |  KLIF  |
| Spikformer |    128     | CIFAR10 |  4   |  400  |     96.06     |  PLIF  |
|            |            ||      |       |               |        |
|    SDT     |    128     | CIFAR10 |  4   |  400  |     95.49     |  CLIF  |
|    SDT     |    128     | CIFAR10 |  4   |  400  |     95.45     |  GLIF  |
|    SDT     |    128     | CIFAR10 |  4   |  400  |     95.63     |  KLIF  |
|    SDT     |    128     | CIFAR10 |  4   |  400  |     95.91     |  PLIF  |
|            |            ||      |       |               |        |
|      Spikformer + SEMM      |    128     | CIFAR10 |  4   |  400  |     95.44     |  CLIF  |
|   Spikformer + SEMM     |    128     | CIFAR10 |  4   |  400  |     95.78     |  GLIF  |
|    Spikformer + SEMM     |    128     | CIFAR10 |  4   |  400  |     95.59     |  KLIF  |
|  Spikformer + SEMM    |    128     | CIFAR10 |  4   |  400  |     95.66     |  PLIF  |


#### Sequential Image Classification

|   Model    | Batch-Size | Dataset | Step | Epoch | Result(Acc@1) |               supp.               |
|:----------:|:----------:|:-------:|:-------:|:-----:|:-------------:|:---------------------------------:|
| Spikformer |    128     | sMNIST  | 4 |  400  |     98.84     |                 -                 |
| Spikformer |    128     | psMNIST | 4 |  400  |     97.97     |                 -                 |
| Spikformer |    128     | sCIFAR10| 4 |  400  |     84.26     |                 -                 |
|            |            |          ||       |
|    SDT     |    128     | sMNIST  | 4 |  400  |     98.77     |                 -                 |
|    SDT     |    128     | psMNIST | 4 |  400  |     97.80     |                 -                 |
|    SDT     |    128     | sCIFAR10| 4 |  400  |     82.31     |                 -                 |
|            |            |          ||       |
| Spikformer + SEMM |    128     | sMNIST  | 4 |  400  |     99.33     |                 -                 |
| Spikformer + SEMM |    128     | psMNIST | 4 |  400  |     98.46     |                 -                 |
| Spikformer + SEMM |    128     | sCIFAR10| 4 |  400  |     85.61     |                 -                 |

#### Encoding Test
|   Model    | Batch-Size | Dataset  | Step | Epoch | Result(Acc@1) | Encoding |
|:----------:|:----------:|:--------:|:----:|:-----:|:-------------:|:--------:|
| Spikformer |    128     | CIFAR10 |  4   |  400  |     82.75     |  phase   |
| Spikformer |    128     | CIFAR10 |  4   |  400  |     82.83     |   rate   |
| Spikformer |    128     | CIFAR10 |  4   |  400  |     82.10     |   ttfs   |
|            |            ||      |       |               |          |
|    SDT     |    128     | CIFAR10 |  4   |  400  |     85.37     |  phase   |
|    SDT     |    128     | CIFAR10 |  4   |  400  |     83.77     |   rate   |
|    SDT     |    128     | CIFAR10 |  4   |  400  |     84.30     |   ttfs   |
|            |            ||      |       |               |          |
|      Spikformer + SEMM      |    128     | CIFAR10 |  4   |  400  |     85.81     |  phase   |
|   Spikformer + SEMM     |    128     | CIFAR10 |  4   |  400  |     83.04     |   rate   |
|    Spikformer + SEMM     |    128     | CIFAR10 |  4   |  400  |     83.37     |   ttfs   |

#### Meta-Transformer Test
**ssa/sdsa**: Spiking Attention from originial Spikformer/SDT

**sdsa_3**: Spiking Attention from Meta-Spikeformer(SDT_2) which substitute linear and regular conv with `RepConv`.

 **random_attn**: Query and Key are randomly initialized and not updated. 

 **sps_1conv**: Using vanilla ViT embedding method and learnable position encoding.

 **sps_2conv**: Reduce the num. of conv layers from 4 to 2 in SPS of Spiking Transformer.

|            Model            | Batch-Size |       Node        | Step | Seed | Result(Acc@1) |          supp.          |
|:---------------------------:|:----------:|:-----------------:|:-------:|:----:|:-------------:|:-----------------------:|
|         Spikformer          |    128     |     Braincog      | 4 |  42  |     95.12     |           ssa           |
|         Spikformer          |    128     |     Braincog      | 4 |  42  |     94.96     |       random_attn       |
|         Spikformer          |    128     |     Braincog      | 4 |  42  |     78.21     |        sps_1conv        |
|         Spikformer          |    128     |     Braincog      | 4 |  42  |     91.92     |        sps_2conv        |
|         Spikformer          |    128     |     Braincog      | 4 |  42  |     95.57     |         sdsa_3          |
|         Spikformer          |    128     |     Braincog      | 4 |  42  |     89.97     |   sdsa_3 & vit_embed    |
|         Spikformer          |    128     |     Braincog      | 4 |  42  |     93.43     |  sdsa_3 & conv2_embed   |
|                             |            |                   ||      |
|             SDT             |     64     |     Braincog      | 4 |  42  |     95.66     |          sdsa           |
|             SDT             |     64     |     Braincog      | 4 |  42  |     96.48     |       random_attn       |
|             SDT             |     64     |     Braincog      | 4 |  42  |     82.17     |        sps_1conv        |
|             SDT             |     64     |     Braincog      | 4 |  42  |     93.03     |        sps_2conv        |
|             SDT             |     64     |     Braincog      | 4 |  42  |     96.48     |         sdsa_3          |
|             SDT             |     64     |     Braincog      | 4 |  42  |     87.86     |   sdsa_3 & vit_embed    |
|             SDT             |     64     |     Braincog      | 4 |  42  |     94.56     |  sdsa_3 & conv2_embed   |
|                             |            |                   ||      |
|      Spikformer + SEMM      |    128     |     Braincog      | 4 |  42  |     94.98     |           ssa           |
|      Spikformer + SEMM      |    128     |     Braincog      | 4 |  42  |     95.57     |       random_attn       |
|      Spikformer + SEMM      |    128     |     Braincog      | 4 |  42  |     89.24     |        sps_1conv        |
|      Spikformer + SEMM      |    128     |     Braincog      | 4 |  42  |     93.33     |        sps_2conv        |
|      Spikformer + SEMM      |    128     |     Braincog      | 4 |  42  |     95.83     |         sdsa_3          |
|      Spikformer + SEMM      |    128     |     Braincog      | 4 |  42  |     84.95    |   sdsa_3 & vit_embed    |
|      Spikformer + SEMM      |    128     |     Braincog      | 4 |  42  |     93.37     |  sdsa_3 & conv2_embed   |
|                             |            |                   ||      |
|             ViT             |    128     |     Braincog      | 4 |  42  |     96.51     |        attention        |
|             ViT             |    128     |     Braincog      | 4 |  42  |     90.89     |        sps_1conv        |
|             ViT             |    128     |     Braincog      | 4 |  42  |     96.41     |       random_attn       |
|             ViT             |    128     |     Braincog      | 4 |  42  |     95.0      |        sps_2conv        |
|             ViT             |    128     |     Braincog      | 4 |  42  |     88.46     | sps_1conv & random_attn |

#### ImageNet-1K
to be updated

### Guidelines
#### Code Environment
1. **The path configuration for your code repository should be as follows.**
```angular2html
.
├── Readme.md
└── cls
    ├── configs
    │   └── spikformer
    │       ├── cifar10.yml
    │       ├── cifar100.yml
    │       ├── imgnet.yml
    │   └── ...
    ├── data
    │   ├── aa_snn.py
    │   ├── loader.py
    │   └── transforms_factory.py
    ├── models
    │   └── static
    │       ├── spikformer_cifar.py
    │       └── spikformer_img.py
    │       └── ...
    ├── requirements.txt 
    ├── train.py
    └── utils
        ├── __init__.py
        └── node.py
```

2. **Install the required packages.**
```angular2html
    conda create -n [your_env_name] python=3.8 -y
    conda activate [your_env_name]
    pip install -r requirements.txt
```

3. **Configure your model**

    To configure your model and place it under ```models/static``` or ```model/dvs``` directory, along with registering the model using timm‘s register function.
    
    To write config files and place them under ```configs/[your_model]``` directory, the format should **strictly** flollow the format of the existing config files. We highly recommend this method which keeps structures of all ```.yml``` files clear and consistent. The reason for separating model configuration and training hyperparameters is to facilitate debugging and make the tuning process easier.
    
    Eventually, to import the registered model in ```train.py```.
```

# dataset
data_dir: '/data/datasets/CIFAR10'
dataset: torch/cifar10
num_classes: 10
img_size: 32

# data augmentation
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

# transformer layer
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
output: "./output"'"

# device
device: 0

```


4. **Run the training script**

Since dynamic and static datasets typically use different loading methods and data augmentation techniques, most models employ two separate scripts with corresponding augmentation strategies. Therefore, we also divide the scripts into two here.

**For Static Datasets:**
```angular2html
    python train.py --config configs/[your_model]/[your_dataset].yml
```



**For DVS Datasets:**
```angular2html
    python train_dvs.py --config configs/[your_model]/[your_dataset].yml 
```

### Supported Datasets
|                                                 Dataset                                                 |  Type  |    Mission     | |                                                    Dataset                                                    |  Type  |    Mission     |
|:-------------------------------------------------------------------------------------------------------:|:------:|:--------------:|:-:|:-------------------------------------------------------------------------------------------------------------:|:------:|:--------------:|
|                         [CIFAR10 ](https://www.cs.toronto.edu/~kriz/cifar.html)                         | Static |      cls       | |                            [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)                            | Static |      cls       |
|                                 [ImageNet](https://www.image-net.org/)                                  | Static |      cls       | |                                  [MNIST](http://yann.lecun.com/exdb/mnist/)                                   | Static |      cls       |
|                                  [sMNIST](https://arxiv.org/abs/1504.00941)                                   | Static | sequential_cls | |                                                  [psMNIST](https://arxiv.org/abs/1504.00941)                                                  | Static | sequential_cls |
|                                  [sCIFAR](https://arxiv.org/abs/1710.02224)                                   | Static | sequential_cls | |                                                       -                                                       |   -    |       -        |
|                                                                                                         |        |                | |                                                                                                               |        |                |
| [CIFAR10-DVS](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00309/full) |  DVS   |      cls       | | [DVS-Gesture](https://research.ibm.com/publications/a-low-power-fully-event-based-gesture-recognition-system) |  DVS   |      cls       | 
|                      [NCARS](https://www.prophesee.ai/2018/03/13/dataset-n-cars/)                       |  DVS   |      cls       | |                     [N-CALTECH101](https://www.garrickorchard.com/datasets/n-caltech101)                      |  DVS   |      cls       |
|                            [HMDB51-DVS](https://arxiv.org/pdf/1910.03579v2)                             |  DVS   |      cls       | |                               [UCF101-DVS](https://arxiv.org/pdf/1910.03579v2)                                |  DVS   |      cls       |

## Visualization
We are currently developing content in the **Benchmark** that is used to **visualize the performance of the Transformer**. You can access it by executing the following commands:

```angular2html
cd SpikingTransformerBenchmark 
conda activate [env]
python -m cls.vis.gradcam_vis
```
Please **make sure to follow the instructions exactly as shown above**, otherwise, errors may occur due to **path issues**.
