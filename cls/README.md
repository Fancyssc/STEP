## Classification
Classification is one of the most fundamental tasks in computer vision. Our framework supports classification tasks on **static datasets, neuromorphic (DVS) datasets,** and **3D point cloud datasets**.

Supported datasets can be refered to **[here](#supported-datasets)**

Checkpoints can be downloaded from **[official repo on huggingface](https://huggingface.co/Fancysean/STEP)**
### Model-zoo
These results reproduce the original-paper scores and are updated periodically.
- Default configs: ```4-384 (4 steps)``` for CIFAR10/100, ```8-768 (4 steps)``` for ImageNet-1K. 
- For CIFAR10-DVS and N-Caltech101 we use ```2-512 (10 steps)```.
- ✅: implemented yet;   ❌: not implemented yet or under considering;

☕️**More Models will be integrated soon...** 


#### Static & DVS Models

|                Model                |                                                                                  Pub. Info.                                                                                  |             CIFAR10/100             |     ImageNet-1K      |         CIFAR10-DVS         | N-Cal101 | status |
|:-----------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------:|:--------------------:|:---------------------------:|:--------:|:------:|
|             Spikformer              |                                                                [ICLR 2023](https://arxiv.org/abs/2209.15425)                                                                 |            95.41/ 78.21             |        74.81         |            78.9             |    -     |   ✅    |
|             Spikformer v2              |                                                                [Arxiv](https://arxiv.org/abs/2401.02020)                                                                 |            - / -             |        80.38 ```(8-512)```         |            -             |    -     |   ❌    |
|              QKFormer               |                                                               [NeurIPS 2024](https://arxiv.org/abs/2403.16552)                                                               |            96.18/ 81.15             | 85.65 ```(10-786)``` |      84.0```(T=16)```       |    -     |   ✅    |
|            Spikingformer            |                                                                  [Arxiv](https://arxiv.org/abs/2304.11954)                                                                   |            95.81/ 79.21             |        75.85         |            79.9             |    -     |   ✅    |
|              SGLFormer              |                           [Frontiers in Neuroscience](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1371290/full)                            |            96.76/ 82.26             |        83.73         |            82.9             |    -     |   ✅    |
|     Spiking Wavelet Transforemr     |                                                  [ECCV 2024](https://link.springer.com/chapter/10.1007/978-3-031-73116-7_2)                                                  |             96.1/ 79.3              | 75.34 ```(8-512)```  |            82.9             |  88.45   |   ✅    |
|  Spike-driven Transformer(SDT v1)   |                     [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ca0f5358dbadda74b3049711887e9ead-Abstract-Conference.html)                     |       95.6/ 78.4```(2-512)```       |        77.07         |      80.0 ```(T=16)```      |    -     |   ✅    |
|       Meta-SpikeFormer(SDT v2)      |                                                                [ICLR 2024](https://arxiv.org/abs/2404.03663)                                                                 |             - / -                   |        80.00        |      -      |    -     |   ✅    |
|        E-SpikeFormer(SDT v3)        |                                                                [TPAMI](https://arxiv.org/abs/2411.16061)                                                                     |             - / -                    |    86.20 ```(T=8)``` |      -      |    -     |   ❌    |
|                  MST                |                                                                [ICCV 2023](https://arxiv.org/abs/2210.01208)                                                                     |  97.27 / 86.91  ```(ANN-to-SNN)```  |  78.51 ```(ANN-to-SNN)``` |      88.12 ```(ANN-to-SNN)```      |    91.38 ```(ANN-to-SNN)```     |    ❌    |
|                 QSD                   |                                                                [ICLR 2025](https://arxiv.org/abs/2501.13492)                                                                 |  98.4 / 87.6    ```(Transfer Learning)``` |        80.3          |      89.8 ```(Transfer Learning)```      |    -     |    ❌    |
|         Spiking Transformer         |                                                                [CVPR 2025](https://arxiv.org/abs/2503.00226)                                                                 |           96.32 / 79.69             |        78.66 ```(10-512)```         |      -      |    -     |  ❌      |
|               SNN-ViT               |                                                                [ICLR 2025](https://openreview.net/forum?id=qzZsz6MuEq)                                                                 |           96.1 / 80.1             |        80.23         |      82.3      |    -     |     ❌    |
|               STSSA               |                                                                [ICASSP 2025](https://ieeexplore.ieee.org/document/10890026)                                                                 |           - / -             |        -         |      83.8      |    81.65     |   ❌     |
|          Spikformer + SEMM          |                                                          [NeurIPS 2024](https://openreview.net/forum?id=WcIeEtY3AG)                                                          |            95.78/ 79.04             | 75.93 ```(8-512)```  |            82.32            |    -   |      ✅    |
|          SpikingResformer           | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Shi_SpikingResformer_Bridging_ResNet_and_Vision_Transformer_in_Spiking_Neural_Networks_CVPR_2024_paper.html) | 97.40/ 85.98 ```(Transfer Learning)``` |        79.40         | 84.8 ```(Transfer Learning)``` |    -     |  ✅        |
|                 TIM                 |                                                        [IJCAI 2024](https://www.ijcai.org/proceedings/2024/0347.pdf)                                                         |                  -                  |          -           |            81.6             |  79.00   |     ✅     |

#### 3D Point Cloud Models
These results reproduce the original-paper scores and are updated periodically.
- Default configs: ```4-512 (4 steps)``` for both datasets. 
- mAcc(%) are record in the chart
- 
|           Model           |                  Pub. Info.                   |  Modelnet40  | Modelnet 10 | ScanObjectNN | Status |
|:-------------------------:|:---------------------------------------------:|:------------:|:-----------:|:------------:|:------:|
| Spiking Point Transformer | [AAAI 2025](https://arxiv.org/abs/2502.15811) | 93.54 |    89.39    |     74.53     |  ✅   |


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
    │   └── ...
    ├── models
    │   └── static
    │       ├── spikformer_cifar.py
    │       └── ...
    ├── requirements.txt 
    ├── train.py
    └── utils
        ├── __init__.py
        └── node.py
```

2. **Install the required packages.**
BrainCog is a **prerequisite for all tasks**. For installation instructions, please refer to [Here](../README.md#braincog-installation).
```angular2html
    conda create -n [your_env_name] python=3.8 -y
    conda activate [your_env_name]
    pip install -r requirements.txt
```

3. **Model Configuration**

    To configure your model and place it under ```models/static``` or ```model/dvs``` directory, along with registering the model using timm‘s register function.
    
    To write config files and place them under ```configs/[your_model]``` directory, the format should **strictly** follow the format of the existing config files. We highly recommend this method which keeps structures of all ```.yml``` files clear and consistent. The reason for separating model configuration and training hyperparameters is to facilitate debugging and make the tuning process easier.
    
    Eventually, to import the registered model in ```train.py```

<details>
<summary> Config File Example (Spikformer/cifar10.yml) </summary>

```
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

**For 3D Point Cloud Datasets:**
```angular2html
    python train_pc.py --config configs/[your_model]/[your_dataset].yml 
```
###


### Supported Datasets
|                                                 Dataset                                                 |      Type      |    Mission     | |                                                    Dataset                                                    |      Type      |    Mission     |
|:-------------------------------------------------------------------------------------------------------:|:--------------:|:--------------:|:-:|:-------------------------------------------------------------------------------------------------------------:|:--------------:|:--------------:|
|                         [CIFAR10 ](https://www.cs.toronto.edu/~kriz/cifar.html)                         |     Static     |      cls       | |                            [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)                            |     Static     |      cls       |
|                                 [ImageNet](https://www.image-net.org/)                                  |     Static     |      cls       | |                                  [MNIST](http://yann.lecun.com/exdb/mnist/)                                   |     Static     |      cls       |
|                               [sMNIST](https://arxiv.org/abs/1504.00941)                                |     Static     | sequential_cls | |                                  [psMNIST](https://arxiv.org/abs/1504.00941)                                  |     Static     | sequential_cls |
|                               [sCIFAR](https://arxiv.org/abs/1710.02224)                                |     Static     | sequential_cls | |                                                       -                                                       |       -        |       -        |
|                                                                                                         |                |                | |                                                                                                               |                |                |
| [CIFAR10-DVS](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00309/full) |      DVS       |      cls       | | [DVS-Gesture](https://research.ibm.com/publications/a-low-power-fully-event-based-gesture-recognition-system) |      DVS       |      cls       | 
|                      [NCARS](https://www.prophesee.ai/2018/03/13/dataset-n-cars/)                       |      DVS       |      cls       | |                     [N-CALTECH101](https://www.garrickorchard.com/datasets/n-caltech101)                      |      DVS       |      cls       |
|                            [HMDB51-DVS](https://arxiv.org/pdf/1910.03579v2)                             |      DVS       |      cls       | |                               [UCF101-DVS](https://arxiv.org/pdf/1910.03579v2)                                |      DVS       |      cls       |
|                                                                                                         |                |                | |                                                                                                               |                |                |
|                           [Modenet40/10](https://modelnet.cs.princeton.edu/)                            | 3D Point Cloud |      cls       | |                              [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/)                               | 3D Point Cloud |      cls       |

## Visualization
We are currently developing content in the **Benchmark** that is used to **visualize the performance of the Transformer**. You can access it by executing the following commands:

```angular2html
cd SpikingTransformerBenchmark 
conda activate [env]
python -m cls.vis.gradcam_vis
```
Please **make sure to follow the instructions exactly as shown above**, otherwise, errors may occur due to **path issues**.

### 🙋‍♂️FAQ
<details>
   <summary>Q: Why some results in the repository may differ from those reported in the original paper?</summary>
   A: To ensure fairness and consistency, the paper did not use exactly the same training scripts and strategies as the original work, which resulted in some differences in the outcomes.
</details>

<details>
   <summary>Q: Why are there multiple training scripts in the cls file?</summary>
   A: Different kinds of datasets require distinct hyper-parameters and data-augmentation strategies, so for ease of use we have divided them into three separate scripts.
</details>
