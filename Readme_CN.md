# Spiking Transformer Benchmark
[English](Readme.md)|简体中文

该项目基于由 [BrainCog](https://github.com/BrainCog-X/Brain-Cog) 开发的 Spiking Transformer 框架。该框架集成了目前大多数开源的 Spiking Transformer 模型及其在相应数据集上的评估结果。

我们的代码仓库将持续更新。如果您有任何疑问，请随时联系我们。

## 图片分类

### 模型 & 待测模型

|     Model     |                    Pub. Info.                    |   Status    |              Model               |                                                              Pub. Info.                                                              |   Status    |
|:-------------:|:------------------------------------------------:|:-----------:|:--------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------:|:-----------:|
|  Spikformer   |  [ICLR 2023](https://arxiv.org/abs/2209.15425)   |   testing   | Spike-driven Transformer(SDT v1) | [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ca0f5358dbadda74b3049711887e9ead-Abstract-Conference.html) | Implemented |
|   QKFormer    | [NeurIPS 2024](https://arxiv.org/abs/2403.16552) | implemented |               TIM                |                                    [IJCAI 2024](https://www.ijcai.org/proceedings/2024/0347.pdf)                                     | Implemented |
| Spikingformer |    [Arxiv](https://arxiv.org/abs/2304.11954)     |   testing   |                -                 |                                                                  -                                                                   |      -      |

更多模型即将实现……


### 实验结果
Spiking Transformer 中默认使用的神经元节点为 `LIFNode(tau=2.,thres=1.0,Sigmoid_Grad(alpha=4.))`，且模型以 `逐层` 的模式运行。如果有任何特殊情况，将在表格的补充说明中予以注明。

其他超参数设定均遵循原文设定

#### CIFAR
|        Model        | Batch-Size | Dataset  | Step | Epoch | Result(Acc@1) |  supp.   |
|:-------------------:|:----------:|:--------:|:-------:|:-----:|:-------------:|:--------:|
|     Spikformer      |    128     | CIFAR10  | 4 |  300  | 94.47(-0.72)  |    -     |
|     Spikformer      |    128     | CIFAR10  | 4 |  400  | 95.03(-0.48)  |    -     |
|                     |            ||||
|         SDT         |     64     | CIFAR10  | 4 |  300  | 95.26(-0.34)  |    -     |
|                     |            ||||
|      QKFormer       |     64     | CIFAR10  | 4 |  400  |  96.5(+0.32)  |    -     |
|                     |            ||||
|    Spikingformer    |    128     | CIFAR10  | 4 |  400  | 95.34(-0.47)  |    -     |
#### ImageNet-1K
待更新

### Guidelines
#### Code Environment
1. **项目的基本配置路径如下**
```angular2html
.
├── Readme.md
└── cls
    ├── configs
    │   └── spikformer
    │       ├── cifar10.yml
    │       ├── cifar100.yml
    │       ├── imgnet.yml
    │       └── train.yml
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

2. **安装环境**
```angular2html
    conda create -n [your_env_name] python=3.8 -y
    conda activate [your_env_name]
    pip install -r requirements.txt
```

3. **配置模型**

   要配置模型并将其放置于 ```models/static``` 或 ```model/dvs``` 目录下，同时使用 timm 的注册函数注册模型。

编写配置文件并将其放置于 ```configs/[your_model]``` 目录下，文件格式必须**严格**遵循现有配置文件的格式。我们强烈推荐使用这种方法，并且尽可能在```.yml```文件中保持结构的统一和明晰。将模型配置与训练超参数分离的原因在于便于调试并简化调参过程。

最终，在 ```train.py``` 或```train_dvs.py```中导入已注册的模型。


4. **启动训练脚本**

由于动态数据集和静态数据集通常使用不同的加载方法和数据增强技术，大多数模型都采用两套独立的脚本并配备相应的数据增强策略。因此，我们在此也将脚本分为两部分。


**For Static Datasets:**
```angular2html
    python train.py --config configs/[your_model]/[your_dataset].yml 
```



**For DVS Datasets:**
```angular2html
    python train_dvs.py --config configs/[your_model]/[your_dataset].yml 
```

### 支持的数据集
|                                                 Dataset                                                 |  Type  | Mission | |                                                    Dataset                                                    |  Type  | Mission  |
|:-------------------------------------------------------------------------------------------------------:|:------:|:-------:|:-:|:-------------------------------------------------------------------------------------------------------------:|:------:|:--------:|
|                                              [CIFAR10 ](https://www.cs.toronto.edu/~kriz/cifar.html)                                               | Static |   cls   | |                                                 [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)                                                  | Static |   cls    |
|                                 [ImageNet](https://www.image-net.org/)                                  | Static |   cls   | |                                  [MNIST](http://yann.lecun.com/exdb/mnist/)                                   | Static |   cls    |
|                                                                                                         |        |         | |                                                                                                               |        |          |
| [CIFAR10-DVS](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00309/full) |  DVS   |   cls   | | [DVS-Gesture](https://research.ibm.com/publications/a-low-power-fully-event-based-gesture-recognition-system) |  DVS   |   cls    | 
|                      [NCARS](https://www.prophesee.ai/2018/03/13/dataset-n-cars/)                       |  DVS   |   cls   | |                     [N-CALTECH101](https://www.garrickorchard.com/datasets/n-caltech101)                      |  DVS   |   cls    |
|                            [HMDB51-DVS](https://arxiv.org/pdf/1910.03579v2)                             |  DVS   |   cls   | |                               [UCF101-DVS](https://arxiv.org/pdf/1910.03579v2)                                |  DVS   |   cls    |

