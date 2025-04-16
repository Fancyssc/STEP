# Spiking Transformer Benchmark
English | [简体中文](Readme_CN.md)

This is based on the Spiking Transformer framework developed by [BrainCog](https://github.com/BrainCog-X/Brain-Cog). The framework integrates most of the existing open-source Spiking Transformers and their evaluation results on the corresponding datasets.
Our code repository will remain updated. If you have any questions, please feel free to contact us.


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
|        SDT        |     64     | CIFAR10 | 4 |  300  |     95.66     |                          -                           |
|                   |            |         ||       |
|     QKFormer      |     64     | CIFAR10 | 4 |  400  |     96.48     |                          -                           |
|                   |            |         ||       |
|   Spikingformer   |    128     | CIFAR10 | 4 |  400  |     95.53     |                          -                           |
|                   |            |         ||       |
| Spikformer + SEMM |    128     | CIFAR10 | 4 |  400  |     94.98     |                          -                           |
|                   |            |         ||       |
|  Spiking Wavelet  |    128     | CIFAR10 | 4 |  400  |     95.31     |                          -                           |
|                   |            |         ||       |
|     SGLFormer     |     16     | CIFAR10 | 4 |  400  |     95.88     |                          -                           |
|                   |            |         ||       |
| Spikingresformer  |    128     | CIFAR10 | 4 |  600  |     95.39     |          Transfer Learning Used Originally           |
|                   |            |         ||       |

#### CIFAR 100
|       Model       | Batch-Size | Dataset  | Step | Epoch | Result(Acc@1) |               supp.               |
|:-----------------:|:----------:|:--------:|:-------:|:-----:|:-------------:|:---------------------------------:|
|    Spikformer     |    128     | CIFAR100 | 4 |  400  |     77.37     |                 -                 |
|                   |            |          ||       |
|        SDT        |     64     | CIFAR10  | 4 |  300  |     79.18     |                 -                 |
|                   |            |          ||       |
|     QKFormer      |     64     | CIFAR100 | 4 |  400  |     81.05     |                 -                 |
|                   |            |          ||       |
|   Spikingformer   |    128     | CIFAR100 | 4 |  400  |     79.12     |                 -                 |
|                   |            |          ||       |
| Spikformer + SEMM |    128     | CIFAR100 | 4 |  400  |     77.59     |                 -                 |
|                   |            |          ||       |
|  Spiking Wavelet  |    128     | CIFAR100 | 4 |  400  |     76.99     |                 -                 |
|                   |            |          ||       |
|     SGLFormer     |     16     | CIFAR100 | 4 |  400  |     80.61     |                 -                 |
|                   |            |          ||       |
| Spikingresformer  |    128     | CIFAR100 | 4 |  600  |     78.29     | Transfer Learning Used Originally |
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


#### Meta-Transformer Test
 **random_attn**: Query and Key are randomly initialized and not updated. 

 **vit_embed**: Using vanilla ViT embedding method and learnable position encoding.

 **conv2_embed**: Reduce the num. of conv layers from 4 to 2 in SPS of Spiking Transformer.

|            Model            | Batch-Size |       Node        | Step | Seed | Result(Acc@1) |          supp.          |
|:---------------------------:|:----------:|:-----------------:|:-------:|:----:|:-------------:|:-----------------------:|
|         Spikformer          |    128     |     Braincog      | 4 |  42  |     94.96     |       random_attn       |
|         Spikformer          |    128     |     Braincog      | 4 |  42  |     78.21     |        vit_embed        |
|         Spikformer          |    128     |     Braincog      | 4 |  42  |     91.92     |       conv2_embed       |
|                             |            |                   ||      |
|             SDT             |     64     |     Braincog      | 4 |  42  |     95.51     |       random_attn       |
|             SDT             |     64     |     Braincog      | 4 |  42  |     82.17     |        vit_embed        |
|             SDT             |     64     |     Braincog      | 4 |  42  |     93.03     |       conv2_embed       |
|                             |            |                   ||      |
|      Spikformer + SEMM      |    128     |     Braincog      | 4 |  42  |     95.57     |       random_attn       |
|      Spikformer + SEMM      |    128     |     Braincog      | 4 |  42  |     89.24     |        vit_embed        |
|      Spikformer + SEMM      |    128     |     Braincog      | 4 |  42  |     93.33     |       conv2_embed       |
|                             |            |                   ||      |
|             ViT             |    128     |     Braincog      | 4 |  42  |     96.41     |       random_attn       |
|             ViT             |    128     |     Braincog      | 4 |  42  |     90.89     |        vit_embed        |
|             ViT             |    128     |     Braincog      | 4 |  42  |     95.0      |       conv2_embed       |
|             ViT             |    128     |     Braincog      | 4 |  42  |     88.46     | vit_embed & random_attn |

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
|                                                 Dataset                                                 |  Type  | Mission | |                                                    Dataset                                                    |  Type  | Mission  |
|:-------------------------------------------------------------------------------------------------------:|:------:|:-------:|:-:|:-------------------------------------------------------------------------------------------------------------:|:------:|:--------:|
|                                              [CIFAR10 ](https://www.cs.toronto.edu/~kriz/cifar.html)                                               | Static |   cls   | |                                                 [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)                                                  | Static |   cls    |
|                                 [ImageNet](https://www.image-net.org/)                                  | Static |   cls   | |                                  [MNIST](http://yann.lecun.com/exdb/mnist/)                                   | Static |   cls    |
|                                                                                                         |        |         | |                                                                                                               |        |          |
| [CIFAR10-DVS](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00309/full) |  DVS   |   cls   | | [DVS-Gesture](https://research.ibm.com/publications/a-low-power-fully-event-based-gesture-recognition-system) |  DVS   |   cls    | 
|                      [NCARS](https://www.prophesee.ai/2018/03/13/dataset-n-cars/)                       |  DVS   |   cls   | |                     [N-CALTECH101](https://www.garrickorchard.com/datasets/n-caltech101)                      |  DVS   |   cls    |
|                            [HMDB51-DVS](https://arxiv.org/pdf/1910.03579v2)                             |  DVS   |   cls   | |                               [UCF101-DVS](https://arxiv.org/pdf/1910.03579v2)                                |  DVS   |   cls    |

## Visualization
We are currently developing content in the **Benchmark** that is used to **visualize the performance of the Transformer**. You can access it by executing the following commands:

```angular2html
cd SpikingTransformerBenchmark 
conda activate [env]
python -m cls.vis.attn_map
```
Please **make sure to follow the instructions exactly as shown above**, otherwise, errors may occur due to **path issues**.
