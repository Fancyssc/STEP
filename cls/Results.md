### Experiment Results
The default neuron node used in spiking transformers are `LIFNode(tau=2.,thres=1.0,Sigmoid_Grad(alpha=4.))` and the models are in the mode of `layer by layer`. If any 
special conditions are considered, it will be noted in the supp. of the table.

Other hyper-param setting are following the original paper.
#### CIFAR 10
|       Model       | Batch-Size | Dataset | Step | Epoch | Result(Acc@1) |                        supp.                         |
|:-----------------:|:----------:|:-------:|:-------:|:-----:|:-------------:|:----------------------------------------------------:|
|    Spikformer     |    128     | CIFAR10 | 4 |  400  |     95.12     |                          -                           |
|                   |            |         ||       |
|        SDT        |    128     | CIFAR10 | 4 |  400  |     95.79     |                          -                           |
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

#### Surrogate Grad Test
|   Model    | Batch-Size | Dataset  | Step | Epoch | Result(Acc@1) | Act_func | Alpha |
|:----------:|:----------:|:--------:|:----:|:-----:|:-------------:|:--------:|:-----:|
| Spikformer |    128     | CIFAR10 |  4   |  400  |     95.12     | Sigmoid  |   4   |
| Spikformer |    128     | CIFAR10 |  4   |  400  |     95.09     | Sigmoid  |   2   |
| Spikformer |    128     | CIFAR10 |  4   |  400  |     95.09     |  QGate   |   4   |
| Spikformer |    128     | CIFAR10 |  4   |  400  |     95.09     |  QGate   |   2   |
|            |            ||      |       |               |          |       |
|    SDT     |    128     | CIFAR10 |  4   |  400  |     95.79     | Sigmiod  |   4   |
|    SDT     |    128     | CIFAR10 |  4   |  400  |     95.81     | Sigmoid  |   2   |
|    SDT     |    128     | CIFAR10 |  4   |  400  |     95.92     |  QGate   |   4   |
|    SDT     |    128     | CIFAR10 |  4   |  400  |     95.81     |  QGate   |   2   |
|            |            ||      |       |               |          |       |


#### Transformer Component Test
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