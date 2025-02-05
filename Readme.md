# Spiking Transformer Benchmark
This is based on the Spiking Transformer framework developed by [BrainCog](https://github.com/BrainCog-X/Brain-Cog). The framework integrates most of the existing open-source Spiking Transformers and their evaluation results on the corresponding datasets.
Our code repository will remain updated. If you have any questions, please feel free to contact us.


## Classification

### Implemented & To-implement Models

|   Model    |                    Pub. Info.                    |   Status    |              Model               |                                                              Pub. Info.                                                              |    Status    |
|:----------:|:------------------------------------------------:|:-----------:|:--------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------:|:------------:|
| Spikformer |  [ICLR 2023](https://arxiv.org/abs/2209.15425)   |   testing   | Spike-driven Transformer(SDT v1) | [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ca0f5358dbadda74b3049711887e9ead-Abstract-Conference.html) | Implemented  |
|  QKFormer  | [NeurIPS 2024](https://arxiv.org/abs/2403.16552) | implemented |               TIM                |                              [IJCAI 2024](https://www.ijcai.org/proceedings/2024/0347.pdf)                                           | To Implement |
More models are to be implemented soon...


### Experiment Results
The default neuron node used in spiking transformers are `LIFNode(tau=2.,thres=1.0,Sigmoid_Grad(alpha=4.))` and the models are in the mode of `layer by layer`. If any 
special conditions are considered, it will be noted in the supp. of the table.

Other hyper-param setting are following the original paper.
#### CIFAR
|   Model    | Batch-Size | Dataset  | Step | Epoch | Result(Acc@1) |  supp.   |
|:----------:|:----------:|:--------:|:-------:|:-----:|:-------------:|:--------:|
| Spikformer |    128     | CIFAR10  | 4 |  300  | 94.47(-0.72)  |    -     |
| Spikformer |    128     | CIFAR10  | 4 |  400  | 95.03(-0.48)  |    -     |
||            ||||
|    SDT     |     64     | CIFAR10  | 4 |  300  | 95.26(-0.34)  |    -     |
||            ||||
|  QKFormer  |     64     | CIFAR10  | 4 |  400  |  96.5(+0.32)  |    -     |

#### ImageNet-1K
to be updated

### How Can You Run
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

2. **Install the required packages.**
```angular2html
    conda create -n [your_env_name] python=3.8 -y
    conda activate [your_env_name]
    pip install -r requirements.txt
```

3. **Configure your model**
    To configure your model and place it under ```models/static``` or ```model/dvs``` directory, along with registering the model using timm‘s register function.
    
    To write config files and place them under ```configs/[your_model]``` directory, the format should **strictly** flollow the format of the existing config files. We highly recommend this method even you can use only one ```.yml``` file to config the model. The reason for separating model configuration and training hyperparameters is to facilitate debugging and make the tuning process easier.
    
    Eventually, to import the registered model in ```train.py```.


4. **Run the training script**
Since dynamic and static datasets typically use different loading methods and data augmentation techniques, most models employ two separate scripts with corresponding augmentation strategies. Therefore, we also divide the scripts into two here.

**For Static Datasets:**
```angular2html
    python train.py --model-config configs/[your_model]/[your_dataset].yml --train-config configs/[your_model]/train.yml
```



**For DVS Datasets:**
```angular2html
    python train_dvs.py --model-config configs/[your_model]/[your_dataset].yml --train-config configs/[your_model]/train.yml
```
