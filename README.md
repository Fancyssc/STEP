# STEP: A Unified Spiking Transformer Evaluation Platform for Fair and Reproducible Benchmarking [NeurIPS 2025](https://arxiv.org/abs/2505.11151)

<p align="center">
  <img src="/imgs/STEP.jpg" alt="mp" style="width: 40%; max-width: 600px; min-width: 200px;" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20|%203.10-blue" alt="python"/>
  <img src="https://img.shields.io/badge/framework-BrainCog-blue" alt="Braincog"/>
  <img src="https://img.shields.io/badge/version-1.1.0-green" alt="Version"/>
  <img src="https://img.shields.io/badge/-continuous_integration-red" alt="Contiguous"/>
</p>

## 🚀 Quick Start

### BrainCog Installation
For the [BrainCog](https://github.com/BrainCog-X/Brain-Cog) framework, we recommend installing it via GitHub. You can use the following command in your terminal to install it from GitHub:
```angular2html
pip install git+https://github.com/braincog-X/Brain-Cog.git
```
### Very First Step for **STEP**
#### STEP installation
```bash
git clone https://github.com/Fancyssc/STEP.git
```
#### 
Start Spikformer Training on CIFAR10 as the "Hello-world" Demo.
```bash
conda activate [your_env]
python train.py --config configs/spikformer/cifar10.yml
```


[//]: # (## ⚡ Introduction)

[//]: # ()
[//]: # (Built on top of **[BrainCog]&#40;https://github.com/BrainCog-X/Brain-Cog&#41;**, this repository reproduces state-of-the-art Spiking Transformer models and offers a unified pipeline for **classification, segmentation, and object detection**. By standardizing data loaders, training routines, and logging, it enables fair, reproducible comparisons while remaining easy to extend with new models or tasks.)

[//]: # ()
[//]: # (- **Modular Design** – Swap neuron models, encodings, or attention blocks with a few lines of code.  )

[//]: # (- **Multi-Task Ready** – Shared backbone, task-specific heads; evaluate *once*, report *everywhere*.  )

[//]: # (- **Cross-Framework Compatibility** – Runs on BrainCog, SpikingJelly, or BrainPy with a thin adapter layer.  )

[//]: # (- **End-to-End Reproducibility** – Version-locked configs and CI scripts guarantee “one-command” reruns.  )

## 📂 DeepDive Guides
For specific tasks, completed models and supported datasets, please refer to the corresponding submodule guides:
- [Classification(cls)](./cls/README.md)  
- [Segmentation(seg)](./seg/README.md)  
- [Detection(det)](./det/README.md)
```plaintext
Spiking-Transformer-Benchmark/
├── cls/               # Classification submodule
│   ├── README.md      
│   └── ...
├── seg/               # Segmentation submodule 
│   ├── README.md      
│   └── ...
├── det/               # Object detection submodule 
│   ├── README.md      
│   └── ...
└── README.md          
```

## 🔑 Key Features of STEP

<p align="center">
  <img src="/imgs/bench.png" alt="mp" style="width: 60%; max-width: 600px; min-width: 200px;" />
</p>

One-stop benchmark for Spiking-Transformer research—classification, segmentation, and detection share the same training & evaluation pipeline.
- **Plug-and-play modules** (neurons, encodings, attention, surrogate gradients, heads) let you prototype new ideas without touching the core loop.
- **Ready-made loaders** cover ImageNet, CIFAR, DVS-CIFAR10, N-Caltech101 … 
- **Task adapters integrate with MMSeg and MMDet**, so dense prediction experiments need only a config tweak.
- **Backend-agnostic code** runs on SpikingJelly, BrainCog, or BrainPy, and every config is version-locked for full reproducibility. neuromorphic vision systems.

### Supported Datasets
#### cls 
**Static**(e.g. CIFAR/ImageNet...)**/Neuromorphic**(e.g. CIFAR10-DVS)/**3D Point Cloud**(e.g. ModelNet40) classification datasets are supported.

See [cls/README.md](./cls/README.md#supported-datasets) for details.

#### seg/det
Frequently used datasets for both tasks which are assembled by MMSeg and MMDet are supported. 

See [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [MMDetection](https://github.com/open-mmlab/mmdetection) for details.

####

## Resources

### Tutorial
The brief tutorial of STEP can be found [here](https://step.readthedocs.io/en/latest/).

### Checkpoints
The main experimental results, including the corresponding log files, configuration files, and checkpoints, can be downloaded [here](https://huggingface.co/Fancysean/STEP).

### Integrated Neurons
|                Neuron Node                 |                              abbreviation                               ||              Neuron Node              |                             abbreviation                              |
|:------------------------------------------:|:-----------------------------------------------------------------------:|:--:|:-------------------------------------:|:---------------------------------------------------------------------:|
|         Integrate-and-Fire Neuron          |                                   IF                                    ||    Leaky Integrate-and-Fire Neuron    |                                  LIF                                  |
| Parametric Leaky Integrate-and-Fire Neuron |                                  PLIF                                   || Exponential Integrate-and-Fire Neuron | [EIF](https://journals.physiology.org/doi/full/10.1152/jn.00686.2005) |
|             Integer LIF Neuron             | [I-LIF](https://link.springer.com/chapter/10.1007/978-3-031-73411-3_15) ||    Normarlized Interger LIF Neuron    |   [NI-LIF](https://ojs.aaai.org/index.php/AAAI/article/view/32126)    |
|         Hybrid Dynamics LIF Neuron         |    [HD-LIF](https://ojs.aaai.org/index.php/AAAI/article/view/35459)     ||           Gated LIF Neuron            |               [GLIF](https://arxiv.org/abs/2210.13768)                |
|             k-based LIF Neuron             |                [KLIF]( https://arxiv.org/abs/2302.09238)                ||       Complementary LIF Neuron        |               [CLIF](https://arxiv.org/abs/2402.04663 )               |
|             Parallel Spiking Neuron        |                 [PSN](https://arxiv.org/abs/2304.12760)                 ||                Hodgkin-Huxley Neuron                        |                              [HHNode](https://pmc.ncbi.nlm.nih.gov/articles/PMC1392413/)                               |
|             Izhikevich Neuron        |               [IzhNode](https://ieeexplore.ieee.org/abstract/document/1257420)               ||                |                                                                       |

### Supported Encoding
When working with static datasets, SNNs typically require encoding of static images. Our framework supports various encoding methods, including **direct, rate, TTFS, and phase** encoding.



## 📝Citation
```angular2html
@misc{shen2025stepunifiedspikingtransformer,
      title={STEP: A Unified Spiking Transformer Evaluation Platform for Fair and Reproducible Benchmarking}, 
      author={Sicheng Shen and Dongcheng Zhao and Linghao Feng and Zeyang Yue and Jindong Li and Tenglong Li and Guobin Shen and Yi Zeng},
      year={2025},
      eprint={2505.11151},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2505.11151}, 
}
```

## 📦 Version & Changelog
### [1.1.0] – 2025-07-03
#### Added
- Support for **3D cloud point classification**.
- Some known bugs fixed.

### [1.0.0] – 2025-5-18
#### Added
- initial version released.


## 💡Acknowledgement
Thanks to the [BrainCog](https://github.com/BrainCog-X/Brain-Cog) for providing the core ideas and components for this repository.

A full list of contributors can be found [here](https://github.com/Fancyssc/STEP/graphs/contributors).
