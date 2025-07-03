# STEP: A Unified Spiking Transformer Evaluation Platform for Fair and Reproducible Benchmarking

<p align="center">
  <img src="/imgs/STEP.jpg" alt="mp" style="width: 40%; max-width: 600px; min-width: 200px;" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20|%203.10-blue" alt="python"/>
  <img src="https://img.shields.io/badge/framework-BrainCog-blue" alt="Braincog"/>
  <img src="https://img.shields.io/badge/version-1.0.0-green" alt="Version"/>
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


## 🔑 Key Features of STEP

<p align="center">
  <img src="/imgs/bench.png" alt="mp" style="width: 60%; max-width: 600px; min-width: 200px;" />
</p>

One-stop benchmark for Spiking-Transformer research—classification, segmentation, and detection share the same training & evaluation pipeline.
- **Plug-and-play modules** (neurons, encodings, attention, surrogate gradients, heads) let you prototype new ideas without touching the core loop.
- **Ready-made loaders** cover ImageNet, CIFAR, DVS-CIFAR10, N-Caltech101 … adding a new dataset is ~50 lines.
- **Task adapters integrate with MMSeg and MMDet**, so dense prediction experiments need only a config tweak.
- **Backend-agnostic code** runs on SpikingJelly, BrainCog, or BrainPy, and every config is version-locked for full reproducibility. neuromorphic vision systems.


## 📂 DeepDive Guides
- [Classification(Cls)](./cls/README.md)  
- [Segmentation(Seg)](./seg/README.md)  
- [Detection(Det)](./det/README.md)
```plaintext
Spiking-Transformer-Benchmark/
├── cls/               # Classification submodule
│   ├── README.md      
│   ├── configs/     
│   ├── datasets/      
│   └── ...
├── seg/               # Segmentation submodule 
│   ├── README.md      
│   ├── configs/       
│   ├── mmseg      
│   └── ...
├── det/               # Object detection submodule 
│   ├── README.md      
│   ├── configs/       
│   ├── mmdet      
│   └── ...
└── README.md          
```

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

## 💡Acknowledgement
Thanks to the [BrainCog](https://github.com/BrainCog-X/Brain-Cog) for providing the core ideas and components for this repository.