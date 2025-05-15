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

## ⚡ Introduction

Built on top of **[BrainCog](https://github.com/BrainCog-X/Brain-Cog)**, this repository reproduces state-of-the-art Spiking Transformer models and offers a unified pipeline for **classification, segmentation, and object detection**. By standardizing data loaders, training routines, and logging, it enables fair, reproducible comparisons while remaining easy to extend with new models or tasks.

- **Modular Design** – Swap neuron models, encodings, or attention blocks with a few lines of code.  
- **Multi-Task Ready** – Shared backbone, task-specific heads; evaluate *once*, report *everywhere*.  
- **Cross-Framework Compatibility** – Runs on BrainCog, SpikingJelly, or BrainPy with a thin adapter layer.  
- **End-to-End Reproducibility** – Version-locked configs and CI scripts guarantee “one-command” reruns.  

### 📂 Task-Specific READMEs

| Task | Documentation |
|------|---------------|
| Classification | [cls/Readme.md](cls/Readme.md) |
| Segmentation   | [seg/Readme.md](seg/Readme.md) |
| Detection      | [det/Readme.md](det/Readme.md) |

## 🔑 Key Features of STEP

<p align="center">
  <img src="/imgs/bench.jpg" alt="mp" style="width: 60%; max-width: 600px; min-width: 200px;" />
</p>

- **Unified Benchmark for Spiking Transformers**  
  STEP offers a single, coherent platform for evaluating classification, segmentation, and detection models, removing fragmented evaluation pipelines and simplifying comparison across studies.

- **Highly Modular Architecture**  
  All major blocks—neuron models, input encodings, attention variants, surrogate gradients, and task heads—are implemented as swappable modules. Researchers can prototype new ideas by mixing and matching components without rewriting the training loop.

- **Broad Dataset Compatibility**  
  Out-of-the-box support spans static vision (ImageNet, CIFAR10/100), event-based neuromorphic data (DVS-CIFAR10, N-Caltech101), and sequential benchmarks. Data loaders follow a common interface, so adding a new dataset is typically a ~50-line effort.

- **Multi-Task Adaptation**  
  Built-in pipelines extend beyond image classification to dense prediction tasks. STEP seamlessly plugs Spiking Transformers into MMSeg (segmentation) and MMDet (object detection) heads such as FCN and FPN, enabling fair cross-task studies with minimal glue code.

- **Backend-Agnostic Implementation**  
  A thin abstraction layer makes the same model definition runnable on SpikingJelly, BrainCog, or BrainPy. This widens hardware and software coverage while promoting reproducible results across laboratories.

- **Reproducibility & Best-Practice Templates**  
  Every experiment ships with version-locked configs, deterministic seeds, and logging utilities. CI scripts validate that reported numbers can be reproduced with a single command, fostering transparent comparison and faster iteration.

> **TL;DR** STEP lowers the barrier to building, training, and fairly benchmarking Spiking Transformers, accelerating progress toward practical neuromorphic vision systems.
## Repository Structure

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
│   └── README.md      
└── README.md          
```

## 🚀 Quick Start

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Fancyssc/STEP.git

```
For the **seg** and **cls** tasks, different environment requirements apply. Please refer to the corresponding README files in each subdirectory for details.

> **Prerequisites**: Python 3.8 or above, PyTorch, and BrainCog.


## Contact & Collaboration

- **Questions or Feedback**  
  If you run into any issues, have questions about STEP, or simply want to share suggestions, please open a GitHub Issue or start a discussion thread. We monitor the repository regularly and aim to respond within a few business days.

- **Integrate Your Model**  
  Have an exciting Spiking Transformer variant or related module you’d like to see supported? We welcome external contributions! Open an Issue describing your model, its licensing, and any specific requirements, or email the maintainers listed in [`MAINTAINERS.md`](./MAINTAINERS.md). We’ll coordinate with you to add the necessary adapters, documentation, and tests.

We look forward to working with the community to make STEP an ever-stronger platform for neuromorphic research.

