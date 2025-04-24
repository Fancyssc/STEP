# Spiking Transformer Benchmark

This repository, built upon the **[BrainCog](https://github.com/BrainCog-X/Brain-Cog)** framework, serves to reproduce existing open-source Spiking Transformer models and provide standardized evaluation results across multiple tasks.

- Classification: [Here](cls/Readme.md)
- Segmentation: [Here](seg/Readme.md)
- Detection: [Here](det/Readme.md)
## Features

- **BrainCog-based**: Integrates and reproduces state-of-the-art Spiking Transformer implementations.
- **Multi-task support**: Includes classification (`cls`) and segmentation (`seg`) submodules, with object detection (`det`) forthcoming.
- **Standardized Evaluation**: Offers unified training and testing pipelines for each task, yielding comparative performance metrics.
- **Extensibility**: Modular design and clear directory structure make it straightforward to add new models and tasks.

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

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/YourUsername/Spiking-Transformer-Benchmark.git

```
For the **seg** and **cls** tasks, different environment requirements apply. Please refer to the corresponding README files in each subdirectory for details.

> **Prerequisites**: Python 3.8 or above, PyTorch, and BrainCog.


## Contributing

Contributions are highly appreciated! To contribute:

1. Fork the repository and create a new branch for your feature or fix.
2. Implement your changes, including examples and documentation.
3. Ensure all tests pass and submit a pull request.

## Contact

If you have questions, suggestions, or encounter issues, feel free to reach out to us by opening an issue.

Thank you for using the Spiking Transformer Benchmark. We hope this project accelerates your research and development!

