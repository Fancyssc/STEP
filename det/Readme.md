## Detection

Detection is a crucial component in image processing. In our implementation, we use the widely adopted [MMDetection](https://github.com/open-mmlab/mmdetection) 
framework to handle detection tasks, while leveraging BrainCog as the backbone framework for building Spiking Neural Networks (SNNs).

The usage of mmdet is very similar to 

### Guidelines
#### Code Environment
1. **The path configuration for your code repository should be as follows.**
```angular2html
.
├── Readme.md
└── cls
    ├── configs
    │   ├── _base_
    │   ├──spikformer
    │   └── ...
    ├── mmdet
    │   ├── apis
    │   ├── datasets
    │   ├── models
    │   ├── engine
    │   └── ...
    ├── requirements
    │   ├── albu.txt
    │   └── ...
    ├── tools
    │   ├── train.py
    │   └── ...
    ├── readme.md
    ├── setup.p
    └── ...
```

2. **Environment**
The mmseg framework has strict environment requirements. We recommend aligning your package versions with the ones listed
below to ensure proper functionality.

3. If you
```angular2html
    python == 3.10.x
    torch == 2.0.1+cu118
    mmcv == 2.0.1
    mmengine == 0.10.7
    mmdet == 3.1.0
    numpy == 1.26.4
```
It is important to note that due to compatibility issues with mmseg, servers running CUDA 12.x may still encounter errors 
even if all package versions are aligned as recommended. **Therefore, we strongly recommend using a CUDA 11.x environment for optimal stability**.

3. **Usage**
- **model construction**: Create your model and place it under ```mmdet/models/[model_type]/```.
- **model register**: Register your model with ```@MODELS.register_module()``` then import it in the corresponding ```__init__.py```.
- **model config**: Create a config file for your model and place it under ```configs/[model]/```. You can import some pre-defined configs from ```configs/_base_/```.

4. **Training**
Here is an example for training Spikformer 8-512 on 1 GPU:
- `cd tools`
- `CUDA_VISIBLE_DEVICES=0 ./dist_train.sh ../configs/xxxx/xxxx 1`