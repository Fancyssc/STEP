# 
This framework primarily supports Spiking Transformer(certainly also available for other SNN models) for tasks such as classification, segmentation, and object detection.

Supported neuron models, datasets, experimental results, and open checkpoints can be found in the official [GitHub repository](https://github.com/Fancyssc/STEP.git) and [Hugging Face repository](https://huggingface.co/Fancysean/STEP).


## Classification(cls)
We recommend using conda to manage your virtual environments and keeping the environments for classification and segmentation/detection **separate**.

This is because our testing has shown that the classification (cls) environment is highly robust and can maintain compatibility across multiple versions of Python packages. In contrast, the segmentation (seg) and detection (det) environments rely on frameworks developed by MMLab, which are relatively more sensitive to dependencies. Therefore, we recommend keeping these environments separate.
```angular2html
    conda create -n [your_env_name] python=3.8 -y
    conda activate [your_env_name]
    cd cls
    pip install -r requirements.txt
```

## Segmentation(seg) & Detection(det)
For the segmentation (seg) and detection (det) environments, we recommend the following configurations:
```angular2html
    python == 3.10.x
    torch == 2.0.1+cu118
    mmcv == 2.0.1
    mmengine == 0.10.7
    mmdet == 3.1.0
    numpy == 1.26.4
```
**Important Note:**
Both mmseg and mmdet may fail to compile on machines with CUDA 12.x. Therefore, we strongly recommend running segmentation and detection tasks on machines with **CUDA 11.x** for compatibility.
