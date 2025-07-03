# Tranining
Because each task relies on different data-augmentation strategies, we have split the augmentations into separate scripts. For the Seg and Det tasks, simply follow the configurations specified in their respective .md files.

## Start Training
**Run the training script**
Since dynamic and static datasets typically use different loading methods and data augmentation techniques, most models employ two separate scripts with corresponding augmentation strategies. Therefore, we also divide the scripts into two here.

**For Static Datasets:**
```angular2html
    python train.py --config configs/[your_model]/[your_dataset].yml
```

**For DVS Datasets:**
```angular2html
    python train_dvs.py --config configs/[your_model]/[your_dataset].yml 
```

**For 3D Point Cloud Datasets:**
```angular2html
    python train_pc.py --config configs/[your_model]/[your_dataset].yml 
```