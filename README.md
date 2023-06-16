# Patent Image Classification

This is implementation for public work "Classification of Visualization Types and Perspectives in Patents"

# Methodology Pipeline

![model pipeline](media/uniModalPipeline.png)

## Get started (Requirements and Setup)
Python version >= 3.9

## Dataset
Dataset distribution for train, validation and test set. 

## Training and testing
```bash
train.py -b 32 --epochs 200 --workers 1 --lr 0.001 --batchbalance yes --augmentdata yes --basemodel resnext101_64x4d --featureSize 2048 --featurelayer -1 --imageSize 224 --output ./outputDir

```

