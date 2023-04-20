# Patent Image Classification

This is implementation for public work "Patent Image Classification for Illustration Type and Perspective Using Deep Learning"

```bash
train.py -b 32 --epochs 200 --workers 1 --lr 0.001 --batchbalance yes --augmentdata yes --basemodel resnext101_64x4d --featureSize 2048 --featurelayer -1 --imageSize 224 --output ./outputDir

```


