Note: This repository will be fully updated once get the official notification.

# Classification of Visualization Types and Perspectives in Patents

This repository represents the implementation for public work "Classification of Visualization Types and Perspectives in Patents"

# Full paper
https://link.springer.com/chapter/10.1007/978-3-031-43849-3_16 

# Methodology Pipeline

![model pipeline](media/uniModalPipeline.png)

## Get started (Requirements and Setup)
Python version >= 3.9

## Datasets

Extended_CLEF_IP_2011_Dataset: https://zenodo.org/records/10019328

USPTO_PIP_Dataset: https://zenodo.org/records/10019506

## Training and testing
```bash
train.py -b 32 --epochs 200 --workers 1 --lr 0.001 --batchbalance yes --augmentdata yes --basemodel resnext101_64x4d --featureSize 2048 --featurelayer -1 --imageSize 224 --output ./outputDir

```

## Cite as
```bash
@InProceedings{10.1007/978-3-031-43849-3_16,
author="Ghauri, Junaid Ahmed
and M{\"u}ller-Budack, Eric
and Ewerth, Ralph",
editor="Alonso, Omar
and Cousijn, Helena
and Silvello, Gianmaria
and Marrero, M{\'o}nica
and Teixeira Lopes, Carla
and Marchesin, Stefano",
title="Classification of Visualization Types and Perspectives in Patents",
booktitle="Linking Theory and Practice of Digital Libraries",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="182--191",
abstract="Due to the swift growth of patent applications each year, information and multimedia retrieval approaches that facilitate patent exploration and retrieval are of utmost importance. Different types of visualizations (e.g., graphs, technical drawings) and perspectives (e.g., side view, perspective) are used to visualize details of innovations in patents. The classification of these images enables a more efficient search in digital libraries and allows for further analysis. So far, datasets for image type classification miss some important visualization types for patents. Furthermore, related work does not make use of recent deep learning approaches including transformers. In this paper, we adopt state-of-the-art deep learning methods for the classification of visualization types and perspectives in patent images. We extend the CLEF-IP dataset for image type classification in patents to ten classes and provide manual ground truth annotations. In addition, we derive a set of hierarchical classes from a dataset that provides weakly-labeled data for image perspectives. Experimental results have demonstrated the feasibility of the proposed approaches. Source code, models, and datasets are publicly available (https://github.com/TIBHannover/PatentImageClassification).",
isbn="978-3-031-43849-3"
}

OR

Ghauri, J.A., Müller-Budack, E., Ewerth, R. (2023). Classification of Visualization Types and Perspectives in Patents. In: Alonso, O., Cousijn, H., Silvello, G., Marrero, M., Teixeira Lopes, C., Marchesin, S. (eds) Linking Theory and Practice of Digital Libraries. TPDL 2023. Lecture Notes in Computer Science, vol 14241. Springer, Cham. https://doi.org/10.1007/978-3-031-43849-3_16
```
