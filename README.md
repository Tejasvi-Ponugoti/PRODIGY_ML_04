# Food Image Classification with Transfer Learning

An exploratory computer-vision project using the Food-101 dataset and an ImageNet-pretrained ResNet50 model to classify food photographs.

## Project scope

The notebook focuses on **food-image classification**. It does not currently estimate calories or nutritional values.

Two experiments are defined:

1. A three-class classifier for apple pie, pizza and omelette
2. An eleven-class classifier covering a broader selection of Food-101 categories

## Dataset

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) contains 101 food categories. The notebook downloads and extracts the dataset, reads its official train/test lists and prepares smaller experiment-specific subsets.

| Experiment | Training images | Validation images | Classes |
|---|---:|---:|---:|
| Small prototype | 2,250 | 750 | 3 |
| Extended prototype | 8,250 | 2,750 | 11 |

## Modelling workflow

- Resize input images to 224 × 224 pixels
- Scale pixel values to the range 0–1
- Apply shear, zoom and horizontal-flip augmentation
- Use ImageNet-pretrained ResNet50 without its original classification head
- Add global average pooling, a 128-unit ReLU layer and dropout
- Train a softmax output layer with SGD
- Save the best validation checkpoint
- Plot training and validation accuracy and loss
- Run predictions on example food images

### Training configuration

| Setting | Value |
|---|---|
| Batch size | 16 |
| Epochs | 30 |
| Learning rate | 0.0001 |
| Momentum | 0.9 |
| Dropout | 0.2 |

## Repository contents

- `TASK05.ipynb` — data preparation, transfer learning and prediction workflow

The Food-101 dataset and trained model checkpoints are not committed because of their size.

## Recommended environment

A GPU-enabled Jupyter or Kaggle environment is recommended.

```bash
git clone https://github.com/Tejasvi-Ponugoti/PRODIGY_ML_04.git
cd PRODIGY_ML_04
pip install tensorflow numpy matplotlib opencv-python jupyter
jupyter notebook TASK05.ipynb
```

The notebook contains environment-specific filesystem paths and should be reviewed and changed for the local or hosted notebook environment before running end to end.

## Results status

No executed training history or validation metric is stored in the committed notebook. This README therefore documents the experiment design without claiming an unverified accuracy value.

## Limitations and future work

- Complete and save a reproducible training run.
- Replace environment-specific paths with configurable relative paths.
- Add test-set evaluation, a confusion matrix and per-class metrics.
- Extend classification to more Food-101 categories.
- Add a separate nutritional database and portion-size method before describing the project as calorie estimation.

## Skills demonstrated

Python · TensorFlow · Keras · ResNet50 · Transfer learning · Image augmentation · Multi-class classification
