# Semantic Segmentation Models for Fisheye Automotive Images

## Overview

This repository contains the code and resources for the comparative analysis of two state-of-the-art semantic segmentation models, DenseASPP and GatedSCNN, on the WoodScapes dataset. The dataset features fisheye images commonly used in automotive applications, presenting significant radial distortion challenges.

The study employs transfer learning to adapt the models, originally trained on the Cityscapes dataset, to handle distortions in WoodScapes. Results indicate that the GatedSCNN model outperforms DenseASPP in mean Intersection over Union (mIoU) and F1-score, showing better boundary precision and class differentiation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Experiments](#experiments)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Pre-trained Models**: Utilizes DenseASPP and GatedSCNN models pre-trained on Cityscapes.
- **Transfer Learning**: Adapts models to the WoodScapes dataset with radial distortions.
- **Performance Metrics**: Evaluates models using mIoU and F1-score.

## Installation

### Prerequisites

- Python 3.6 or higher
- PyTorch
- CUDA (for GPU support)


## Dataset

The WoodScapes dataset is used for training and evaluation. It includes high-resolution fisheye images with significant radial distortion.

- **Download**: [WoodScapes Dataset](http://link_to_dataset.com)
- **Preparation**: Follow the scripts provided in the `data_preparation` directory to preprocess and annotate the dataset.

## Experiments

### Experiment 1: Pre-trained Model Evaluation

- **Objective**: Evaluate pre-trained models on Cityscapes and WoodScapes datasets.
- **Results**: Initial poor performance due to radial distortions.

### Experiment 2: Fine-tuning

- **Objective**: Adapt models using transfer learning.
- **Results**: Significant performance improvement on WoodScapes.

## Results

- **Metrics**: mIoU and F1-score
- **Comparison**:
  - DenseASPP: mIoU - 38.54%, F1-score - 0.46
  - GatedSCNN: mIoU - 50.28%, F1-score - 0.62

## Contributing

Contributions are welcome! Please follow the [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

- **Andrea Marinelli** - andrea.marinelli@studenti.unipd.it
- **Gianmarco Betti** - gianmarco.betti@studenti.unipd.it

For more information, please refer to the [project report](https://github.com/andrea3425/semantic_segmentation_models_for_fisheye_automotive_images/blob/main/Computer_Vision_Project.pdf).
