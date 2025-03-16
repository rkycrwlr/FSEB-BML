# FSEB-BML

This repository implements **Function-Space Empirical Bayes Regularization (FS-EB)** as introduced in the paper ["Function-Space Empirical Bayes Regularization"](https://arxiv.org/pdf/2312.17162). FS-EB is a novel regularization method that operates in the function space, improving robustness and generalization compared to traditional parameter-space regularization.

## Features

- **FS-EB Regularization**: Implements function-space regularization using empirical Bayes principles.
- **Synthetic and Real Data**: Includes experiments on Two Moons and MNIST datasets.
- **Adversarial Robustness**: Tools for generating adversarial examples and evaluating robustness.
- **Visualization Tools**: Decision boundary plots and histograms for model analysis.

## Directory Structure

fseb-bml/
├── fseb_2d.ipynb # FS-EB implementation on synthetic datasets
├── fseb_mnist.ipynb # FS-EB implementation on MNIST dataset
└── src/
├── adversarial.py # Adversarial attack methods
├── fseb.py # FS-EB regularization implementation
├── model.py # Model architectures (MLP, CNN)
└── utils.py # Utility functions (evaluation, visualization)

## Usage

### Synthetic Dataset
Run `fseb_2d.ipynb` to train models with FS-EB on synthetic datasets like Two Moons.

### MNIST Dataset
Use `fseb_mnist.ipynb` to apply FS-EB regularization on MNIST data.
