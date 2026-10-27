# Kaggle MNIST Project 🧠

A modular PyTorch project for training a simple neural network on MNIST, including test-set accuracy evaluation.

## Run on Kaggle

In a Kaggle notebook, run:

```bash
!git clone https://github.com/nameissaurabh/kaggle_mnist_project.git
%cd kaggle_mnist_project
!pip install -r requirements.txt
!python train.py
```

The script downloads the MNIST dataset, trains for the configured number of epochs, reports test accuracy, and saves the trained weights to `mnist_model.pt`.
