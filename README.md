# MobileNet-based Image Classification for Edge Computing

This project demonstrates how to train a MobileNet-based model on the CIFAR-10 dataset, convert the trained model to TensorFlow Lite for deployment on edge devices, and simulate real-time inference using TensorFlow Lite Interpreter. It also integrates federated learning to train a model across multiple edge devices collaboratively while keeping data decentralized.

## Project Overview

**Dataset**: CIFAR-10, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 testing images.

**Model**: MobileNet-based architecture (lightweight, efficient, and designed for edge devices).

**Framework**: TensorFlow for model training and TensorFlow Lite for converting the model for edge deployment.

**Federated Learning**: A simple federated learning setup to train the model collaboratively across multiple edge devices.

**Edge Simulation**: The model's performance (inference speed and latency) is simulated using TensorFlow Lite Interpreter in Google Colab.

---

1. Clone the Repository

```bash
git clone https://github.com/yourusername/Edge-Computing-MobileNet.git
cd Edge-Computing-MobileNet
2. Install Dependencies
bash
!pip install tensorflow tensorflow-federated numpy matplotlib
