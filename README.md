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

git clone https://github.com/elprofessor-15/Edge-Computing-MobileNet.git

2. Install Dependencies

Install the necessary Python packages required for this project

3. Prepare the Dataset

The CIFAR-10 dataset is automatically loaded using TensorFlow’s keras.datasets module. It consists of 10 classes, including animals and vehicles.

Classes:
	•	Airplane
	•	Automobile
	•	Bird
	•	Cat
	•	Deer
	•	Dog
	•	Frog
	•	Horse
	•	Ship
	•	Truck

Training data: 50,000 images
Testing data: 10,000 images

4. Train the MobileNet Model

We use the MobileNet model without the top layers, adding custom classification layers for CIFAR-10. The model is trained for 5 epochs.

5. Convert the Model to TensorFlow Lite

Convert the trained model into a TensorFlow Lite model for deployment on edge devices.

6. Simulate Edge Inference with TensorFlow Lite

Use TensorFlow Lite Interpreter to simulate inference on an edge device. This step demonstrates how TensorFlow Lite can be used for real-time image classification on edge devices.

7. Federated Learning (Optional)

Federated Learning allows multiple devices to collaboratively train a model without sharing their data. Here, we simulate a simple federated learning setup for the MobileNet model.

8. Measure Latency and Validate Edge Performance

Measure the inference latency of the TensorFlow Lite model to validate how well it performs on edge devices.

Conclusion

This project showcases the deployment of a MobileNet-based model for image classification on edge devices using TensorFlow Lite. The integration of federated learning allows for decentralized training across multiple devices, ensuring that data privacy is maintained. Finally, the project measures the latency of the edge device inference to evaluate real-world performance.
