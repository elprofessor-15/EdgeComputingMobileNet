MobileNet-based Image Classification for Edge Computing
This project demonstrates how to train a MobileNet-based model on the CIFAR-10 dataset, convert the trained model to TensorFlow Lite for deployment on edge devices, and simulate real-time inference using TensorFlow Lite Interpreter. It also integrates federated learning to train a model across multiple edge devices collaboratively while keeping data decentralized.

Project Overview
Dataset: CIFAR-10, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 testing images.
Model: MobileNet-based architecture (lightweight, efficient, and designed for edge devices).
Framework: TensorFlow for model training and TensorFlow Lite for converting the model for edge deployment.
Federated Learning: A simple federated learning setup to train the model collaboratively across multiple edge devices.
Edge Simulation: The model's performance (inference speed and latency) is simulated using TensorFlow Lite Interpreter in Google Colab.
Key Steps
Train MobileNet on CIFAR-10: Train a MobileNet-based model for image classification.
Convert to TensorFlow Lite: Convert the trained model to TensorFlow Lite format to be deployed on edge devices.
Simulate Edge Inference: Use TensorFlow Lite Interpreter in Google Colab to simulate inference on an edge device.
Federated Learning: Extend the project to use federated learning for training across multiple edge devices.
Setup Instructions
1. Clone the Repository
Clone this repository to your local machine to get started.

bash
Copy code
git clone https://github.com/yourusername/Edge-Computing-MobileNet.git
cd Edge-Computing-MobileNet
2. Install Dependencies
Install the necessary Python packages required for this project:

bash
Copy code
!pip install tensorflow tensorflow-federated numpy matplotlib
3. Prepare the Dataset
The CIFAR-10 dataset is automatically loaded using TensorFlow's keras.datasets module. It consists of 10 classes, including animals and vehicles.

Classes:
Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
Training data: 50,000 images
Testing data: 10,000 images
python
Copy code
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0
4. Train the MobileNet Model
We use the MobileNet model without the top layers, adding custom classification layers for CIFAR-10. The model is trained for 5 epochs.

python
Copy code
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten

# Load the base MobileNet model
base_model = MobileNet(input_shape=(32, 32, 3), include_top=False, weights=None)

# Add custom layers for CIFAR-10 classification
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(10, activation='softmax')(x)

# Build the model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
5. Convert the Model to TensorFlow Lite
Convert the trained model into a TensorFlow Lite model for deployment on edge devices.

python
Copy code
# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('mobilenet_cifar10.tflite', 'wb') as f:
    f.write(tflite_model)
6. Simulate Edge Inference with TensorFlow Lite
Use TensorFlow Lite Interpreter to simulate inference on an edge device. This step demonstrates how TensorFlow Lite can be used for real-time image classification on edge devices.

python
Copy code
# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="mobilenet_cifar10.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare an image for inference
input_data = np.expand_dims(x_test[0], axis=0).astype(np.float32)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Predicted class:", np.argmax(output_data))
7. Federated Learning (Optional)
Federated Learning allows multiple devices to collaboratively train a model without sharing their data. Here, we simulate a simple federated learning setup for the MobileNet model.

python
Copy code
import tensorflow_federated as tff

# Define model function for federated learning
def model_fn():
    model = MobileNet(input_shape=(32, 32, 3), include_top=False, weights=None)
    x = Flatten()(model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return tff.learning.from_keras_model(model, 
                                         input_spec=(tf.TensorSpec(shape=[None, 32, 32, 3], dtype=tf.float32),
                                                     tf.TensorSpec(shape=[None, 1], dtype=tf.int64)))

# Create federated dataset
federated_data = [tf.data.Dataset.from_tensor_slices((x_train[i:i+100], y_train[i:i+100])) for i in range(0, len(x_train), 100)]

# Create federated learning process
iterative_process = tff.learning.build_federated_averaging_process(model_fn)
state = iterative_process.initialize()

# Simulate federated training across devices
for round_num in range(5):
    state, metrics = iterative_process.next(state, federated_data)
    print(f"Round {round_num}, Metrics: {metrics}")
8. Measure Latency and Validate Edge Performance
Measure the inference latency of the TensorFlow Lite model to validate how well it performs on edge devices.

python
Copy code
import time

# Measure the inference time
start_time = time.time()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
end_time = time.time()

# Calculate the latency
latency = end_time - start_time
print(f"Inference latency: {latency:.4f} seconds")
9. Visualizing the Results
Plot training and validation accuracy, and loss during training:

python
Copy code
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
Conclusion
This project showcases the deployment of a MobileNet-based model for image classification on edge devices using TensorFlow Lite. The integration of federated learning allows for decentralized training across multiple devices, ensuring that data privacy is maintained. Finally, the project measures the latency of the edge device inference to evaluate real-world performance.

Folder Structure
graphql
Copy code
/Edge-Computing-MobileNet
│
├── mobilenet_cifar10.tflite         # The TensorFlow Lite model file
├── README.md                       # This file
├── requirements.txt                # Python dependencies
└── training_script.py              # Python script for training the model
