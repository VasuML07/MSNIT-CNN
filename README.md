🧠 MSNIT-CNN
Regularized Convolutional Neural Network for MNIST Digit Classification
📌 Overview

MSNIT-CNN is a structured and regularized Convolutional Neural Network built using TensorFlow and Keras to classify handwritten digits (0–9) from the MNIST dataset.

This project emphasizes:

Clean CNN architecture

Overfitting prevention

Stable optimization

Practical inference pipeline

It demonstrates disciplined deep learning engineering rather than brute-force experimentation.

⚙️ What It Does

Loads MNIST dataset

Normalizes pixel values (0–255 → 0–1)

Builds a regularized CNN model

Uses EarlyStopping to prevent overfitting

Evaluates performance

Saves trained model

Predicts custom handwritten digit images

🧠 Model Architecture
🔹 Input Layer

28 × 28 grayscale image

1 channel

🔹 Convolution Block 1

Conv2D (32 filters, 3×3, padding="same")

L2 Regularization (0.0005)

Batch Normalization

ReLU Activation

MaxPooling (2×2)

Dropout (25%)

🔹 Convolution Block 2

Conv2D (64 filters, 3×3, padding="same")

L2 Regularization (0.0005)

Batch Normalization

ReLU Activation

MaxPooling (2×2)

Dropout (25%)

🔹 Dense Block

Flatten

Dense (256 neurons)

L2 Regularization (0.0005)

Batch Normalization

ReLU Activation

Dropout (50%)

🔹 Output Layer

Dense (10 units → digits 0–9)

Logits output

Loss: SparseCategoricalCrossentropy (from_logits=True)

🚀 Training Configuration

Optimizer: Adam (learning_rate = 0.001)

Loss Function: SparseCategoricalCrossentropy

Metrics: Accuracy

EarlyStopping:

Monitor: val_loss

Patience: 3

Restore Best Weights: True

Regularization techniques improve generalization and reduce overfitting.

📊 Expected Performance

With sufficient training epochs:

~98–99% accuracy on MNIST test dataset

(Default epochs can be increased for better results.)

🛠 Tech Stack
💻 Core Language

🧠 Deep Learning




📊 Numerical Computing

🖼 Image Processing

📁 Project Structure
MSNIT-CNN/
│
├── train.py
├── predict.py
├── digit.png
├── cnn_model_regularized.keras
├── requirements.txt
└── README.md
⚙️ Run Locally
📥 Clone Repository
git clone https://github.com/VasuML07/MSNIT-CNN.git
cd MSNIT-CNN
📦 Install Dependencies
pip install -r requirements.txt
▶ Train Model
python train.py

This will:

Train the CNN

Evaluate on test dataset

Save model as cnn_model_regularized.keras

🔍 Predict Custom Digit

Place your image as:

digit.png

Run:

python predict.py

Output:

Predicted digit: X
🖼 Image Requirements for Prediction

Grayscale image

Auto-resized to 28×28

White digit on black background

Clear handwritten digit

The script automatically:

Converts to grayscale

Resizes

Inverts colors

Normalizes

Reshapes for inference

⚠ Limitations

MNIST is a simple benchmark dataset

Default training epochs are limited

No data augmentation

Performance may decrease on noisy real-world digits

🎯 What This Project Demonstrates

CNN architecture design

Regularization in practice

Overfitting prevention

Model saving and loading
