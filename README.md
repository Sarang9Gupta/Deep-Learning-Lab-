## Lab 1: Fully Connected Neural Network for MNIST Classification

### Overview 
Implemented a fully connected neural network using *NumPy* to classify handwritten digits from the *MNIST dataset*. The experiment demonstrates:
- *Data Preprocessing*: Normalization, one-hot encoding.
- *Model Training*: Forward and backward propagation.
- *Optimization*: Loss functions, gradient descent.
- *Evaluation*: Accuracy, loss metrics.
- *Data Augmentation*: Random rotation, horizontal flipping.

### Concepts
- *MNIST Dataset*: A benchmark dataset for digit classification.
- *Forward Propagation*: Computes predictions.
- *Backward Propagation*: Updates weights using gradients.
- *Activation Functions*: ReLU, Sigmoid for non-linearity.
- *Loss Function*: Cross-entropy for classification tasks.
- *Optimization*: Stochastic Gradient Descent (SGD).

---

## Lab 2: Neural Networks on Linearly and Non-Linearly Separable Data

### Overview
Trained neural networks on *linearly separable (e.g., line) and non-linearly separable (e.g., Moon, Circle datasets)* using NumPy. The experiment demonstrates:
- *Effect of Hidden Layers*: A single-layer network struggles with complex decision boundaries.
- *Activation Functions: Importance of **ReLU, Sigmoid* in deep learning.
- *Comparative Analysis*: Performance of different architectures on different datasets.

### Concepts
- *Linearly Separable Data*: Can be separated by a straight line.
- *Non-Linearly Separable Data*: Requires non-linear transformations.
- *ReLU (Rectified Linear Unit)*: Solves vanishing gradient problem.
- *Sigmoid*: Used in binary classification.

---

## Lab 3: Convolutional Neural Networks (CNNs) for Image Classification

### Overview
Implemented *CNNs* for classifying images from the *Cats vs. Dogs* and *CIFAR-10* datasets. The experiment includes:
- *CNN Architecture*: Feature extraction using convolutional layers.
- *Activation Functions*: Experimenting with ReLU, Tanh, and Leaky ReLU.
- *Weight Initialization*: Xavier, Kaiming, and Random.
- *Optimizers*: SGD, Adam, RMSprop.
- *Performance Evaluation: Comparing CNNs with a fine-tuned **ResNet-18*.

### Concepts
- *CNN (Convolutional Neural Network)*: Extracts spatial features from images.
- *ResNet-18*: A deep CNN architecture with skip connections.
- *Adam Optimizer*: Adaptive learning rate optimization.
- *Xavier & Kaiming Initialization*: Methods to stabilize gradient flow in deep networks.

---

## How to Run
1. Ensure *NumPy, Matplotlib, and necessary libraries* are installed.
2. Load datasets using standard libraries.
3. Train models using the provided scripts.
4. Evaluate performance on test data.

---

## Results & Observations
- Adding *hidden layers* improves performance on non-linear datasets.
- *ReLU activation* is preferred for deep networks.
- *CNNs outperform fully connected networks* in image classification.
- *Fine-tuned ResNet-18* achieves better accuracy than custom CNNs.

---

## Future Improvements
- Implementing *Batch Normalization* and *Dropout*.
- Experimenting with *Transfer Learning* for small datasets.
- Fine-tuning hyperparameters for better generalization.
"""

## Lab 4:  Poetry Generator using RNN (LSTM)

This project is a simple poetry generator using a Recurrent Neural Network (RNN) with LSTM layers. It trains on a dataset of poems and generates new poetic lines based on learned patterns.

## Features
- Uses LSTM-based neural networks for text generation.
- Trained on a dataset of poems.
- Generates poetry based on a seed input.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/ArpanBareja/Deep-Learning-Lab.git
   cd lab5
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Train the Model
Run the following command to train the model:
```sh
python train.py
```
This will process the dataset, train the LSTM model, and save it for later use.

### Generate Poetry
After training, you can generate poetry by running:
```sh
python generate.py
```
It will prompt you to enter a seed text and generate poetry based on it.

## Dependencies
- TensorFlow
- NumPy
- Pickle

Install all dependencies using:
```sh
pip install -r requirements.txt
```

## Dataset
poems-100 dataset available on Kaggle
