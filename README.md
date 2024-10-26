# Expression Classification from Facial Images

This project aims to classify facial expressions using deep learning techniques. We utilize the Expression in-the-Wild (ExpW) dataset, which contains 91,793 face images categorized into seven basic expressions: **angry**, **disgust**, **fear**, **happy**, **sad**, **surprise**, and **neutral**. The project includes data preprocessing, model training, testing, and evaluation using convolutional neural networks (CNNs) with Keras and TensorFlow.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training the Model](#training-the-model)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Installation and Setup](#installation-and-setup)
9. [References](#references)

## Project Overview
This project is designed to develop a computer vision system that can classify facial expressions based on image input. The main objectives are:
- **Understanding** deep CNN architectures
- **Working** with real-world image datasets
- **Implementing** data augmentation and resampling techniques to handle imbalanced classes
- **Evaluating** model performance with accuracy, F1 scores, and confusion matrices

## Dataset
The **Expression in-the-Wild (ExpW)** dataset is used, containing:
- **91,793** face images
- **7** expression categories: angry, disgust, fear, happy, sad, surprise, and neutral

## Data Preprocessing
1. **Loading Dataset**: Images and labels are loaded from the ExpW dataset.
2. **Face Detection**: Using bounding box coordinates to crop faces from images.
3. **Image Resizing**: Resizing each face to **64x64 pixels**.
4. **Class Imbalance Handling**: Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance classes.
5. **Data Normalization**: Scaling pixel values between 0 and 1.
6. **One-Hot Encoding**: Converting expression labels into a categorical format.

## Model Architecture
The CNN architecture used includes:
- **Input Layer**: Image shape (64, 64, 3)
- **Three Convolutional Blocks**: Each with Conv2D, BatchNormalization, MaxPooling, and Dropout layers
- **Fully Connected Layers**: Flattening layer followed by Dense layers with dropout for regularization
- **Output Layer**: Dense layer with 7 units (one per expression category) and softmax activation

## Training the Model
- **Data Augmentation**: Applied random transformations (rotation, shifting, shearing, zooming, and flipping) using `ImageDataGenerator`.
- **Training Configuration**:
  - **Optimizer**: Adam
  - **Loss Function**: Categorical Cross-Entropy
  - **Metrics**: Accuracy
  - **Epochs**: 50
  - **Batch Size**: 32
- **Train-Test Split**: 70% training, 15% validation, 15% test

## Evaluation
1. **Accuracy and Loss Curves**: Plotted to assess training and validation performance.
2. **Confusion Matrix**: Visualized for detailed analysis of predictions per class.
3. **Classification Report**: Detailed metrics for precision, recall, and F1-score.
4. **Top Predictions**: Displayed a subset of test images with their predicted labels.

## Results
- **Train Accuracy**: Achieved high accuracy on the training data.
- **Validation Accuracy**: Maintained competitive performance on validation data.
- **Test Accuracy**: Detailed results on test set performance.
- **F1 Score**: High F1 score indicating balanced precision and recall across classes.

## Installation and Setup
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/expression-classification.git
    ```
2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Dataset Download**: Download the ExpW dataset and adjust paths as needed.
4. **Run the Code**:
    - Ensure paths for dataset (`images_path` and `label_path`) are correctly set.
    - Train the model using:
      ```python
      python train_model.py
      ```
5. **Visualize Results**:
    - Generate evaluation metrics and visualizations by running:
      ```python
      python evaluate_model.py
      ```

## References
- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- **ExpW Dataset**: For details on the dataset, refer to the original publication.

## Acknowledgments
Special thanks to the researchers who curated the ExpW dataset and contributors to Keras and TensorFlow for making deep learning accessible.

---

**Note**: Update `requirements.txt` with the exact dependencies used in your project.
