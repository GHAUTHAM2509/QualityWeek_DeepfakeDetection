# Project Name: Face2Face Deepfake Detection using Mesonet

## Overview
This project implements a Convolutional Neural Network (CNN) for binary image classification. The model is designed to distinguish between two categories using TensorFlow and Keras. The training process includes data augmentation, learning rate scheduling, and various callbacks to improve performance and prevent overfitting.

## Features
- **Data Augmentation**: Enhances the dataset using random transformations (rotation, zoom, brightness shifts, horizontal flips, etc.).
- **Customizable CNN Model**: The architecture consists of multiple convolutional layers followed by batch normalization and max pooling.
- **Training Pipeline**: Supports training with validation splits, learning rate decay, early stopping, and checkpoint saving.
- **Evaluation & Prediction**: Provides functionality for evaluating the trained model on test data and generating classification reports.
- **Visualization**: Includes utilities to plot loss curves and visualize activations from convolutional layers.
- **Model Checkpointing**: Saves the best-performing model based on validation accuracy.
- **TensorBoard Integration**: Logs training metrics for visualization.

## Dependencies
Ensure the following libraries are installed:
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Dataset Structure
The dataset should be organized in the following format:
```
/train_data_dir
    /class_1
        image1.jpg
        image2.jpg
    /class_2
        image1.jpg
        image2.jpg
/test_data_dir
    /class_1
        image1.jpg
        image2.jpg
    /class_2
        image1.jpg
        image2.jpg
```

## Usage
### Training the Model
Modify the `train_model` function parameters as needed and run the script:
```python
model = build_model()
history = train_model(
    model,
    train_data_dir='path/to/train_data',
    validation_split=0.2,
    epochs=25,
    batch_size=256,
    lr=1e-3,
    checkpoint=True,
    tensorboard=True
)
```

### Evaluating the Model
After training, evaluate the model using:
```python
evaluate_model(model, 'path/to/test_data', batch_size=64)
```

### Generating Predictions
To predict labels for new images:
```python
data = temp('path/to/test_data', batch_size=64)
predictions = model.predict(data)
print(predictions)
```
Node the file structure should be as follows test_data/images/image1.png.

## Model Architecture
- **Convolutional Layers**: Extracts features using different kernel sizes.
- **Batch Normalization**: Stabilizes training and accelerates convergence.
- **Dropout Layers**: Prevents overfitting by randomly deactivating neurons.
- **Fully Connected Layers**: Combines extracted features and outputs classification results.
- **Sigmoid Activation**: Used for binary classification.

## Performance Monitoring
- **Loss Curve**: Plots training and validation loss after training.
- **TensorBoard Logs**: Stores training logs for visualization.

## Testing on colab
if you would like to test the pre-trained models provided or train your on model and test it out you can do so on colab
- ** https://colab.research.google.com/drive/1T1kYnPkDfwJA5l9C52WdqnLzW1wj64IG?usp=sharing

## Credits & Acknowledgments

This project includes code and concepts from various sources. Below are the references to the original implementations:
- ** https://colab.research.google.com/github/AliaksandrSiarohin/first-order-model/blob/master/demo.ipynb
- ** https://github.com/DariusAf/MesoNet/blob/master/classifiers.py

## License
This project is open-source and available for modification and enhancement.

