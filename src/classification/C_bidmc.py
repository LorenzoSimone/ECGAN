import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, LSTM, Dense, Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dropout

"""
Two types of neural network-based arrhythmia classifiers

Functions:
- build_CNN(input_shape): Constructs a Convolutional Neural Network (CNN) model.
- build_LSTM(input_shape): Constructs a Long Short-Term Memory (LSTM) network model.

Usage:
1. Specify the input shape of your data.
2. Call the respective function to create the desired model.
3. Compile and train the model as per dataset.
"""

def build_CNN(input_shape):
    """
    Constructs a Convolutional Neural Network (CNN) model.

    Parameters:
    - input_shape: Tuple specifying the shape of the input data (e.g., (timesteps, features)).

    Returns:
    - model: A compiled Keras Model object.
    """
    
    # Define the CNN model structure
    input_layer = Input(input_shape)  # Input layer
    
    # First convolutional block
    conv1 = Conv1D(filters=64, kernel_size=6, activation='relu', padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)  # Normalize activations for faster convergence
    conv1 = Dropout(0.1)(conv1)  # Dropout for regularization
    conv1 = MaxPooling1D(pool_size=3, strides=2, padding='same')(conv1)  # Pooling to reduce dimensions

    # Fully connected layers for classification
    mlp = Flatten()(conv1)
    mlp = Dense(64, activation='relu')(mlp)
    mlp = Dense(64, activation='relu', name='hidden_repr')(mlp)  # Hidden representation layer
    out = Dense(1, activation='sigmoid')(mlp)  # Output layer for binary classification

    # Create and return the model
    return keras.models.Model(inputs=input_layer, outputs=out)

def build_LSTM(input_shape):
    """
    Constructs a Long Short-Term Memory (LSTM) network model.

    Parameters:
    - input_shape: Tuple specifying the shape of the input data (e.g., (timesteps, features)).

    Returns:
    - model: A compiled Keras Model object.
    """

    # Define the LSTM model structure
    input_layer = Input(input_shape)  # Input layer
    
    # Stacked LSTM layers
    lstm = LSTM(124, return_sequences=True)(input_layer)  # First LSTM layer
    lstm = LSTM(124, return_sequences=False)(lstm)  # Second LSTM layer

    # Fully connected layers for classification
    mlp = Flatten()(lstm)
    mlp = Dense(128, activation='tanh', name='hidden_repr')(mlp)  # Hidden representation layer
    mlp = Dense(64, activation='tanh')(mlp)
    mlp = Dense(32, activation='tanh')(mlp)
    out = Dense(1, activation='sigmoid')(mlp)  # Output layer for binary classification

    # Create and return the model
    return keras.models.Model(inputs=input_layer, outputs=out)