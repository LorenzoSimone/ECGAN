import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, LSTM, Dense, Conv1D, BatchNormalization, Flatten
"""
This script defines neural network models for arrhythmia classification using the PTB dataset.

Functions:
- build_CNN: Builds a Convolutional Neural Network (CNN) for time-series data.
- build_LSTM: Builds a Long Short-Term Memory (LSTM) model for sequential data.

Models return binary predictions for arrhythmia classification.
"""

def build_CNN(input_shape):
    """
    Builds a Convolutional Neural Network (CNN) model.

    Arguments:
    input_shape -- tuple, the shape of the input data (e.g., (sequence_length, num_features)).

    Returns:
    A Keras Model object representing the CNN.
    """

    # Input layer
    input_layer = Input(input_shape)

    # First convolutional block
    conv1 = Conv1D(filters=64, kernel_size=6, activation='relu', padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)  # Normalize the activations to stabilize training

    # Flatten and Fully Connected Layers
    mlp = Flatten()(conv1)
    mlp = Dense(64, activation='relu')(mlp)  # First dense layer
    mlp = Dense(64, activation='relu', name='hidden_repr')(mlp)  # Representation layer

    # Output layer
    out = Dense(1, activation='sigmoid')(mlp)  # Binary classification output

    return keras.models.Model(inputs=input_layer, outputs=out)

def build_LSTM(input_shape):
    """
    Builds a Long Short-Term Memory (LSTM) model.

    Arguments:
    input_shape -- tuple, the shape of the input data (e.g., (sequence_length, num_features)).

    Returns:
    A Keras Model object representing the LSTM.
    """

    # Input layer
    input_layer = Input(input_shape)

    # LSTM layers
    lstm = LSTM(124, return_sequences=True)(input_layer)  # First LSTM layer with sequences returned
    lstm = LSTM(124, return_sequences=False)(lstm)  # Second LSTM layer without sequences returned

    # Flatten and Fully Connected Layers
    mlp = Flatten()(lstm)
    mlp = Dense(128, activation='tanh', name='hidden_repr')(mlp)  # Representation layer
    mlp = Dense(64, activation='tanh')(mlp)  # Second dense layer
    mlp = Dense(32, activation='tanh')(mlp)  # Third dense layer

    # Output layer
    out = Dense(1, activation='sigmoid')(mlp)  # Binary classification output

    return keras.models.Model(inputs=input_layer, outputs=out)