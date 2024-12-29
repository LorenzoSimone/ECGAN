import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Dense, Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dropout

"""
This script defines Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) models 
for arrhythmia classification using the MIT-BIH dataset.

Functions:
- build_CNN: Constructs a CNN for time-series arrhythmia classification.
- build_LSTM: Constructs an LSTM model for sequential arrhythmia classification.

Both models are designed for binary classification (e.g., normal vs. abnormal heartbeat).
"""

def build_CNN(input_shape):
    """
    Constructs a Convolutional Neural Network (CNN) for time-series data classification.

    Arguments:
    input_shape -- tuple, defining the shape of the input data 
                   (e.g., (sequence_length, num_features)).

    Returns:
    A Keras Model object representing the CNN.
    """
    # Define the input layer
    input_layer = Input(input_shape)

    # Convolutional block
    conv1 = Conv1D(filters=64, kernel_size=6, activation='relu', padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)  # Normalize feature maps for training stability
    conv1 = Dropout(0.1)(conv1)  # Apply dropout to reduce overfitting
    conv1 = MaxPooling1D(pool_size=3, strides=2, padding='same')(conv1)  # Downsample feature maps

    # Fully connected block
    mlp = Flatten()(conv1)  # Flatten the feature maps for dense layers
    mlp = Dense(64, activation='relu')(mlp)  # Dense layer for feature extraction
    mlp = Dense(64, activation='relu', name='hidden_repr')(mlp)  # Dense representation layer

    # Output layer
    out = Dense(1, activation='sigmoid')(mlp)  # Output for binary classification

    # Build and return the model
    return keras.models.Model(inputs=input_layer, outputs=out)

def build_LSTM(input_shape):
    """
    Constructs a Long Short-Term Memory (LSTM) model for sequential data classification.

    Arguments:
    input_shape -- tuple, defining the shape of the input data 
                   (e.g., (sequence_length, num_features)).

    Returns:
    A Keras Model object representing the LSTM.
    """
    # Define the input layer
    input_layer = Input(input_shape)

    # LSTM layers
    lstm = LSTM(124, return_sequences=True)(input_layer)  # First LSTM layer returning sequences
    lstm = LSTM(124, return_sequences=False)(lstm)  # Second LSTM layer outputting a single vector

    # Fully connected block
    mlp = Flatten()(lstm)  # Flatten the LSTM outputs for dense layers
    mlp = Dense(128, activation='tanh', name='hidden_repr')(mlp)  # Dense representation layer
    mlp = Dense(64, activation='tanh')(mlp)  # Second dense layer
    mlp = Dense(32, activation='tanh')(mlp)  # Third dense layer

    # Output layer
    out = Dense(1, activation='sigmoid')(mlp)  # Output for binary classification

    # Build and return the model
    return keras.models.Model(inputs=input_layer, outputs=out)

# Example usage with shape information (uncomment to execute)
# cnn_model = build_CNN((sequence_length, num_features))
# cnn_model.summary()  # Display model architecture
