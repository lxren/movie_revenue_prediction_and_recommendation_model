# -*- coding: utf-8 -*-
from src.features.select_features import select_features
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def create_and_fit_scaler(X):
    """
    Creates a scaler, fits it on the provided data, and returns the fitted scaler.

    Args:
        X (pd.DataFrame): Data to fit the scaler on.

    Returns:
        StandardScaler: The fitted scaler.
    """
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

def transform_data_with_scaler(X, scaler):
    """
    Transforms data using the provided fitted scaler.

    Args:
        X (pd.DataFrame): Data to transform.
        scaler (StandardScaler): A previously fitted scaler.

    Returns:
        np.array: The scaled data.
    """
    X_scaled = scaler.transform(X)
    return X_scaled


def train_model(selected_features, df, scaler=None):
    """
    Train a neural network model on provided data.

    Args:
        selected_features (list): List of features to use.
        df (pd.DataFrame): The DataFrame containing the training data.
        scaler (StandardScaler): Optional. A previously fitted scaler. If not provided, a new scaler will be created and fitted.

    Returns:
        tf.keras.Model: The trained neural network model.
    """
    X = df[selected_features]
    y = np.log1p(df['gross_adjusted'])  # Apply logarithmic transformation

    if not scaler:
        scaler = create_and_fit_scaler(X)  # Fit a new scaler if not provided

    X_scaled = transform_data_with_scaler(X, scaler)  # Scale the data

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Create the neural network model with improved regularization and dropout.
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=X_train.shape[1], kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model with an adjusted learning rate in the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Use Early Stopping to avoid over-setting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Training the model with more times and a tight batch size
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    
    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test)
    print("Root mean square error in the test set:", test_loss)

    return model, scaler 

if __name__ == '__main__':
    movies_df = pd.read_csv("data/interim/gross_built_features_dataset.csv")
    selected_features = select_features(movies_df)
    scaler = create_and_fit_scaler(movies_df[selected_features])  # Create and fit the scaler
    trained_model, scaler = train_model(selected_features, movies_df, scaler)
    joblib.dump(scaler, 'models/scaler.gz')
    trained_model.save('models/gross_prediction_model.keras')


