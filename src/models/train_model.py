# -*- coding: utf-8 -*-
from src.features.select_features import select_features
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_model(selected_features_list_corrected, df):

    # List of selected features
    selected_features_list_corrected = [
        'Adjusted_Revenue', 'Adjusted_Budget', 'Weighted_Rating_Adjusted_Revenue_Mean',
        'Weighted_Rating_Adjusted_Budget_Mean', 'Weighted_Rating_Adjusted_Revenue_Median',
        'Weighted_Rating_Adjusted_Budget_Median', 'Weighted_Rating_Actors', 'Weighted_Director',
        'Weighted_Rating_Country', 'Average_Weighted_Rating', 'Weighted_Rating',
        'Weighted_Rating_Companies', 'Average_Adjusted_Ratings'
    ]

    # Separate the characteristics and the target variable
    X = df[selected_features_list_corrected]
    y = df['gross_adjusted']

    # Apply a logarithmic transformation to the target variable to reduce the bias
    y = np.log1p(y)

    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and test sets
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

    return model

if __name__ == '__main__':
    #Extract data from csv file
    movies_df = pd.read_csv("data/interim/gross_built_features_dataset.csv")
    selected_features = select_features(movies_df)
    trained_model = train_model(selected_features, movies_df)
    trained_model.save('models/gross_prediction_model.keras')


