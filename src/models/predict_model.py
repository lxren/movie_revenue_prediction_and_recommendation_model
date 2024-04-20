# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime
import os
from keras.models import load_model
from src.features.build_features import adapt_features_for_predictions
from src.features.select_features import select_features
import joblib


def predict_model(movies_to_predict, model_path='models/gross_prediction_model.keras', scaler_path='models/scaler.gz'):
    # Load pretrained model
    if not os.path.exists(model_path):
        print('There is no model to predict with. You need to train a model first.')
        return

    model = load_model(model_path)

    # Check if the scaler exists
    if not os.path.exists(scaler_path):
        print('Scaler not found. Please ensure the scaler used during training is available.')
        return

    scaler = joblib.load(scaler_path)  # Load the scaler

    # Assume adapt_features_for_predictions and select_features functions are available and properly defined
    movies_df = adapt_features_for_predictions(movies_to_predict)
    selected_features = select_features()

    # Extract the relevant features and scale them
    X = movies_df[selected_features]
    X_scaled = scaler.transform(X)  # Use the loaded scaler

    # Make predictions
    predictions = model.predict(X_scaled)
    
    return predictions

    

def gather_movie_data():
    columns = [
        'Title', 'Budget', 'Director', 'Lead Actor', 'Production Company',
        'Country of production', 'Genres', 'Release Date (dd/mm/yyyy)', 'Rating', 'Runtime'
    ]
    data = []

    while True:
        print('\nEntering new movie to predict gross...')
        print("Enter movie details or type 'stop' to finish:")
        movie = {}
        for column in columns:
            while True:
                value = input(f"Enter {column}: ")
                if column == 'Rating':
                    try:
                        # Convert to float and check if in range
                        value = float(value)
                        if 0 <= value <= 10:
                            break
                        else:
                            print("Rating must be between 0 and 10. Please try again.")
                    except ValueError:
                        print("Invalid input. Please enter a numeric value for rating.")
                elif column == 'Release Date (dd/mm/yyyy)':
                    try:
                        # Try to convert the input to a date and extract the year
                        date_value = datetime.strptime(value, '%d/%m/%Y')
                        movie['Release Year'] = date_value.year  # Extract and store the year
                        break
                    except ValueError:
                        print("Invalid date. Please enter the date in format dd/mm/yyyy.")
                elif column == 'Budget':
                    try:
                        # Ensure budget is a numeric value
                        value = float(value)
                        break
                    except ValueError:
                        print("Invalid input. Please enter a numeric value for the budget.")
                elif column == 'Runtime':
                    try:
                        # Ensure runtime is a numeric integer value
                        value = int(value)
                        if value > 0:
                            break
                        else:
                            print("Runtime must be a positive integer. Please try again.")
                    except ValueError:
                        print("Invalid input. Please enter an integer value for runtime.")
                elif value.lower() in ['stop', 'quit', 'exit', 'continue']:
                    return pd.DataFrame(data)
                else:
                    break
            movie[column] = value
        data.append(movie)

    return pd.DataFrame(data)


if __name__ == '__main__':

    # Ask user for movies input to predict gross
    prediction_movies = gather_movie_data()
    print(f'Movies to predict gross:\n {prediction_movies}\n')
    predictions = predict_model(prediction_movies)
    print(predictions)


