# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime
import os
from keras.models import load_model
from src.features.build_features import adapt_features_for_predictions
import src.features.build_features as bf
from src.features.select_features import select_features
import joblib
import numpy as np
import sys
import pdb

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
    predictions = np.expm1(predictions)
    return predictions

    

def display_options(options, category_name):
    print(f"\nAvailable {category_name}:")
    for idx, option in enumerate(options, 1):
        print(f"{idx}. {option}")
    return options

def get_user_selection(options, input_prompt, allow_multiple = False):
    selected_options = []

    while True:
        user_input = input(input_prompt)
        if (user_input.lower() == 'done') and (allow_multiple):
            if not selected_options:
                print("At least one genre must be selected. Please select at least one genre.")
                continue
            break
        elif user_input.isdigit() and 1 <= int(user_input) <= len(options):
            choice = options[int(user_input) - 1]
            if choice not in selected_options:
                selected_options.append(choice)
                print(f"Added: {choice}")
                if not allow_multiple:
                    break
            else:
                print(f"{choice} is already selected.")
        elif user_input in options:
            if user_input not in selected_options:
                selected_options.append(user_input)
                print(f"Added: {user_input}")
                if not allow_multiple:
                    break
            else:
                print(f"{user_input} is already selected.")
        else:
            print("Invalid option, please try again.")
        
    return ', '.join(selected_options)

def gather_movie_data():
    genre_list = sorted(bf.get_genre_list()) + ['Other Genre']
    top_producers = sorted(list(bf.top_producers.keys())) + ['Other Production Company']
    top_countries = sorted(bf.top_countries) + ['Other Country']
    top_stars = sorted(bf.top_stars) + ['Other Actor']
    top_directors = sorted(bf.top_directors) + ['Other Director']

    categories = {
        'Genres' : genre_list, 
        'Director' : top_directors, 
        'Lead Actor' : top_stars, 
        'Production Company' : top_producers, 
        'Country of production' : top_countries
    }

    columns = [
        'Title', 'Budget', 'Director', 'Lead Actor', 'Production Company',
        'Country of production', 'Genres', 'Release Date (dd/mm/yyyy)', 'Rating', 'Runtime'
    ]
    data = []
    

    while True:
        print('\nEntering new movie to predict gross...')
        print("\nEnter movie details or type 'done or 'd' to finish entering movies. \
              You can cancel the process by writing 'quit' or 'q':\n")
        movie = {}
        for column in columns:
            if column == 'Genres':
                print(f"\n\nSelect Genres (you can choose multiple, but one at a time, write 'done' when you finish):")
                options =  display_options(categories[column], column)
                movie[column] = get_user_selection(categories[column], f"Enter {column} (or number): ", allow_multiple=True)
            elif column in categories.keys():
                # Using the display and selection logic for specific columns
                options =  display_options(categories[column], column)
                movie[column] = get_user_selection(options, f"Enter {column} (or number): ")

            else:
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
                            date_value.year  # Extract year to verify acceptance
                            movie[column] = date_value
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
                    elif value.lower() in ['done', 'd']:
                        return pd.DataFrame(data)
                    elif value.lower() in ['quit', 'q']:
                        print("Canceling process of Gross Prediction ...\n")
                        sys.exit(0)
                    else:
                        break
                
                movie[column] = value

        data.append(movie)

    return pd.DataFrame(data)



if __name__ == '__main__':

    # Ask user for movies input to predict gross
    prediction_movies = gather_movie_data()
    print(f'\n\nMovies to predict gross:\n {prediction_movies}\n\n')
    predictions = predict_model(prediction_movies)
    predictions_df = pd.DataFrame()
    predictions_df['predicted gross'] = pd.DataFrame(predictions)
    predictions_df = predictions_df['predicted gross'].apply(lambda x: "${:,.2f}".format(x))
    predictions_df = pd.concat([prediction_movies['title'], predictions_df], axis=1)
    print(f'\nCalculated Gross Predictions for each input film are: \n\n {predictions_df}')


