# -*- coding: utf-8 -*-
import ast
import cpi
import pandas as pd
from pathlib import Path
from keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os
from tensorflow import keras
import pdb

def standardize_data(movies_df):
    """
    Standardizes the DataFrame to have the necessary columns for processing.

    Args:
        movies_df (pd.DataFrame): The DataFrame to standardize.

    Returns:
        pd.DataFrame: The standardized DataFrame.
    """
    # Define the standard column names expected by build_features
    standard_columns = {'Title': 'title', 
                        'Budget': 'budget', 
                        'Director': 'director',
                        'Lead Actor': 'star', 
                        'Production Company': 'production_companies',
                        'Country of production': 'production_countries', 
                        'Genres': 'genre',
                        'Release Year': 'year', 
                        'Release Date (dd/mm/yyyy)': 'release_date',
                        'Rating': 'rating',
                        'Runtime': 'runtime'}

    # Rename columns to match standard column names
    rename_dict = {col: standard_columns[col] for col in standard_columns if col in movies_df.columns}
    movies_df.rename(columns=rename_dict, inplace=True)

    # If 'Release Year' is not in the DataFrame, extract it from 'Release Date'
    if 'year' not in movies_df.columns and 'release_date' in movies_df.columns:
        movies_df['year'] = pd.to_datetime(movies_df['release_date'], format='%d/%m/%Y').dt.year

    return movies_df

def adjust_data_format(movies_df):
    # Convert columns to the right data types
    columns_to_int = ['year', 'runtime', 'gross', 'budget', 'vote_count_letterboxd', 'vote_count_imdb']
    for col in columns_to_int:
        try:
            movies_df[col] = movies_df[col].astype(int)
        except:
            print('Could not validate')
    movies_df['release_date'] = pd.to_datetime(movies_df['release_date'])

    # Merge genre columns and eliminate duplicated genres
    if 'genre' not in movies_df.columns:
        movies_df['genre'] = movies_df['genre_letterboxd'].astype(str).str.replace(r"[\"\[\]]", "", regex=True) + ', ' + \
                            movies_df['genre_tmdb'] + ', ' + movies_df['genre_imdb']
        movies_df['genre'] = movies_df['genre'].str.lower().str.replace(' ', '').apply(lambda x: ', '.join(list(set(x.split(',')))))
        movies_df.drop(columns=['genre_letterboxd', 'genre_imdb', 'genre_tmdb'], inplace=True)

    # Extract only the first production country
    movies_df['production_countries'] = movies_df['production_countries'].apply(ast.literal_eval)
    max_countries = movies_df['production_countries'].apply(len).max()
    for i in range(max_countries):
        col_name = f'country_{i+1}'
        movies_df[col_name] = movies_df['production_countries'].apply(lambda x: x[i] if i < len(x) else None)
    movies_df.drop(columns=[f'country_{i}' for i in range(2, max_countries+1)], inplace=True)
    movies_df.rename(columns={'country_1': 'production_country'}, inplace=True)
    movies_df.drop('production_countries', axis=1, inplace=True)


    # Clean and deduplicate production companies names
    movies_df['production_companies'] = movies_df['production_companies'].str.lower().apply(lambda x: ', '.join(list(set(x.split(', ')))))

    # Classify the season a movie was released
    movies_df['season'] = movies_df['release_date'].dt.month.apply(get_season)
    movies_df.drop('release_date', axis=1, inplace=True)

    return movies_df

def adjust_values(movies_df):
    # Adjust money values according to inflation through years
    if 'gross' in movies_df.columns:
        movies_df['gross_adjusted'] = movies_df.apply(lambda x: adjust_for_inflation(x['gross'], x['year']), axis=1)
        movies_df['budget_adjusted'] = movies_df.apply(lambda x: adjust_for_inflation(x['budget'], x['year']), axis=1)
        movies_df.drop(columns=['budget', 'gross'], inplace=True)
    else: 
        movies_df['budget_adjusted'] = movies_df.apply(lambda x: adjust_for_inflation(x['budget'], x['year']), axis=1)
        movies_df.drop(columns=['budget'], inplace=True)

    # Merge 'rating_letterboxd' and 'rating_imdb' columns by calculating a weighted average
    movies_df['Weighted_Rating'] = (
        ((movies_df['rating_letterboxd'] * movies_df['vote_count_letterboxd']) +
         (movies_df['rating_imdb'] * movies_df['vote_count_imdb'])) /
        (movies_df['vote_count_letterboxd'] + movies_df['vote_count_imdb'])
    )
    movies_df.drop(columns=['rating_letterboxd', 'vote_count_letterboxd', 'rating_imdb', 'vote_count_imdb'], inplace=True)

    # One-hot-encode genres
    # Generate one-hot encoding of genres
    one_hot_encoded_genres = movies_df['genre'].str.get_dummies(sep=', ')

    # Treat similar genres as one
    try:
        one_hot_encoded_genres['crime']  = one_hot_encoded_genres['crime'] | one_hot_encoded_genres['film-noir']
        one_hot_encoded_genres['musical']  = one_hot_encoded_genres['music'] | one_hot_encoded_genres['musical']
        one_hot_encoded_genres['sci-fi']  = one_hot_encoded_genres['sci-fi'] | one_hot_encoded_genres['sciencefiction']
        one_hot_encoded_genres.drop(columns=['film-noir', 'music', 'sciencefiction'], inplace=True)
    except:
        print('No similar genres to combine')

    # Integrate one-hot encoding sub set
    one_hot_encoded_genres = one_hot_encoded_genres.add_prefix('genre_')
    movies_df = pd.concat([movies_df.drop(columns=['genre'], axis=1), one_hot_encoded_genres], axis=1)

    return movies_df

def weight_rating_by_director(movies_df, search_director):
    """ Adjusts the weighting of movie ratings based on the relevance of the director."""
    # Select most relevant directors according to gross influence.
    # Info source: https://www.the-numbers.com/box-office-star-records/worldwide/lifetime-specific-technical-role/director
    top_directors = [
        "Steven Spielberg", "James Cameron", "Anthony Russo", "Joe Russo", "Peter Jackson", "Michael Bay",
        "David Yates", "Christopher Nolan", "J.J. Abrams", "Ridley Scott", "Tim Burton", "Robert Zemeckis",
        "Jon Favreau", "Ron Howard", "Sam Raimi", "James Wan"
    ]

    instances = 0
    irrelevant_directors = []
    not_found_directors = []
    for director in top_directors:
        instances += search_director(movies_df, director, {'year': 'year', 'director': 'director'},
                                     irrelevant_directors, not_found_directors)

    # Counting Anthony Russo and Joe Russo as one: Russo Brothers, as they are together in all the instances
    top_directors[top_directors.index('Anthony Russo')] = 'Russo Brothers'
    top_directors.remove('Joe Russo')
    movies_df['director'] = movies_df['director'].replace('Anthony Russo,Joe Russo', 'Russo Brothers')

    # Remove irrelevant and not found directors from the top_directors list
    for director in irrelevant_directors + not_found_directors:
        if director in top_directors:
            top_directors.remove(director)

    # Apply the weighting factor based on the director's membership in the top list
    movies_df['Weighted_Director'] = movies_df.apply(
        lambda row: row['Weighted_Rating'] * 1.1 if row['director'] in top_directors else row['Weighted_Rating'] * 0.9,
        axis=1
    )
    return movies_df

def weight_rating_by_actor(movies_df, search_star, is_top_star):
    """
    Adjusts movie ratings based on the relevance of actors.
    `search_star` is a function that checks the presence of a star in the movies dataset and returns a count of instances.
    `is_top_star` is a function that checks if a star is among the top stars.

    Args:
    movies_df (DataFrame): The dataframe containing movie data.
    search_star (function): Function to search for stars within the dataset.
    is_top_star (function): Function to check if a star is top-rated.

    Returns:
    DataFrame: The modified DataFrame with adjusted ratings.
    """
    # Select most relevant actors according to gross influence.
    # Info source: https://www.the-numbers.com/box-office-star-records/worldwide/lifetime-acting/top-grossing-leading-stars
    top_stars = [
        "Samuel L. Jackson", "Scarlett Johansson", "Robert Downey Jr.", "Zoe Saldana", "Chris Pratt",
        "Tom Cruise", "Vin Diesel", "Chris Hemsworth", "Bradley Cooper", "Chris Evans", "Tom Hanks",
        "Johnny Depp", "Dwayne Johnson", "Tom Holland", "Mark Ruffalo", "Emma Watson", "Don Cheadle",
        "Dave Bautista", "Jeremy Renner", "Will Smith", "Karen Gillan", "Elizabeth Olsen", "Josh Brolin",
        "Daniel Radcliffe", "Benedict Cumberbatch", "Harrison Ford", "Chadwick Boseman", "Rupert Grint",
        "Letitia Wright", "Leonardo DiCaprio", "Steve Carell", "Sebastian Stan", "Matt Damon", "Danai Gurira",
        "Tom Hiddleston", "Brad Pitt", "Paul Bettany", "Jack Black", "Bruce Willis", "Eddie Murphy",
        "Liam Neeson", "Pom Klementieff", "Benedict Wong", "Sam Worthington", "Ben Stiller", "Hugh Jackman",
        "Jason Statham", "Ian McKellen", "Gwyneth Paltrow", "Jennifer Lawrence", "Mark Wahlberg",
        "Nicolas Cage", "Cameron Diaz", "Ewan McGregor", "Christian Bale"
    ]

    instances = 0
    irrelevant_stars = list()
    not_found_stars = list()
    for star in top_stars:
        instances += search_star(movies_df, star, {'year':'year', 'star':'star'}, irrelevant_stars, not_found_stars)

    # Remove irrelevant and not found stars from the top_stars list
    for star in not_found_stars:
        top_stars.remove(star)
    for star in irrelevant_stars:
        top_stars.remove(star)
    
    # Apply the function to adjust the Weighted_Rating based on the actors.
    movies_df['Weighted_Rating_Actors'] = movies_df.apply(
        lambda row: row['Weighted_Rating'] * 1.1 if is_top_star(row['star'], top_stars) else row['Weighted_Rating'] * 0.9, axis=1
    )
    return movies_df

def weight_rating_by_country(movies_df):
    """
    Adjusts the Weighted_Rating of movies based on the production country's influence on the global film market.

    Parameters:
    movies_df (DataFrame): The DataFrame containing movie data with columns 'Weighted_Rating' and 'production_country'.

    Returns:
    DataFrame: The modified DataFrame with updated 'Weighted_Rating_Country' values.
    """
    top_countries = [
        "United States of America", "United Kingdom", "China", "France", "Japan",
        "Germany", "South Korea", "Canada", "India", "Australia", "Hong Kong",
        "New Zealand", "Italy", "Spain"
    ]
    # Adjust the Weighted_Rating based on the country of production.
    movies_df['Weighted_Rating_Country'] = movies_df.apply(
        lambda row: (row['Weighted_Rating'] * 1.1) if row['production_country'] in top_countries else (row['Weighted_Rating'] * 0.9),
        axis=1
    )
    return movies_df

def weight_ratings_by_producer_relevance(movies_df):
    """
    Adjusts the movie ratings based on the relevance of production companies.

    Args:
    movies_df (DataFrame): The DataFrame containing movie data.

    Returns:
    DataFrame: The modified DataFrame with a new column for weighted ratings by company relevance.
    """
    # Define top producers and their replacements in the dataset
    top_producers = [
        "Warner Bros", "Universal Pictures", "Columbia Pictures", "Marvel Studios",
        "Walt Disney Pictures", "Paramount", "20th Century Fox", "Legendary Pictures",
        "New Line Cinema", "DreamWorks Animation", "Dune Entertainment",
        "Amblin Entertainment", "Disney-Pixar", "Relativity Media",
        "Metro-Goldwyn-Mayer Pictures", "Village Roadshow Productions",
        "DreamWorks Pictures", "Heyday Films", "Regency Enterprises", "Lucasfilm",
        "Walt Disney Animation Studios", "Lionsgate", "TSG Entertainment",
        "RatPac Entertainment", "Illumination Entertainment", "Original Film",
        "Skydance Productions", "Summit Entertainment", "Touchstone Pictures",
        "di Bonaventura Pictures"
    ]

    replacements = {
        'Warner Bros': [
            'Warner Bros. Pictures',
            'Warner Bros. Korea',
            'Warner Bros. Television',
            'Warner Bros-Seven Arts',
            'Warner Bros. Animation',
            'Warner Bros. Family Entertainment',
            'Warner Bros. Pictures Animation',
            'Warner Bros. Entertainment España',
            'Warner Bros. Digital'
        ],
        'Universal Pictures': [
            'Universal Pictures',
            'Universal Pictures do Brasil',
            'Universal Pictures International (UPI)'
        ],
        'Columbia Pictures': [
            'Columbia Pictures',
            'Columbia Pictures Film Production Asia',
            'Columbia Pictures Producciones Mexico'
        ],
        'Paramount': [
            'Paramount',
            'Paramount Famous Lasky Corporation',
            'Paramount Players',
            'Paramount Animation',
            'Paramount Vantage',
            'Paramount Pictures Canada'
        ],
        '20th Century Fox': [
            '20th Century Fox',
            '20th Century Fox Animation',
            '20th Century Fox Home Entertainment',
            '20th Century Fox Argentina'
        ],
        'Dune Entertainment': [
            'Dune Entertainment',
            'Dune Entertainment III'
        ],
        'Metro-Goldwyn-Mayer': [
            'Metro-Goldwyn-Mayer',
            'Metro-Goldwyn Pictures Corporation'
        ],
        'Village Roadshow Pictures': [
            'Village Roadshow Pictures',
            'Village Roadshow Pictures Asia'
        ],
        'Walt Disney': [
            'Walt Disney Pictures',
            'Walt Disney Productions',
            'Walt Disney Animation Studios',
            'Walt Disney Animation',
            'Walt Disney Feature Animation'
        ],
        'Lionsgate': [
            'Lionsgate',
            'Lionsgate Home Entertainment'
        ],
        'Illumination': [
            'Illumination',
            'Illuminations Films'
        ]
    }

    # Replace values for different production companies
    movies_df = replace_values(movies_df, {'producer':'production_companies'}, replacements)

    # Apply the corrections considering that the name of the columns may be slightly different
    updated_top_producer_columns_corrected = [
        'producer_Warner Bros', 'producer_Universal Pictures', 'producer_Columbia Pictures',
        # Additional corrected producer columns not listed for brevity
    ]

    existing_columns = [col for col in updated_top_producer_columns_corrected if col in movies_df.columns]
    movies_df['Weighted_Rating_Companies'] = movies_df.apply(
        lambda row: calculate_corrected_weighted_rating_companies_v2(row, existing_columns), axis=1
    )
    return movies_df

def encode_movie_genres(movies_df, encoder_path='models/genre_encoder.keras'):
    # Identify genre one-hot encoded features
    genre_columns = [column for column in movies_df.columns if "genre_" in column]

    # Separate genre data and drop original genre columns
    genres = movies_df[genre_columns]
    movies_df.drop(genre_columns, axis=1, inplace=True)

    # Load or train genre encoder model
    if os.path.exists(encoder_path):
        encoder = load_model(encoder_path)
    else:
        input_dim = genres.shape[1]
        encoding_dim = 5  # Dimension of the encoded space

        # Define autoencoder architecture
        input_layer = Input(shape=(input_dim,))
        dropout_layer = Dropout(0.6)(input_layer)
        encoded = Dense(encoding_dim, activation='relu', activity_regularizer=l1_l2(l1=0.01, l2=0.01),  name='codes')(dropout_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)
        autoencoder = Model(inputs=input_layer, outputs=decoded)

        # Compile and train the autoencoder
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        autoencoder.fit(genres, genres, epochs=100, batch_size=32, shuffle=True, validation_split=0.2, callbacks=[early_stopping])

        # Evaluate and save the encoder part of the autoencoder
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('codes').output)
        encoder.save(encoder_path)
    
    # Encode genres and add them back to the DataFrame
    genres_encoded = encoder.predict(genres)
    genre_encoded_df = pd.DataFrame(genres_encoded, columns=[f'genre_encoded_{i}' for i in range(genres_encoded.shape[1])])
    movies_df = pd.concat([movies_df, genre_encoded_df], axis=1)

    return movies_df

def weight_ratings_by_budget(movies_df):
    """
    Applies budget-based adjustments to the movie ratings.

    Args:
    movies_df (DataFrame): The DataFrame containing movie data.

    Returns:
    DataFrame: The modified DataFrame with new rating columns adjusted by budget.
    """
    # Calculate median and mean of budget_adjusted
    median_budget = movies_df['budget_adjusted'].median()
    mean_budget = movies_df['budget_adjusted'].mean()

    # Calculate the average weighted rating from various sources
    movies_df['Average_Weighted_Rating'] = movies_df[['Weighted_Director', 'Weighted_Rating_Actors',
                                                      'Weighted_Rating_Country', 'Weighted_Rating_Companies']].mean(axis=1)

    # Apply rating adjustments based on median and mean budget
    movies_df['Weighted_Rating_Adjusted_Budget_Median'] = movies_df.apply(
        lambda x: adjust_rating_by_median_budget(x, median_budget), axis=1)
    movies_df['Weighted_Rating_Adjusted_Budget_Mean'] = movies_df.apply(
        lambda x: adjust_rating_by_mean_budget(x, mean_budget), axis=1)

    # Adjust budget based on ratings
    movies_df['Adjusted_Budget'] = movies_df.apply(adjust_budget_by_ratings, axis=1)
    
    return movies_df






def get_season(month):
    """Return the corresponding season for a given month."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def adjust_for_inflation(amount, year):
    """Adjust a monetary amount from a given year to its 2022-equivalent value using CPI data."""
    try:
        return cpi.inflate(amount, year)
    except:
        return amount
    
def search_director(dataframe:pd.DataFrame, director_name:str, columns_name:dict,  irrelevants:list, not_found:list):
    '''
        This function searches in the DataFrame for instances cointaining the name of the given director.
        
        Inputs:
        dataframe: Pandas dataframe where to search the directors.
        director_name: Name of the director to search for.
        column_name: Dictionary of the names of the columns to consult from the dataset. It must contain the following items:
            {
                'year'      : (name of the year column of the dataframe)
                'director'  : (name of the director column of the dataframe)
            }

        Outputs:
        coincidences.shape[0] : Number of found instances of the provided director name
        irrelevants : list with irrelevants directors (those who haven't published any movie in the past 25 years), as appended item
        not_found : list with not found directors, as appended item
    '''
    irrelevants = list()
    not_found = list()
    
    coincidences = dataframe[dataframe[columns_name['director']].str.contains(director_name, case=True)]
    if not coincidences.empty:

        '''
        Printing recent movies of the director for checking validity
        Sources:
        - https://www.bfi.org.uk/lists/10-times-great-directors-left-really-long-gaps-between-films
        - https://screenrant.com/best-director-comebacks-after-breaks/
        - https://screenrant.com/directors-semi-retired-hiatus-great-movie-comeback/

        Longest period break found: 25 years
        Selected tolerance period: 25 years
        If a director didn't make a movie within this period, they're not relevant anymore
        '''

        relevants = coincidences[coincidences[columns_name['year']] >= (2022 - 25)]
        if relevants.empty:
            irrelevants.append(director_name)
    else:
        not_found.append(director_name)

    return coincidences.shape[0]

def search_star(dataframe:pd.DataFrame, star_name:str, columns_name:dict, irrelevants:list, not_found:list):
    '''
        This function searches in the DataFrame for instances cointaining the name of the given actor.
        
        Inputs:
        dataframe: Pandas dataframe where to search the stars.
        star_name: Name of the actor to search for.
        column_name: Dictionary of the names of the columns to consult from the dataset. It must contain the following items:
            {
                'year'      : (name of the year column of the dataframe)
                'star'  : (name of the star/actor column of the dataframe)
            }

        Outputs:
        coincidences.shape[0] : Number of found instances of the provided star name
        irrelevants : list with irrelevants star (those who haven't published any movie in the past 15 years), as appended item
        not_found : list with not found stars, as appended item
    '''

    coincidences = dataframe[dataframe[columns_name['star']].str.contains(star_name, case=True)]
    if not coincidences.empty:

        '''
        Printing recent movies of the star for checking validity
        Sources:
        - https://stephenfollows.com/how-long-is-the-typical-film-actors-career/#:~:text=The%20average%20career%20length%20was,between%2020%20and%2040%20years.
        - https://www.cbr.com/long-acting-breaks-that-actors-were-able-to-successfully-return-from/
        - https://brightside.me/articles/10-actors-who-returned-to-the-screen-after-a-long-hiatus-809693/

        Longest period break found: 13 years
        Selected tolerance period: 15 years
        If stars didn't appear in a movie within this period, they're not relevant anymore
        '''

        relevants = coincidences[coincidences['year'] >= (2022 - 15)]
        if relevants.empty:
            irrelevants.append(star_name)

    else:
        not_found.append(star_name)

    return coincidences.shape[0]

def replace_values(dataframe:pd.DataFrame, column_name:dict, replacements:dict):
    '''
        This function searches in the DataFrame for instances cointaining the name of the given director.
        
        Inputs:
        dataframe: Pandas dataframe where to search the values to replace.
        column_name: Dictionary with the names of the columns to consult from the dataset. It must contain the following items:
            {
                'producer'      : (name of the production company column of the dataframe)
            }
        replacements: Dictionary with the current_values to search for to replace with the new value. Structure must be:
            {
                (new value) : [(List with current possible values to search for in the dataframe)],
                .
                .
                .
            }

        Outputs:
        coincidences.shape[0] : Number of found instances of the provided director name
        irrelevants : list with irrelevants directors (those who haven't published any movie in the past 25 years), as appended item
        not_found : list with not found directors, as appended item
    '''
    for new_value, current_values in replacements.items():
        dataframe[column_name['producer']] = dataframe[column_name['producer']].replace(current_values, new_value)
    return dataframe

def is_top_star(stars, top_stars_list:list):
    # Function to determine if any actor in the film is in the top list
    star_list = [star.strip() for star in stars.split(",")]
    return any(star in top_stars_list for star in star_list)

def calculate_corrected_weighted_rating_companies_v2(row,existing_columns):
    ''' Function for calculating the Weighted_Rating_Companies based on the presence of top producers'''
    # Check if any top producers are present
    if any(row[col] == 1 for col in existing_columns):
        return row['Weighted_Rating'] * 1.1  # Adjust the rating if the production company is a top producer
    else:
        return row['Weighted_Rating'] * 0.9  # Adjust the rating if there are no top producers

# Function to adjust the Weighted_Rating based on the median of budget_adjusted
def adjust_rating_by_median_budget(row, median_budget):
    base_rating = row['Average_Weighted_Rating']  # Usar el promedio de ratings ajustados como base
    if row['budget_adjusted'] >= median_budget:
        return base_rating * 1.1
    else:
        return base_rating * 0.95

# Function to adjust the Weighted_Rating based on the average of budget_adjusted
def adjust_rating_by_mean_budget(row, mean_budget):
    base_rating = row['Average_Weighted_Rating']  # Use the average of adjusted ratings as a basis
    if row['budget_adjusted'] >= mean_budget:
        return base_rating * 1.1
    else:
        return base_rating * 0.95

# Function to adjust the budget_adjusted based on the Average_Weighted_Rating
def adjust_budget_by_ratings(row):
    if row['Average_Weighted_Rating'] >= 6:
        return row['budget_adjusted'] * 1.1  # Increase the budget if the rating is high
    elif row['Average_Weighted_Rating'] >= 5:
        return row['budget_adjusted'] * 1.05  # Slightly increase if moderately high
    else:
        return row['budget_adjusted'] * 0.95  # Decrease if low


def build_features(movies_df):
    """
    Enhance and clean a DataFrame of movie data to prepare for further analysis.

    Args:
        movies_df (pandas.DataFrame): The DataFrame containing the movies data.

    Returns:
        pandas.DataFrame: The cleaned and enhanced DataFrame.
    """
    print("Starting to standardize features names..")
    movies_df = standardize_data(movies_df)
    
    print("Removing initial unnecessary columns...")
    initial_drop_columns = [
        'spoken_languages', 'popularity', 'description_tmdb', 'adult', 
        'Unnamed: 0', 'imdb_id', 'original_language', 'description_letterboxd', 
        'tmdb_id', 'description_imdb', 'director_id', 'star_id', 'revenue'
    ]
    try:
        movies_df.drop(columns=initial_drop_columns, inplace=True, errors='ignore')
    except:
        print('Columns not found to remove')
    print("Adjusting format of dataset columns...")
    movies_df = adjust_data_format(movies_df)

    print("Adjusting monetary values and ratings...")
    movies_df = adjust_values(movies_df)

    print("Applying weighting by director...")
    movies_df = weight_rating_by_director(movies_df, search_director)

    print("Applying weighting by lead actor...")
    movies_df = weight_rating_by_actor(movies_df, search_star, is_top_star)

    print("Applying weighting by country...")
    movies_df = weight_rating_by_country(movies_df)

    print("Applying weighting by producer relevance...")
    movies_df = weight_ratings_by_producer_relevance(movies_df)

    print("Encoding movie genres into numeric values...")
    movies_df = encode_movie_genres(movies_df)

    print("Applying weighting based on budget considerations...")
    movies_df = weight_ratings_by_budget(movies_df)

    print("Removing final unnecessary columns...")
    final_drop_columns = [
        'title', 'year', 'director', 'star', 'runtime', 
        'production_companies', 'production_country', 'season', 'budget_adjusted'
    ]
    try:
        movies_df.drop(columns=final_drop_columns, inplace=True, errors='ignore')
    except:
        print('Columns not found to remove')
    print("Feature building process completed.")

    return movies_df


if __name__ == '__main__':
    #Extract data from CSV file
    movies_df = pd.read_csv("data/interim/cleaned_film_dataset.csv")
    movies_df = build_features(movies_df)

    # Save to interim data
    filepath = Path('data/interim/gross_built_features_dataset.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    movies_df.to_csv(filepath, index=True)