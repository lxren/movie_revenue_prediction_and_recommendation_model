# -*- coding: utf-8 -*-
import ast
import cpi
import pandas as pd
from pathlib import Path

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


def adjust_for_inflation(amount, year):
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

# Function to adjust the Weighted_Rating based on gender, using the Average_Weighted_Rating
def adjust_rating_by_genre(row):
    # adjust more if the average is high, less if it is low.
    if row['Average_Weighted_Rating'] >= 6:
        return row['Weighted_Rating'] * 1.1
    elif row['Average_Weighted_Rating'] >= 5:
        return row['Weighted_Rating'] * 1.05
    else:
        return row['Weighted_Rating'] * 0.95

# Function to adjust the revenue_adjusted based on the Average_Adjusted_Ratings
def adjust_revenue_by_ratings(row):
    # We will adjust the revenue_adjusted more if the average adjusted ratings are high, and less if they are low
    if row['Average_Adjusted_Ratings'] >= 6:
        return row['revenue_adjusted'] * 1.1
    elif row['Average_Adjusted_Ratings'] >= 5:
        return row['revenue_adjusted'] * 1.05
    else:
        return row['revenue_adjusted'] * 0.95

# Function to adjust the Weighted_Rating based on the median and average of adjusted ratings.
def adjust_rating_by_median_revenue(row, median_revenue):
    base_rating = row['Average_Weighted_Rating']  # Use the average of adjusted ratings as a basis
    if row['revenue_adjusted'] >= median_revenue:
        return base_rating * 1.1
    else:
        return base_rating * 0.95

# Function to adjust the Weighted_Rating based on the mean and average of adjusted ratings.
def adjust_rating_by_mean_revenue(row, mean_revenue):
    base_rating = row['Average_Weighted_Rating']  # Use the average of adjusted ratings as a basis
    if row['revenue_adjusted'] >= mean_revenue:
        return base_rating * 1.1
    else:
        return base_rating * 0.95

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

def build_features():
    movies_df = pd.read_csv("data/interim/cleaned_film_dataset.csv")

    # Remove the unnecessary columns for gross prediction
    movies_df.drop(columns=[
        'spoken_languages',
        #'popularity',              # CHECK!!!
        'description_tmdb',
        'adult',
        'Unnamed: 0',
        'imdb_id',
        'original_language',
        'description_letterboxd',
        'tmdb_id',
        'description_imdb',
        'director_id',
        'star_id'],
                inplace=True)

    #------------------------------------------------- FORMAT ADJUSTMENT ---------------------------------------------------#

    # Take columns to the right data type
    movies_df['year']=movies_df['year'].astype(int)
    movies_df['runtime']=movies_df['runtime'].astype(int)
    movies_df['gross']=movies_df['gross'].astype(int)
    movies_df['revenue']=movies_df['revenue'].astype(int)
    movies_df['budget']=movies_df['budget'].astype(int)
    movies_df['vote_count_letterboxd']=movies_df['vote_count_letterboxd'].astype(int)
    movies_df['vote_count_imdb']=movies_df['vote_count_imdb'].astype(int)
    movies_df['release_date'] = pd.to_datetime(movies_df['release_date'])
    movies_df['popularity']=movies_df['popularity'].astype(int)


    # Merge 'genre_letterboxd', 'genre_tmdb', and 'genre_imdb' columns into one feature and eliminate duplicated genres.
    movies_df['genre_letterboxd'] = movies_df['genre_letterboxd'].astype(str).str.replace(r"[\"\[\]]", "", regex=True)      # Make genre_letterboxd column has the same format as the other two genre columns
    movies_df['genre'] = movies_df['genre_letterboxd']+ ', ' + movies_df['genre_tmdb'] + ', ' + movies_df['genre_imdb']     # Merge the three genre columns into one general genre column
    movies_df['genre'] = movies_df['genre'].str.lower().str.replace(' ', '')                                                # Take the genre column values to a standard format by removing spaces and applying lowercase
    movies_df['genre'] = movies_df['genre'].apply(lambda x: list(set(x.split(','))))                                        # Split the genre column into individual genres, removing the duplicates
    movies_df['genre'] = movies_df['genre'].apply(lambda x: ', '.join(x))                                                   # Combine unique genres
    movies_df.drop(columns=['genre_letterboxd', 'genre_imdb', 'genre_tmdb'], inplace=True)                                  # Drop unnecessary columns


    # Extract only the first production country
    movies_df['production_countries'] = movies_df['production_countries'].apply(ast.literal_eval)
    max_countries = movies_df['production_countries'].apply(len).max()
    for i in range(max_countries):
        col_name = f'country_{i+1}'
        movies_df[col_name] = movies_df['production_countries'].apply(lambda x: x[i] if i < len(x) else None)
    movies_df = movies_df.drop(columns=[f'country_{i}' for i in range(2, max_countries+1)])
    movies_df.drop('production_countries', axis=1, inplace=True, errors='ignore')
    movies_df.rename(columns={'country_1': 'production_country'}, inplace=True)


    # Lower case production companies names
    movies_df['production_companies'] = movies_df['production_companies'].str.lower()
    movies_df['production_companies'] = movies_df['production_companies'].apply(lambda x: list(set(x.split(', '))))


    # Clasiffy the season a movie was released according to 'release_date' column
    movies_df['season'] = movies_df['release_date'].dt.month.apply(get_season)
    movies_df= movies_df.drop('release_date', axis=1)


    #------------------------------------------ VALUES ADJUSTMENT -------------------------------------------#
    
    # Adjust money values according to inflation trhough years
    movies_df['gross_adjusted'] = movies_df.apply(lambda x: adjust_for_inflation(x['gross'], x['year']), axis=1)
    movies_df['budget_adjusted'] = movies_df.apply(lambda x: adjust_for_inflation(x['budget'], x['year']), axis=1)
    movies_df['revenue_adjusted'] = movies_df.apply(lambda x: adjust_for_inflation(x['revenue'], x['year']), axis=1)
    movies_df.drop(columns=['budget','gross','revenue'], inplace=True)


    # Merge 'rating_letterboxd' and 'rating_imdb' columns by calculating a weighted average 
    movies_df['Weighted_Rating'] = (((movies_df['rating_letterboxd'] * movies_df['vote_count_letterboxd']) +
                                     (movies_df['rating_imdb'] * movies_df['vote_count_imdb']))            /
                                     (movies_df['vote_count_letterboxd'] + movies_df['vote_count_imdb']))
    
    movies_df.drop(columns=['rating_letterboxd', 'vote_count_letterboxd', 'rating_imdb', 'vote_count_imdb'], inplace=True)


    #----------------------------------- WEIGHTING RATING ACCORDING TO RELEVANCE OF DIRECTOR -----------------------------#
    # Select most relevant directors according to gross influence. Info source: https://www.the-numbers.com/box-office-star-records/worldwide/lifetime-specific-technical-role/director 
    top_directors= [
        "Steven Spielberg",
        "James Cameron",
        "Anthony Russo",
        "Joe Russo",
        "Peter Jackson",
        "Michael Bay",
        "David Yates",
        "Christopher Nolan",
        "J.J. Abrams",
        "Ridley Scott",
        "Tim Burton",
        "Robert Zemeckis",
        "Jon Favreau",
        "Ron Howard",
        "Sam Raimi",
        "James Wan"
    ]

    instances = 0
    irrelevant_directors = list()
    not_found_directors = list()
    for director in top_directors:
        instances += search_director(movies_df, director, {'year':'year', 'director':'director'},
                                     irrelevant_directors, not_found_directors)

    # Counting Anthony Russo and Joe Russo as one: Russo Brothers, as they are together in all the instances
    top_directors[top_directors.index('Anthony Russo')] = 'Russo Brothers'
    top_directors.remove('Joe Russo')
    movies_df['director'] = movies_df['director'].replace('Anthony Russo,Joe Russo', 'Russo Brothers')

    # Remove irrelevant and not found directors form the top_directors list
    for director in irrelevant_directors:
        top_directors.remove(director)
    for director in not_found_directors:
        top_directors.remove(director)

    # Apply the weighting factor based on the director's membership in the top list
    movies_df['Weighted_Director'] = movies_df.apply(lambda row: row['Weighted_Rating'] * 1.1 if row['director'] in top_directors else row['Weighted_Rating'] * 0.9, axis=1)


    #----------------------------------- WEIGHTING RATING ACCORDING TO RELEVANCE OF ACTOR ---------------------------------------#
    # Select most relevant actors according to gross influence. Info source: https://www.the-numbers.com/box-office-star-records/worldwide/lifetime-acting/top-grossing-leading-stars
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

    # Remove irrelevant and not found stars form the top_stars list
    for star in not_found_stars:
        top_stars.remove(star)
    for star in irrelevant_stars:
        top_stars.remove(star)
    
    # Apply the function to adjust the Weighted_Rating based on the actors.
    movies_df['Weighted_Rating_Actors'] = movies_df.apply(lambda row: row['Weighted_Rating'] * 1.1 if is_top_star(row['star'], top_stars) else row['Weighted_Rating'] * 0.9, axis=1)

    #----------------------------------- WEIGHTING RATING ACCORDING TO RELEVANCE OF COUNTRY ---------------------------------------#
    # Select most relevant actors according to gross influence. Info source: https://www.the-numbers.com/movies/production-countries/#tab=territory
    top_countries = [
        "United States of America",
        "United Kingdom",
        "China",
        "France",
        "Japan",
        "Germany",
        "South Korea",
        "Canada",
        "India",
        "Australia",
        "Hong Kong",
        "New Zealand",
        "Italy",
        "Spain"
    ]
    # Apply function to djust the Weighted_Rating based on the country of production.
    movies_df['Weighted_Rating_Country'] = movies_df.apply(lambda row: row['Weighted_Rating'] * 1.1 if row['production_country'] in top_countries else row['Weighted_Rating'] * 0.9, axis=1)


    #------------------------------ WEIGHTING RATING ACCORDING TO RELEVANCE OF PRODUCTION COMPANY -------------------------------#
    # Select most relevant actors according to gross influence. Info source: https://www.the-numbers.com/movies/production-companies/#production_companies_overview=od3
    top_producers = [
        "Warner Bros",
        "Universal Pictures",
        "Columbia Pictures",
        "Marvel Studios",
        "Walt Disney Pictures",
        "Paramount",
        "20th Century Fox",
        "Legendary Pictures",
        "New Line Cinema",
        "DreamWorks Animation",
        "Dune Entertainment",
        "Amblin Entertainment",
        "Disney-Pixar",
        "Relativity Media",
        "Metro-Goldwyn-Mayer Pictures",
        "Village Roadshow Productions",
        "DreamWorks Pictures",
        "Heyday Films",
        "Regency Enterprises",
        "Lucasfilm",
        "Walt Disney Animation Studios",
        "Lionsgate",
        "TSG Entertainment",
        "RatPac Entertainment",
        "Illumination Entertainment",
        "Original Film",
        "Skydance Productions",
        "Summit Entertainment",
        "Touchstone Pictures",
        "di Bonaventura Pictures"
    ]

    
    # Dictionary of values to be replaced for different production companies
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

    # Replace value for each production company
    movies_df = replace_values(movies_df, {'producer':'production_companies'}, replacements)

    # Updated list of top producers
    updated_top_companies = [
        "Warner Bros", "Universal Pictures", "Columbia Pictures", "Marvel Studios", "Walt Disney Pictures",
        "Paramount Pictures", "20th Century Fox", "Legendary Pictures", "New Line Cinema", "DreamWorks Animation",
        "Dune Entertainment", "Amblin Entertainment", "Disney-Pixar", "Relativity Media", "Metro-Goldwyn-Mayer",
        "Village Roadshow Productions", "DreamWorks Pictures", "Heyday Films", "Regency Enterprises", "Lucasfilm Ltd.",
        "Walt Disney Animation Studios", "Lionsgate", "TSG Entertainment", "RatPac Entertainment", "Illumination Entertainment",
        "Original Film", "Skydance", "Summit Entertainment", "Touchstone Pictures", "di Bonaventura Pictures"
    ]

    # Prepare a list of columns for top producers based on corrected names
    updated_top_producer_columns_corrected = [
        'producer_Warner Bros', 'producer_Universal Pictures', 'producer_Columbia Pictures',
        'producer_Marvel Studios', 'producer_Disney', 'producer_Paramount',
        'producer_20th Century Studios', 'producer_Legendary', 'producer_New Line Cinema',
        'producer_DreamWorks', 'producer_Dune Entertainment', 'producer_Amblin',
        'producer_Pixar', 'producer_Relativity Media', 'producer_Metro-Goldwyn-Mayer',
        'producer_Village Roadshow', 'producer_DreamWorks', 'producer_Heyday Films',
        'producer_Regency Enterprises', 'producer_Lucasfilm', 'producer_Disney',
        'producer_Lionsgate', 'producer_TSG Entertainment', 'producer_RatPac-Dune Entertainment',
        'producer_Illumination', 'producer_Original Film', 'producer_Skydance',
        'producer_Summit Entertainment', 'producer_Touchstone Pictures', 'producer_di Bonaventura Pictures'
    ]

    # Apply the correction considering that the name of the columns may be slightly different
    # and some of them may not be present in the data.
    existing_columns = [col for col in updated_top_producer_columns_corrected if col in movies_df.columns]
    
    # Apply the function to each row of the dataframe
    movies_df['Weighted_Rating_Companies'] = movies_df.apply(lambda row: calculate_corrected_weighted_rating_companies_v2(row, existing_columns), axis=1)


    #------------------------------ WEIGHTING RATING ACCORDING TO GENRE -------------------------------#
    movies_df['Average_Weighted_Rating'] = movies_df['Weighted_Rating_Companies']
    movies_df['Weighted_Rating_Genre'] = movies_df.apply(adjust_rating_by_genre, axis=1) # Apply the function to adjust the Weighted_Rating based on the gender


    #------------------------------ WEIGHTING RATING ACCORDING TO REVENUE -------------------------------#
    movies_df['Average_Adjusted_Ratings'] = movies_df[['Weighted_Rating_Companies', 'Weighted_Rating_Genre']].mean(axis=1)
    movies_df['Adjusted_Revenue'] = movies_df.apply(adjust_revenue_by_ratings, axis=1)      # Apply the function to adjust the revenue_adjusted based on the adjusted ratings

    # Calculate the median and mean of revenue_adjusted
    median_revenue = movies_df['revenue_adjusted'].median()
    mean_revenue = movies_df['revenue_adjusted'].mean()

    # Use the adjusted Weighted_Rating columns provided.
    movies_df['Average_Weighted_Rating'] = movies_df[['Weighted_Director', 'Weighted_Rating_Actors',
                                            'Weighted_Rating_Country', 'Weighted_Rating_Companies',
                                            'Weighted_Rating_Genre']].mean(axis=1)


    # Apply the adjusted functions
    movies_df['Weighted_Rating_Adjusted_Revenue_Median'] = movies_df.apply(lambda x: adjust_rating_by_median_revenue(x, median_revenue), axis=1)
    movies_df['Weighted_Rating_Adjusted_Revenue_Mean'] = movies_df.apply(lambda x: adjust_rating_by_mean_revenue(x, mean_revenue), axis=1)


    #------------------------------ WEIGHTING RATING ACCORDING TO BUDGET -------------------------------#
    # Calculate median and mean of budget_adjusted
    median_budget = movies_df['budget_adjusted'].median()
    mean_budget = movies_df['budget_adjusted'].mean()

    # Use the adjusted Weighted_Rating columns provided to calculate an average.
    movies_df['Average_Weighted_Rating'] = movies_df[['Weighted_Director', 'Weighted_Rating_Actors',
                                            'Weighted_Rating_Country', 'Weighted_Rating_Companies',
                                            'Weighted_Rating_Genre']].mean(axis=1)

    # Apply the adjusted functions
    movies_df['Weighted_Rating_Adjusted_Budget_Median'] = movies_df.apply(lambda x: adjust_rating_by_median_budget(x, median_budget), axis=1)
    movies_df['Weighted_Rating_Adjusted_Budget_Mean'] = movies_df.apply(lambda x: adjust_rating_by_mean_budget(x, mean_budget), axis=1)

    # Apply the function to adjust the budget_adjusted
    movies_df['Adjusted_Budget'] = movies_df.apply(adjust_budget_by_ratings, axis=1)

    #------------------------------ DROP COLUMNS NOT LONGER NECESSARY -------------------------------#
    # List of columns to delete
    columns_to_drop = [
        'title', 'popularity', 'year', 'director', 'star', 'runtime',
        'production_companies', 'genre',
        'production_country', 'season', 'budget_adjusted', 'revenue_adjusted'
    ]
    # Remove columns from the DataFrame
    movies_df.drop(columns=columns_to_drop, inplace=True)

    # Save to interim data
    filepath = Path('data/interim/gross_predict_dataset.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    movies_df.to_csv(filepath, index=True)


if __name__ == '__main__':
    build_features()