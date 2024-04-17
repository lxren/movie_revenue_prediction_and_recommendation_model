# -*- coding: utf-8 -*-
from pathlib import Path
import pandas

def make_dataset():

    # Read csv and generate dataframes
    letterboxd_data = pandas.read_csv("data/raw/letterboxd_data.csv")
    imdb_data = pandas.read_csv("data/raw/imdb.csv")
    tmdb_data = pandas.read_csv("data/raw/TMDB_movie_dataset_v11.csv")

    # Drop duplicates
    letterboxd_data = letterboxd_data.drop_duplicates(['imdb_id'])
    letterboxd_data = letterboxd_data.drop_duplicates(['tmdb_id'])

    # Letterboxd data drop and rename
    columns_to_drop = ['_id', 'image_url', 'imdb_link', 'movie_id', 'tmdb_link', 'runtime', 'spoken_languages']
    columns_to_rename = {'genres' : 'genre_letterboxd',
                     'movie_title': 'title',
                     'overview' : 'description_letterboxd',
                     'year_released': 'year',
                     'vote_average': 'rating_letterboxd',
                     'vote_count': 'vote_count_letterboxd'}
    if(any(column in columns_to_drop for column in letterboxd_data.columns)):
        letterboxd_data.drop(columns_to_drop, axis=1, inplace=True)
    else:
        print("Column already dropped")

    letterboxd_data.rename(columns=columns_to_rename, inplace=True)

    # IMDB data drop and rename
    columns_to_drop = ['Unnamed: 0', 'Unnamed: 1', 'certificate', 'year', 'movie_name', 'runtime']
    columns_to_rename = {'movie_id' : 'imdb_id',
                     'genre' : 'genre_imdb',
                     'description' : 'description_imdb',
                     'gross(in $)' : 'gross',
                     'rating' : 'rating_imdb',
                     'votes' : 'vote_count_imdb'}
    if(any(column in columns_to_drop for column in imdb_data.columns)):
        imdb_data.drop(columns_to_drop, axis=1, inplace=True)
    else:
        print("Column already dropped")

    imdb_data.rename(columns=columns_to_rename, inplace=True)

    # TMDB data drop and rename
    columns_to_drop = ['vote_average',
                   'vote_count',
                   'status',
                   'release_date',
                   'backdrop_path',
                   'homepage',
                   'imdb_id',
                   'original_language',
                   'original_title',
                   'popularity',
                   'poster_path',
                   'tagline',
                   'production_countries',
                   'title'
                   ]
    columns_to_rename = {'id' : 'tmdb_id',
                      'overview' : 'description_tmdb',
                      'genres' : 'genre_tmdb',
                    }
    if(any(column in columns_to_drop for column in tmdb_data.columns)):
        tmdb_data.drop(columns_to_drop, axis=1, inplace=True)
    else:
        print("Column already dropped")
    tmdb_data.rename(columns=columns_to_rename, inplace=True)

    tmdb_data = tmdb_data.drop_duplicates(['tmdb_id'])

    # Clean null values in Letterboxd
    columns_to_exclude = ['original_language',
                      'production_countries',
                      'release_date',
                      'spoken_languages',
                      'vote_count'
                      ]
    letterboxd_data_clean = letterboxd_data.dropna(subset=[col for col in letterboxd_data.columns if col not in columns_to_exclude])

    # Clean null values in IMDB
    columns_to_exclude = ['runtime_imdb']
    imdb_data_clean = imdb_data.dropna(subset=[col for col in imdb_data.columns if col not in columns_to_exclude])

    # Clean null values in TMDB
    columns_to_exclude = ['runtime_tmdb',
                      'spoken_languages']
    tmdb_data_clean = tmdb_data.dropna(subset=[col for col in tmdb_data.columns if col not in columns_to_exclude])

    # Merge IMDB with Letterboxd
    imdb_data_merged_letterboxd = pandas.merge(letterboxd_data_clean, imdb_data_clean, on=['imdb_id'], how='inner')

    # Merge all three
    tmdb_imdb_letterboxd_merged = pandas.merge(imdb_data_merged_letterboxd, tmdb_data_clean, on=['tmdb_id'], how='inner')

    # Save to interim data
    filepath = Path('data/interim/cleaned_film_dataset.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmdb_imdb_letterboxd_merged.to_csv(filepath, index=True)


if __name__ == '__main__':
    make_dataset()
 