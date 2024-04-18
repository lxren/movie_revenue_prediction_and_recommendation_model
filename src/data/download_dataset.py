import os
from urllib.request import urlretrieve

def initialize_data_paths():
    paths = ['data/raw', 'data/external', 'data/interim', 'data/processed']

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def download_dataset():
    initialize_data_paths()

    urls = [
        ("1PWLqXQZzcBOcxNZVTRsro2H3xWjwzhHC",
        "TMDB_movie_dataset_v11.csv"),
        ("15Nz4ldGeVGYK8znnPnhvtQL6GoL0hE2_",
        "letterboxd_data.csv"),
        ("1DcyGoLsT8vvDwQX-tOge8YBlWURaCzfl",
        "imdb.csv")
    ]

    for (fileId, filename) in urls:
        urlretrieve(f"https://drive.usercontent.google.com/download?id={fileId}&export=download&authuser=1&confirm=t", f"data/raw/{filename}")

if __name__ == '__main__':
    download_dataset()
    
 
