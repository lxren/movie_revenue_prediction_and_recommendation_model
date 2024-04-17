from urllib.request import urlretrieve

urls = [
    ("https://drive.google.com/file/d/1PWLqXQZzcBOcxNZVTRsro2H3xWjwzhHC/view?usp=drive_link",
     "TMDB_movie_dataset_v11.csv"),
    ("https://drive.google.com/file/d/15Nz4ldGeVGYK8znnPnhvtQL6GoL0hE2_/view?usp=drive_link",
     "letterboxd_data.csv"),
    ("https://drive.google.com/file/d/1DcyGoLsT8vvDwQX-tOge8YBlWURaCzfl/view?usp=drive_link",
     "imdb.csv")
]

for (url, filename) in urls:
    urlretrieve(url, f"data/{filename}")


