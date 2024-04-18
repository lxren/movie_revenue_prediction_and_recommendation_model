from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import argparse

def recommend_movie(movie_input):
    df = pd.read_csv('./data/interim/cleaned_film_dataset.csv')
    df['description_imdb'] = df['description_imdb'].str.replace('See full summary','',regex=True)
    for index, row in df.iterrows():
        df.at[index,'descriptions'] = row['description_letterboxd']+" "+row['description_imdb']+" "+row['description_tmdb']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['descriptions'])
    tfidf_matrix.shape

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    df['year'] = df['year'].astype(int)
    df['title_with_year'] = df['title'] + ' (' + df['year'].astype(str) + ')'

    indices = pd.Series(df.index, index=df['title_with_year']).drop_duplicates()
    if movie_input not in indices:
        raise Exception('Sorry, movie not found in database')

    idx = indices[movie_input]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    return df['title_with_year'].iloc[movie_indices]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--movie', help=f"Provide a movie title followed by year in parentheses e.g. Finding Nemo (2003)")
    args = parser.parse_args()
    movie= args.movie
    if not movie:
        movie = input('Please provide a movie title and year in parentheses e.g. Finding Nemo (2003): ')
    output = recommend_movie(movie)
    print(output)
