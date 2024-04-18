import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import argparse

def make_wordcloud(genre_input):

    # Read interim data file and choose the longest string for genre (most inclusive)
    df = pd.read_csv('./data/interim/cleaned_film_dataset.csv')
    df['genre_letterboxd'] = df['genre_letterboxd'].str.replace('[\[\]"]', '', regex=True)
    def longest_string(row):
        if len(row['genre_imdb']) >= len(row['genre_tmdb']):
            return row['genre_imdb']
        else:
            return row['genre_tmdb']
    df['genre'] = df.apply(longest_string, axis=1)

    # Split and one-hot encode the genres
    genres = np.unique(', '.join(df['genre']).split(', '))
    for genre in genres:
        df[genre] = df['genre'].str.contains(genre).astype('int')

    # Process description text
    custom_stop_words = ['find','one','two','life','man','woman','young','must','take','boy','girl']
    stop_words = set(stopwords.words('english'))#+custom_stop_words)
    lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [(lemmatizer.lemmatize(word)) for word in tokens if word not in stop_words]
        return " ".join(tokens)
    
    df['description_imdb'] = df['description_imdb'].str.replace('See full summary','',regex=True)
    df['description_letterboxd_processed'] = df['description_letterboxd'].apply(preprocess_text)
    df['description_imdb_processed'] = df['description_imdb'].apply(preprocess_text)
    df['description_tmdb_processed'] = df['description_tmdb'].apply(preprocess_text)

    for index, row in df.iterrows():
        df.at[index,'descriptions'] = row['description_letterboxd']+" "+row['description_imdb']+" "+row['description_tmdb']
    
    # genres = ['Action', 'Adult', 'Adventure','Animation', 'Biography', 'Comedy', 'Crime', 'Documentary',
    #       #'Drama','Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music','Musical',
    #       #'Mystery', 'Romance', 'Sci-Fi', 'Science Fiction', 'Sport','TV Movie', 'Thriller', 'War',
    #       #'Western'
    #       ]

    colormap_mapping = {
        'Action': 'Reds',  # Exciting and intense
        'Adult': 'cividis',  # Neutral and mature
        'Adventure': 'Blues',  # Expansive and adventurous
        'Animation': 'Greens',  # Bright and playful
        'Biography': 'Greys',  # Reflective and factual
        'Comedy': 'Oranges',  # Lighthearted and humorous
        'Crime': 'PuBuGn',  # Intense and gritty
        'Documentary': 'Purples',  # Informative and factual
        'Drama': 'YlOrBr',  # Emotional and introspective
        'Family': 'RdYlGn',  # Warm and inclusive
        'Fantasy': 'RdPu',  # Imaginative and magical
        'Film-Noir': 'RdYlBu',  # Dark and atmospheric
        'History': 'PuBu',  # Educational and enlightening
        'Horror': 'BuPu',  # Eerie and suspenseful
        'Music': 'YlGnBu',  # Uplifting and rhythmic
        'Musical': 'Wistia',  # Melodic and theatrical
        'Mystery': 'OrRd',  # Intriguing and suspenseful
        'Romance': 'RdGy',  # Passionate and romantic
        'Sci-Fi': 'GnBu',  # Futuristic and imaginative
        'Science Fiction': 'cubehelix',  # Speculative and visionary
        'Sport': 'coolwarm',  # Energetic and competitive
        'TV Movie': 'hot',  # Varied depending on theme
        'Thriller': 'plasma',  # Suspenseful and thrilling
        'War': 'inferno',  # Heroic and patriotic
        'Western': 'twilight'  # Frontier and adventurous
    }
    
    output_dir = './reports/figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    colormap = colormap_mapping.get(genre_input, 'viridis')
    genre_df = df[df[genre_input] == 1]
    text = ' '.join(genre_df['description_imdb_processed'])
    wordcloud = WordCloud(width=800, height=400, background_color ='white',colormap=colormap).generate(text)
    output_file = os.path.join(output_dir, f'{genre_input}_wordcloud.png')
    wordcloud.to_file(output_file)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {genre_input} Movies')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--genre', help="Provide a genre from the following list: 'Action', 'Adult', 'Adventure','Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama','Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Science Fiction', 'Sport','TV Movie', 'Thriller', 'War', 'Western'")
    args = parser.parse_args()
    genre = args.genre
    if not genre:
        genre = input('Please provide a genre you would like to visualize: ')

    make_wordcloud(genre)
