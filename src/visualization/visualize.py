import ast
import numpy as np
import pandas as pd
import plotly.express as px

def extract_ids(column):
        return column.str.extractall('nm(\d+)/').groupby(level=0).agg(','.join)[0]

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def cleanup_dataframe(movies_df):
    """Clean and prepare dataframe for visualization."""
    drop_columns = ['genre_imdb','spoken_languages','description_tmdb','Unnamed: 0','genre_letterboxd', 'imdb_id', 'original_language','description_letterboxd','tmdb_id','description_imdb']
    movies_df.drop(columns=drop_columns, inplace=True, errors='ignore')
    convert_to_int = ['year', 'runtime', 'gross', 'revenue', 'budget', 'vote_count_letterboxd', 'vote_count_imdb']
    movies_df[convert_to_int] = movies_df[convert_to_int].apply(pd.to_numeric, errors='coerce', axis=1)
    movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')

def process_columns(movies_df):
    """Process specific columns for further analysis."""
    movies_df['main_genre'] = movies_df['genre_tmdb'].str.split(',').str[0]
    movies_df['main_director'] = movies_df['director'].str.split(',').str[0]
    movies_df['director_ids'] = extract_ids(movies_df['director_id'])

    split_stars = movies_df['star'].str.split(',', n=2, expand=True)
    movies_df['first_star'], movies_df['second_star'] = split_stars[0], split_stars[1].str.replace('\n', '', regex=False)
    movies_df['star_ids'] = extract_ids(movies_df['star_id'])
    movies_df[['star_id_1', 'star_id_2']] = movies_df['star_ids'].str.split(',', expand=True, n=2)[[0, 1]]

    # Handling production countries as a list
    movies_df['production_countries'] = movies_df['production_countries'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
    max_countries = movies_df['production_countries'].apply(len).max()
    for i in range(max_countries):
        col_name = f'country_{i+1}'
        movies_df[col_name] = movies_df['production_countries'].apply(lambda x: x[i] if i < len(x) else None)

    movies_df.drop(columns=[f'country_{i}' for i in range(2, max_countries+1)], inplace=True)
    movies_df.rename(columns={'country_1': 'production_country'}, inplace=True)
    movies_df['season'] = movies_df['release_date'].dt.month.apply(get_season)

def visualize_data(movies_df):
    """Create visualizations for the movie data."""
        # Visualize budget vs gross
    fig = px.scatter(movies_df, x='budget', y='gross', trendline='ols')
    fig.update_layout(title='Budget vs Gross')
    fig.show()

    # Visualize runtime vs gross
    fig = px.scatter(movies_df, x='runtime', y='gross', trendline='ols')
    fig.update_layout(title='Runtime vs Gross Revenue')
    fig.show()

    # Visualize imdb rating vs. gross
    fig = px.scatter(movies_df, x='rating_imdb', y='gross', trendline='ols')
    fig.update_layout(title='IMDB Rating vs Gross')
    fig.show()

    # Visualize averae gross over time
    fig = px.line(movies_df[['year','gross']].groupby('year').mean().reset_index(), x='year', y='gross')
    fig.update_layout(title='Average Gross Over Time')
    fig.show()

    # Visualize average seasonal gross
    seasonal_gross = movies_df.groupby(['year', 'season'])['gross'].mean().reset_index()
    fig = px.line(movies_df.groupby(['year', 'season'])['gross'].mean().reset_index(), x='year', y='gross', color='season',
                title='Average Gross for Season trough the years',
                labels={'year': 'Year', 'gross': 'Average Gross'},
                category_orders={"Season": ["Winter", "Spring", "Summer", "Fall"]})

    fig.update_layout(xaxis_title='Year', yaxis_title='Average Gross', hovermode='x')

    fig.show()

    # Visualize Primary Genre vs. Gross
    fig = px.bar(movies_df[['main_genre','gross']].groupby('main_genre').mean().reset_index(), x='main_genre', y='gross')
    fig.update_layout(title='Primary Genre vs Gross', xaxis={'categoryorder':'total descending'})
    fig.show()
    
    # Visualize distribution of primary genre
    fig = px.pie(movies_df['main_genre'], names='main_genre')
    fig.update_layout(title='Distribution of Primary Genre')
    fig.show()

    # Visualize Top 20 directors by gross and budget
    gross_by_director = movies_df.groupby('main_director')['gross'].sum().reset_index()
    top_directors = gross_by_director.sort_values('gross', ascending=False).head(20)

    df_top_directors = movies_df[movies_df['main_director'].isin(top_directors['main_director'])]

    fig = px.scatter(df_top_directors,
                    x='budget',
                    y='gross',
                    color='main_director',  # Color by director to differentiate them
                    size='gross',  # Use 'gross' as bubble size to highlight top-grossing movies
                    hover_name='main_director',  #Shows the director's name when you mouse over the bubble
                    title='Top 20 Directors by Gross: Gross vs. Budget')

    fig.update_layout(xaxis_title='Budget', yaxis_title='Gross', legend_title='Top 20 Directors by Gross')
    fig.show()

    # Visualize Top 20 directors by gross and number of movies
    director_counts = movies_df.groupby('main_director').agg({'gross': 'sum', 'title': 'count'}).reset_index()

    director_counts.columns = ['main_director', 'total_gross', 'movie_count']

    top_directors = director_counts.sort_values('total_gross', ascending=False).head(20)

    fig = px.scatter(top_directors,
                    x='movie_count',
                    y='total_gross',
                    size='total_gross',
                    color='main_director',
                    hover_name='main_director',
                    title='Top 20 Directors: Number of movies vs. Gross Total')

    fig.update_layout(xaxis_title='Number of movies', yaxis_title='Gross Total', legend_title='Director')
    fig.show()

    # Visualize Top 50 Actors by gross and budget
    actor_stats = movies_df.groupby('first_star').agg({'gross': 'sum', 'budget': 'sum', 'title': 'count'}).reset_index()

    actor_stats.columns = ['star', 'total_gross', 'total_budget', 'movie_count']

    top_actors = actor_stats.sort_values('total_gross', ascending=False).head(50)

    fig = px.scatter(top_actors,
                    x='total_budget',
                    y='total_gross',
                    size='movie_count',
                    color='star',
                    hover_name='star',
                    title='Top 50 1- Actors: Gross Total vs. Budget Total')

    fig.update_layout(xaxis_title='Budget Total', yaxis_title='Gross Total', legend_title='Actor')

    fig.show()

    actor_stats = movies_df.groupby('second_star').agg({'gross': 'sum', 'budget': 'sum', 'title': 'count'}).reset_index()

    actor_stats.columns = ['star', 'total_gross', 'total_budget', 'movie_count']

    top_actors = actor_stats.sort_values('total_gross', ascending=False).head(50)

    fig = px.scatter(top_actors,
                    x='total_budget',
                    y='total_gross',
                    size='movie_count',
                    color='star',
                    hover_name='star',
                    title='Top 50 2- Actors: Gross Total vs. Budget Total')

    fig.update_layout(xaxis_title='Budget Total', yaxis_title='Gross Total', legend_title='Actor')

    fig.show()

    # Visualize top 50 Actors by gross and number of appearances
    fig = px.scatter(top_actors,
                 x='movie_count',
                 y='total_gross',
                 size='movie_count',
                 color='star',
                 hover_name='star',
                 title='Top 50 1- Actors: Gross Total vs. Number of Appearances')

    fig.update_layout(xaxis_title='Number of Appearances', yaxis_title='Gross Total', legend_title='Actor')
    fig.show()

    fig = px.scatter(top_actors,
                 x='movie_count',
                 y='total_gross',
                 size='movie_count',
                 color='star',
                 hover_name='star',
                 title='Top 50 2- Actors: Gross Total vs. Number of Appearances')

    fig.update_layout(xaxis_title='Number of Appearances', yaxis_title='Gross Total', legend_title='Actor')
    fig.show()

def visualize(movies_df):
    """Main function to handle visualization process."""
    cleanup_dataframe(movies_df)
    process_columns(movies_df)
    visualize_data(movies_df)

if __name__ == '__main__':
    movies_df = pd.read_csv("data/interim/cleaned_film_dataset.csv")
    visualize(movies_df)