<img src ="https://github.com/lxren/movie_revenue_prediction_and_recommendation_model/assets/167150651/f7296a1f-7b4b-46af-a31f-aca892ee2138"/>

# Model Card
Last updated: April 2024
## Model Details
### Model Name
movie_revenue_prediction_and_recommendation_model
### Model Version
0.1.0
### Model Type
Regression models using Random Forest and Neural Network and NLP-based recommendation engine
### Developers
- Lily Ren [@lxren](https://github.com/lxren)
- Fatima Ramirez Rodriguez [@fmramirez](https://github.com/fmramirez)
- Andrea Rodriguez Moreno [@Mariette182](https://github.com/Mariette182)
- Paolo Andrés Pancho Ramírez [@Zar93Paolo](https://github.com/Zar93Paolo)
### Release Date
April 19, 2024
## Intended Use
### Primary Use
- To predict the gross revenue of movies based on director, actors, main genre, production country, and production companies. 
- To generate movie recommendation based on user provided input.
### Intended Users
- Investors & Filmmakers
- Everyday Users & Film Enjoyers
### Out-of-Scope Use Cases
- The model ascribes weighting based on the ranking of directors, actors, and production companies and should be used in tandem with industry research and further data refinement
- The model has a limited dataset of movies after combining multiple film databases due to missing and unreliable data, movies not included in the dataset will generate exception message
## Data Description
### Data Used
- IMDB Dataset: 243,438 rows, 14 features
- TMDB Dataset: 1,014,305 rows, 23 columns
- Letterboxd Dataset: 285,964 rows, 19 columns
- Post Merging & EDA Dataset: 11,241 rows, 30 columns
### Features
### Model Architecture
#### Gross Revenue Predction Model
#### Recommendation Engine
The recommendation engine uses NLTK to process film descriptions collated from three databases. The descriptions are tokenized, lemmatized using WordNet, and Stop Words were implemented. The wordcloud library was used to generate word clouds across genres to implement additional stop words and visualize word clouds for term frequency. TF-IDF was used the vectorize the tokens and cosine similarity was performed on the TF-IDF matrix. Recommendations are made based on cosine similarity between tokenized film descriptions. 
## Training & Evaluation
### Training Procedure
### Evaluation Metrics
### Baseline Comparison

## Ethical Considerations
### Fairness & Bias
There may be bias in the dataset as it includes only a subset of movies due to data availability. The dataset is English despite the inclusion of foreign films, translation biases may be captured. The film's rating scores are weighted by the number of votes on each database platform in attempt to reduce bias and outliers. 
### Privacy
N/A
### Security
N/A

## Limitations & Recommendations
### Known Limitations
### Recommendations for Use
