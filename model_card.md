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
- 'Budget': (Float) Budget used to produce film
- 'Director': (Str) Main Director's name
- 'Lead Actor': (Str) Main actor starring in the film
- 'Production Company': (Str) Main company that produced the film
- 'Country of production': (Str) Country where the movie was produced
- 'Genres': (List) List of genres that describe the film
- 'Release Year': (Int) Year in which the movie was released
- 'Season': (Str) Season of the year when the movie was released
- 'Rating': (Float) Preliminary rating given by critics, from 0 - 10
- 'Runtime': (Int) Duration of the film in minutes
- 'Title': (Str) Name of the movie
- 'Description' (Str) Sypnosis of the movie trama 

### Model Architecture

#### Gross Revenue Predction Model

Network Type: Sequential

##### Configuration:
- Input Layer: 128 units, ReLU activation, L2 regularization (0.01).
- Dropout: 30% after the first and second dense layers.
- Hidden Layers:
    - First hidden layer: 64 units, ReLU activation, L2 regularization (0.01).
    - Second hidden layer: 32 units, ReLU activation.
- Output Layer: Single unit for regression output.
- Optimizer: Adam, learning rate 0.001.
- Loss Function: Mean Squared Error (MSE).

#### Recommendation Engine
The recommendation engine uses NLTK to process film descriptions collated from three databases. The descriptions are tokenized, lemmatized using WordNet, and Stop Words were implemented. The wordcloud library was used to generate word clouds across genres to implement additional stop words and visualize word clouds for term frequency. TF-IDF was used the vectorize the tokens and cosine similarity was performed on the TF-IDF matrix. Recommendations are made based on cosine similarity between tokenized film descriptions. 
## Training & Evaluation
### Training Procedure
#### Data preparation
- Feature Engineering: Input features were processed to generate training features by implementing statistical calculations, and filtering methods
- Feature Selection: Features were selected based on their importance as determined by the select_features function.
- Scaling: Data was standardized using a StandardScaler. If a scaler was not provided, a new scaler was created and fitted to the training data. The scaling process ensures that all input features contribute equally to model training by converting them to have zero mean and unit variance.
- Data Splitting: The dataset was split into training and test sets with a ratio of 80:20 using a random seed for reproducibility.
#### Training Parameters:
- Epochs: 100
- Batch Size: 32
- Validation Split: 20%
- Early Stopping: Monitored on validation loss with a patience of 10 epochs.
- Optimizer: Adam, with learning rate of 0.001.
This model uses dropout and L2 regularization to mitigate overfitting, employing early stopping during training to prevent overtraining on the validation set.

### Evaluation Metrics
Metric for Loss: Mean Square Error 
Metric for monitoring early stopping: val_loss

### Baseline Comparison
N/A

## Ethical Considerations
### Fairness & Bias
There may be bias in the dataset as it includes only a subset of movies due to data availability. The dataset is English despite the inclusion of foreign films, translation biases may be captured. The film's rating scores are weighted by the number of votes on each database platform in attempt to reduce bias and outliers. Enhancing the training dataset to cover a broader spectrum of films to reduce geographic and cultural biases is highly recommended.
### Privacy
N/A
### Security
N/A

## Limitations & Recommendations
### Known Limitations
- This model don't consider the changing relevance of actors and directors, as any of them can become highly profitable suddenly
- Even though a Director's name can have significant impact on the gross of a movie, the quantity of films a director produced could be very reduced. 
- There's is a considerable bias towards American films, since its prevalence in movies productions, which can make gross prediction for movies from other countries less reliable.  
- Obtained data for training was limited due to the lack of reliable information for film gross and budget. More informative datasets are sold in industry competitive prices, e.g. IMDB's full database API.
- A considerable proportion of the training data is historical data, which can interefere with the gross prediction when combining with frequently changing features, such as directors and actors. Time and changing trends would greatly impact the results of this model. 

### Recommendations for Use
- Use the model's outputs alongside industry research and market trends for informed decisions.
- Test the model with diverse datasets before real-world application.
- Keep the model updated with new data to maintain relevance.

