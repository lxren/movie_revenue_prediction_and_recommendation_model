Movie Revenue Prediction and Recommendation Model
==============================

This model predicts a movie's gross revenue based on select feature data and includes an engine that generates movie recommendations based on user input and NLP techniques. 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── download_dataset.py
    │   │   └── make_dataset.py    
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   ├── build_features.py
    │   │   └── select_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
Getting Started
--------
These instructions will get you a copy of the project up and runnining on your local machine. 

### Prerequisites
Install Python 3.5 or greater

Optional: We recommend the use of a virtual environment (venv) when running the code on your local machine

## Running the code

### Install libraries
```bash
pip install .
```
### Download data
```bash
python src/data/download_dataset.py
```
### Run EDA
```bash
python src/data/make_dataset.py
```
### Visualize Genre in Wordcloud
```bash
python src/visuaization/make_wordcloud.py --genre <genre_name>

genre_name: Action|Adult|Adventure|Animation|Biography|Comedy|Crime|Documentary|Drama|Family|Fantasy|Film-Noir|History|Horror|Music|Musical|Mystery|Romance|Sci-Fi|Science Fiction|Sport|TV Movie|Thiller|War|Western
```
### Generate Movie Recommendation Based on Movie Title
```bash
python src/model/recommend_movie.py --movie <move_name_and_year>

movie_name_and_year: Inception (2010)|Her (2013)|Dune (2021)|La La Land (2016)|Mars Attacks! (1996)|The Departed (2006)
```

### Visualize initial feature patterns in data for Gross prediction
```bash
python /src/visualization/visualize.py
```

### Adapt features for Gross prediction through feature engineering
```bash
python /src/features/build_features.py
```
### Traing moodel for Gross prediction
```bash
python /src/models/train_model.py
```

### Make Gross predictions through manual entries
```bash
python /src/models/predict_model.py
```


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
