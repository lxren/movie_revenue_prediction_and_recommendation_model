# -*- coding: utf-8 -*-
import pandas as pd
import pdb


def select_features(movies_df, threshold=0.2):
    """
    Selects features based on their correlation with the target variable 'gross_adjusted',
    excluding features that have a correlation below a specified threshold.

    Args:
        movies_df (pd.DataFrame): The DataFrame containing the movie data.
        threshold (float): The minimum absolute correlation required to keep a feature.

    Returns:
        list: A list of selected feature names that have a correlation above the threshold with 'gross_adjusted'.
    """
    # Calculating the correlation matrix between the characteristics and the target variable
    correlation_matrix = movies_df.corr()

    # Extracting the specific correlation with 'gross_adjusted'.
    # Excluding correlation with itself and 'Unnamed: 0' if it exists
    correlation_with_target = correlation_matrix['gross_adjusted'].drop('gross_adjusted', errors='ignore')

    # Optionally drop other non-feature columns like 'Unnamed: 0' if they exist in the correlation matrix
    correlation_with_target = correlation_with_target.drop(['Unnamed: 0'], errors='ignore')

    # Sorting the characteristics by their absolute correlation to see the most significant ones
    sorted_correlation = correlation_with_target.abs().sort_values(ascending=False)

    # Filter out features with low correlation
    selected_features = [feature for feature, corr in sorted_correlation.items() if corr >= threshold]

    return selected_features


if __name__ == '__main__':
    #Extract data from csv file
    movies_df = pd.read_csv("data/interim/gross_built_features_dataset.csv")
    print(movies_df)
    print(select_features(movies_df))