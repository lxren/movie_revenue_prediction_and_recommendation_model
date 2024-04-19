# -*- coding: utf-8 -*-
import pandas as pd


def select_features(movies_df):

    # Calculating the correlation matrix between the characteristics and the target variable
    correlation_matrix = movies_df.corr()

    # Extracting the specific correlation with 'gross_adjusted'.
    correlation_with_target = correlation_matrix['gross_adjusted'].drop('gross_adjusted')  # Excluding correlation with itself

    # Sorting the characteristics by their absolute correlation to see the most significant ones.
    sorted_correlation = correlation_with_target.abs().sort_values(ascending=False)

    selected_features = list(sorted_correlation[:13].index)

    return selected_features


if __name__ == '__main__':
    #Extract data from csv file
    movies_df = pd.read_csv("data/interim/gross_built_features_dataset.csv")
    select_features(movies_df)