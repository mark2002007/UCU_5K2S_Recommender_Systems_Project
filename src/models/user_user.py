import os
ROOT = os.path.join('..', '..')
import sys
sys.path.append(ROOT)
#
import numpy as np
import pandas as pd
from src.metrics import (
    ml_precision_at_k, ml_recall_at_k, ml_f1_at_k
)
from src.models.base import BaseRecommender

class UserUserColaborativeFiltering(BaseRecommender):
    def __init__(self, ml_movies_df, ml_users_df):
        super().__init__(ml_movies_df, ml_users_df)
    
    def fit(self, ml_ratings_train_df):
        # Create a full user-item dataframe
        self.ratings_matrix = pd.DataFrame(index=ml_ratings_train_df["UserID"].unique(), columns=self.ml_movies_df["MovieID"].unique())
        # Create a user-item dataframe from training set and update the full one
        self.ratings_matrix.update(
            ml_ratings_train_df.pivot(values='Rating', index='UserID', columns='MovieID')
        )
        # Create user-user similarity df
        self.user_similarity_matrix = self.ratings_matrix.T.corr(method='pearson')
    
    def predict(self, user_id, n_recommendations, k_users=15):
        # If user_id is not in the user_similarity_matrix, return the most popular movies
        if user_id not in self.user_similarity_matrix.index:
            return self.ratings_matrix.sum(axis=0).sort_values(ascending=False).head(n_recommendations)
        # get most similar users series
        k_similar_users = self.user_similarity_matrix[user_id].sort_values(ascending=False).head(k_users)
        # identify unrated moves ids
        unrated_movies_ids = self.ratings_matrix.loc[user_id][self.ratings_matrix.loc[user_id].isna()].index
        # get unrated movies ratings from similar users
        k_similar_users_ratings = self.ratings_matrix.loc[k_similar_users.index, unrated_movies_ids]
        # Make dataframe of weights
        k_similar_users_weights = (~k_similar_users_ratings.isna() * np.array(k_similar_users_ratings))
        # Normalize by column
        k_similar_users_weights = k_similar_users_weights / (k_similar_users_weights.sum(axis=0) + 1e-12)
        # Weight ratings
        k_similar_users_ratings = k_similar_users_ratings * k_similar_users_weights
        # Get scores
        scores = k_similar_users_ratings.sum(axis=0).sort_values(ascending=False).head(n_recommendations)
        return scores