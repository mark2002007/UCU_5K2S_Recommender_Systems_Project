import os

ROOT = os.path.join("..", "..")
import sys

sys.path.append(ROOT)
#
import numpy as np
import pandas as pd
from src.metrics import ml_precision_at_k, ml_recall_at_k, ml_f1_at_k
from src.models.base import BaseRecommender


class UserUserColaborativeFiltering(BaseRecommender):
    def __init__(self, ml_movies_df, ml_users_df):
        super().__init__(ml_movies_df, ml_users_df)

    def fit(self, ml_ratings_train_df):
        # Create a full user-item dataframe
        self.utility_df = pd.DataFrame(
            index=ml_ratings_train_df["UserID"].unique(), columns=self.ml_movies_df["MovieID"].unique()
        )
        # Create a user-item dataframe from training set and update the full one
        self.utility_df.update(ml_ratings_train_df.pivot(values="Rating", index="UserID", columns="MovieID"))
        # Create user-user similarity df
        self.user_similarity_df = self.utility_df.T.corr(method="pearson")

    def predict(self, user_id, n_recommendations, k_users=15):
        # If user_id is not in the user_similarity_df, return the most popular movies
        if user_id not in self.user_similarity_df.index:
            return self.utility_df.sum(axis=0).sort_values(ascending=False).head(n_recommendations)
        # Get most similar users series
        k_similar_users = self.user_similarity_df[user_id].sort_values(ascending=False).head(k_users)
        # Identify unrated moves ids
        unrated_movies_ids = self.utility_df.loc[user_id][self.utility_df.loc[user_id].isna()].index
        # Get unrated movies ratings from similar users
        k_similar_users_ratings = self.utility_df.loc[k_similar_users.index, unrated_movies_ids]
        # Make dataframe of weights
        k_similar_users_weights = ~k_similar_users_ratings.isna() * np.array(k_similar_users_ratings)
        # Normalize by column
        k_similar_users_weights = k_similar_users_weights / (k_similar_users_weights.sum(axis=0) + 1e-12)
        # Weight ratings
        k_similar_users_ratings = k_similar_users_ratings * k_similar_users_weights
        # Get scores
        scores = k_similar_users_ratings.sum(axis=0).sort_values(ascending=False).head(n_recommendations)
        scores.columns = ["Score"]
        return scores
