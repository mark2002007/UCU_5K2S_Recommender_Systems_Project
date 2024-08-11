import os
ROOT = os.path.join("..", "..")
import sys
sys.path.append(ROOT)
#
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from src.models.base import BaseRecommender


#
class SVDCollaborativeFiltering(BaseRecommender):
    def __init__(self, ml_movies_df, ml_users_df):
        super().__init__(ml_movies_df, ml_users_df)

    def fit(self, ml_ratings_train_df, n_features=10):
        utility_matrix_df = pd.DataFrame(
            index=ml_ratings_train_df["UserID"].unique(), columns=ml_ratings_train_df["MovieID"].unique()
        )
        utility_matrix_df.update(ml_ratings_train_df.pivot(values="Rating", index="UserID", columns="MovieID"))
        user_ids = list(utility_matrix_df.index)
        movie_ids = list(utility_matrix_df.columns)
        
        global_mean = utility_matrix_df.mean().mean()
        utility_matrix_df = utility_matrix_df.fillna(global_mean)
        utility_matrix_df = utility_matrix_df - global_mean
        utility_matrix = utility_matrix_df.to_numpy()
        U, sigma, Vt = svds(utility_matrix, k=n_features)
        sigma = np.diag(sigma)
        predicted_ratings = np.dot(np.dot(U, sigma), Vt) + global_mean
        self.predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=movie_ids, index=user_ids)
        
        self.ml_ratings_train_df = ml_ratings_train_df

    def predict(self, user_id, n_recommendations):
        user_predicted_ratings = self.predicted_ratings_df.loc[user_id].sort_values(ascending=False)
        user_rated_movies_idx = self.ml_ratings_train_df[self.ml_ratings_train_df["UserID"] == user_id]["MovieID"].values
        recommendations = user_predicted_ratings.drop(user_rated_movies_idx, errors='ignore').head(n_recommendations)
        recommendations.columns = ["Score"]
        return recommendations