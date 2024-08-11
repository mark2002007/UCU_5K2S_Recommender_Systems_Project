import os
ROOT = os.path.join("..", "..")
import sys
sys.path.append(ROOT)
#
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.models.base import BaseRecommender


class SVDALSCollaborativeFiltering(BaseRecommender):
    def __init__(self, ml_movies_df, ml_users_df):
        super().__init__(ml_movies_df, ml_users_df)

    def fit(self, ml_ratings_train_df, n_factors=10, regularization=0.1, iterations=10):
        utility_matrix_df = pd.DataFrame(
            index=ml_ratings_train_df["UserID"].unique(), columns=ml_ratings_train_df["MovieID"].unique()
        )
        utility_matrix_df.update(ml_ratings_train_df.pivot(values="Rating", index="UserID", columns="MovieID"))
        utility_matrix = utility_matrix_df.fillna(0)
        user_ids = utility_matrix.index
        movie_ids = utility_matrix.columns

        utility_matrix_sparse = csr_matrix(utility_matrix.values)
        
        n_users, n_items = utility_matrix.shape
        self.user_factors = np.random.normal(scale=1./n_factors, size=(n_users, n_factors))
        self.item_factors = np.random.normal(scale=1./n_factors, size=(n_items, n_factors))

        for iteration in range(iterations):
            for u in range(n_users):
                self.user_factors[u, :] = self.solve(utility_matrix_sparse[u, :].toarray().reshape(-1), self.item_factors, regularization)
            for i in range(n_items):
                self.item_factors[i, :] = self.solve(utility_matrix_sparse[:, i].toarray().reshape(-1), self.user_factors, regularization)
        
        self.user_ids = user_ids
        self.movie_ids = movie_ids
        self.ml_ratings_train_df = ml_ratings_train_df

    def solve(self, ratings, factors, regularization):
        yty = factors.T @ factors
        lambda_i = np.eye(yty.shape[0]) * regularization
        a = yty + lambda_i
        b = ratings @ factors
        return np.linalg.solve(a, b)

    def predict(self, user_id, n_recommendations):
        user_idx = self.user_ids.get_loc(user_id)
        user_ratings = np.dot(self.user_factors[user_idx, :], self.item_factors.T)
        predicted_ratings_df = pd.DataFrame(user_ratings, index=self.movie_ids, columns=["Rating"])
        user_rated_movies_idx = self.ml_ratings_train_df[self.ml_ratings_train_df["UserID"] == user_id]["MovieID"].values
        recommendations = predicted_ratings_df.drop(user_rated_movies_idx, errors='ignore').sort_values(by="Rating", ascending=False).head(n_recommendations)
        
        recommendations.columns = ["Score"]
        return recommendations