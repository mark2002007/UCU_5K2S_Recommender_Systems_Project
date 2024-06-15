import os

ROOT = os.path.join("..", "..")
import sys

sys.path.append(ROOT)
#
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.metrics import ml_precision_at_k, ml_recall_at_k, ml_f1_at_k
from src.models.base import BaseRecommender


class PageRankRecommender(BaseRecommender):
    def __init__(self, ml_movies_df, ml_users_df):
        super().__init__(ml_movies_df, ml_users_df)
        self.A_norm = None
        if os.path.exists("A_norm.npy"):
            self.A_norm = np.load("A_norm.npy")

    def fit(self, ml_ratings_train_df):
        if self.A_norm is None:
            print("Initializing adjacency matrix...")
            self._init_adjacency_matrix(ml_ratings_train_df)
            np.save("A_norm.npy", self.A_norm)
        self.pagerank_scores = self._pagerank(self.A_norm)
        self.ml_ratings_train_df = ml_ratings_train_df

        movies_ids = self.ml_movies_df["MovieID"].values
        self.scores = pd.DataFrame(self.pagerank_scores, index=movies_ids, columns=["Score"]).sort_values(
            "Score", ascending=False
        )

    def predict(self, user_id, n_recommendations):
        user_rated_movies_idx = self.ml_ratings_train_df[self.ml_ratings_train_df["UserID"] == user_id][
            "MovieID"
        ].values
        scores = self.scores.drop(user_rated_movies_idx, errors="ignore").head(n_recommendations)
        return scores

    def _init_adjacency_matrix(self, ml_ratings_train_df):
        movies_ids = self.ml_movies_df["MovieID"].values
        A_df = pd.DataFrame(0, index=movies_ids, columns=movies_ids)
        user_groups = ml_ratings_train_df.sort_values("Rating", ascending=False).groupby("UserID")
        for _, user_group in tqdm(user_groups):
            ratings = user_group["Rating"].values
            movie_ids = user_group["MovieID"].values
            diff_matrix = -np.subtract.outer(ratings, ratings)
            diff_matrix = np.maximum(diff_matrix, 0)
            A_df.loc[movie_ids, movie_ids] += diff_matrix
        A = A_df.values
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Handle rows with all zeros
        A_norm = A / row_sums
        self.A_norm = A_norm

    def _pagerank(self, A, d=0.85, eps=1e-8, max_iter=100):
        N = A.shape[0]
        p = np.ones(N) / N
        for _ in tqdm(range(max_iter)):
            p_new = (1 - d) / N + d * A.T.dot(p)
            if np.linalg.norm(p_new - p) < eps:
                break
            p = p_new
        return p
