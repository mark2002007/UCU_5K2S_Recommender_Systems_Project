import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm
from src.models.base import BaseRecommender

class SVDGradientDescentRecommender(BaseRecommender):
    def __init__(self, ml_movies_df, ml_users_df):
        super().__init__(ml_movies_df, ml_users_df)

    def fit(self, ml_ratings_train_df, n_factors=10, learning_rate=0.01, regularization=0.1, iterations=10):
        self.utility_matrix = ml_ratings_train_df.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
        self.user_ids = self.utility_matrix.index
        self.movie_ids = self.utility_matrix.columns

        self.n_users, self.n_items = self.utility_matrix.shape
        self.n_factors = n_factors

        # Initialize user and item factors
        self.user_factors = np.random.normal(scale=1./self.n_factors, size=(self.n_users, self.n_factors))
        self.item_factors = np.random.normal(scale=1./self.n_factors, size=(self.n_items, self.n_factors))

        # Convert utility matrix to a sparse matrix for efficiency
        self.utility_matrix_sparse = csr_matrix(self.utility_matrix.values)

        # Training process
        for iteration in tqdm(range(iterations)):
            self.gradient_descent(learning_rate, regularization)
            mse = self.compute_mse()
            if (iteration + 1) % 1 == 0:
                print(f"Iteration {iteration + 1}: MSE = {mse}")

        self.ml_ratings_train_df = ml_ratings_train_df

    def gradient_descent(self, learning_rate, regularization):
        for i, j in zip(*self.utility_matrix_sparse.nonzero()):
            rating = self.utility_matrix_sparse[i, j]
            prediction = self.user_factors[i, :].dot(self.item_factors[j, :].T)
            error = rating - prediction

            # Update user and item factors
            self.user_factors[i, :] += learning_rate * (error * self.item_factors[j, :] - regularization * self.user_factors[i, :])
            self.item_factors[j, :] += learning_rate * (error * self.user_factors[i, :] - regularization * self.item_factors[j, :])

    def compute_mse(self):
        mse = 0
        for i, j in zip(*self.utility_matrix_sparse.nonzero()):
            rating = self.utility_matrix_sparse[i, j]
            prediction = self.user_factors[i, :].dot(self.item_factors[j, :].T)
            mse += (rating - prediction) ** 2
        mse /= self.utility_matrix_sparse.nnz
        return mse

    def predict(self, user_id, n_recommendations):
        user_idx = self.user_ids.get_loc(user_id)
        user_ratings = np.dot(self.user_factors[user_idx, :], self.item_factors.T)
        predicted_ratings_df = pd.DataFrame(user_ratings, index=self.movie_ids, columns=["Rating"])
        user_rated_movies_idx = self.ml_ratings_train_df[self.ml_ratings_train_df["UserID"] == user_id]["MovieID"].values
        recommendations = predicted_ratings_df.drop(user_rated_movies_idx, errors='ignore').sort_values(by="Rating", ascending=False).head(n_recommendations)
        
        recommendations.columns = ["Score"]
        return recommendations