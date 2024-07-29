import numpy as np
import pandas as pd
from src.models.base import BaseRecommender

class MarkovChainRecommender(BaseRecommender):
    def __init__(self, ml_movies_df: pd.DataFrame, ml_users_df: pd.DataFrame):
        super().__init__(ml_movies_df, ml_users_df)
        self.transition_matrix = None
        self.movie_index = None
        self.index_movie = None

    def fit(self, ml_ratings_train_df: pd.DataFrame):
        # Create a list of unique movies
        unique_movies = self.ml_movies_df['MovieID'].unique()
        n_movies = len(unique_movies)

        # Map movies to indices and vice versa
        self.movie_index = {movie: idx for idx, movie in enumerate(unique_movies)}
        self.index_movie = {idx: movie for movie, idx in self.movie_index.items()}

        # Initialize transition matrix
        self.transition_matrix = np.zeros((n_movies, n_movies))

        # Fill the transition matrix based on user interaction sequences
        for user_id in ml_ratings_train_df['UserID'].unique():
            user_data = ml_ratings_train_df[ml_ratings_train_df['UserID'] == user_id]
            user_data = user_data.sort_values('Timestamp')
            user_movies = user_data['MovieID'].values

            for i in range(len(user_movies) - 1):
                current_movie = user_movies[i]
                next_movie = user_movies[i + 1]
                self.transition_matrix[self.movie_index[current_movie], self.movie_index[next_movie]] += 1

        # Normalize the transition matrix to get probabilities
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        
        self.ml_ratings_train_df = ml_ratings_train_df

    def predict(self, user_id: int, n_recommendations: int):
        # Get the last watched movie by the user from the ratings data
        user_data = self.ml_users_df[self.ml_users_df['UserID'] == user_id]
        
        if user_data.empty:
            return pd.DataFrame(columns=['MovieID', 'Score'])
        
        # Check if 'Timestamp' column is present in ml_users_df
        if 'Timestamp' in user_data.columns:
            last_movie_id = user_data.sort_values('Timestamp').iloc[-1]['MovieID']
        else:
            # Use the training ratings data to get the last watched movie
            user_ratings_data = self.ml_ratings_train_df[self.ml_ratings_train_df['UserID'] == user_id]
            if user_ratings_data.empty:
                return pd.DataFrame(columns=['MovieID', 'Score'])
            last_movie_id = user_ratings_data.sort_values('Timestamp').iloc[-1]['MovieID']

        if last_movie_id not in self.movie_index:
            return pd.DataFrame(columns=['MovieID', 'Score'])

        # Get the transition probabilities from the last watched movie
        movie_idx = self.movie_index[last_movie_id]
        next_movie_probs = self.transition_matrix[movie_idx]

        # Get the top-n recommendations
        top_n_indices = next_movie_probs.argsort()[-n_recommendations:][::-1]
        recommendations = [(self.index_movie[idx], next_movie_probs[idx]) for idx in top_n_indices]

        recommendations_df = pd.DataFrame(recommendations, columns=['MovieID', 'Score'])
        return recommendations_df.set_index('MovieID')