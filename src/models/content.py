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
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedFiltering(BaseRecommender):
    def __init__(self, ml_movies_df, ml_users_df):
        super().__init__(ml_movies_df, ml_users_df)

    def fit(self, ml_ratings_train_df):
        # Select features for simmilarity
        features = [col for col in self.ml_movies_df.columns if col.startswith('Is')]
        feature_matrix = self.ml_movies_df[features].values
        # Calculate simmilarity matrix
        similarity_matrix = cosine_similarity(feature_matrix, feature_matrix)
        self.similarity_df = pd.DataFrame(similarity_matrix, index=self.ml_movies_df['MovieID'], columns=self.ml_movies_df['MovieID'])
        self.ml_ratings_train_df = ml_ratings_train_df

    def predict(self, user_id, n_recommendations, n_simmilar=10):
        # Get user info
        user_id = self.ml_ratings_train_df['UserID'].values[0]
        user_ratings = self.ml_ratings_train_df[self.ml_ratings_train_df['UserID'] == user_id]
        user_movies = user_ratings['MovieID'].values
        user_ratings = user_ratings['Rating'].values
        # For each movie, find n_simmilar movies and add to recommendations
        recommendations = None
        for movie_id in user_movies:
            similar_movies = self.similarity_df[movie_id].drop(user_movies).sort_values(ascending=False).head(n_simmilar+1).tail(n_simmilar)
            similar_movies = similar_movies.reset_index()
            similar_movies.columns = ['MovieID', 'Score']
            recommendations = pd.concat([recommendations, similar_movies]) if recommendations is not None else similar_movies
        # Group by movie and mean score
        recommendations.groupby('MovieID').mean().sort_values('Score', ascending=False).head(10)
        
        return recommendations.set_index('MovieID')