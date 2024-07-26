import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from src.models.base import BaseRecommender

class LTRLinearRegressionRecommender(BaseRecommender):
    def __init__(self, ml_movies_df: pd.DataFrame, ml_users_df: pd.DataFrame):
        super().__init__(ml_movies_df, ml_users_df)
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.user_mean_ratings = None
        self.movie_mean_ratings = None
        self.global_mean_rating = None

    def fit(self, ml_ratings_train_df: pd.DataFrame):
        # Calculate mean ratings for users and movies
        self.user_mean_ratings = ml_ratings_train_df.groupby('UserID')['Rating'].mean()
        self.movie_mean_ratings = ml_ratings_train_df.groupby('MovieID')['Rating'].mean()
        self.global_mean_rating = ml_ratings_train_df['Rating'].mean()
        
        # Prepare features
        X = ml_ratings_train_df[['UserID', 'MovieID']].values
        y = ml_ratings_train_df['Rating'].values
        
        # Add user and movie mean ratings as features
        user_means = self.user_mean_ratings[ml_ratings_train_df['UserID']].values
        movie_means = self.movie_mean_ratings[ml_ratings_train_df['MovieID']].values
        X = np.column_stack([X, user_means, movie_means])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled, y)
        
        self.ml_ratings_train_df = ml_ratings_train_df

    def predict(self, user_id: int, n_recommendations: int):
        # Generate candidates (all movies not rated by the user)
        user_rated_movies = self.ml_ratings_train_df[self.ml_ratings_train_df['UserID'] == user_id]['MovieID'].unique()
        candidate_movies = self.ml_movies_df[~self.ml_movies_df['MovieID'].isin(user_rated_movies)]['MovieID'].values
        
        # Prepare features for prediction
        user_mean = self.user_mean_ratings.get(user_id, self.global_mean_rating)
        movie_means = np.array([self.movie_mean_ratings.get(movie, self.global_mean_rating) for movie in candidate_movies])
        
        X_pred = np.column_stack([
            np.full(len(candidate_movies), user_id),
            candidate_movies,
            np.full(len(candidate_movies), user_mean),
            movie_means
        ])
        
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Predict ratings
        predicted_ratings = self.model.predict(X_pred_scaled)
        
        # Create DataFrame with predictions
        recommendations = pd.DataFrame({
            'MovieID': candidate_movies,
            'Rating': predicted_ratings
        })
        
        # Sort and return top n recommendations
        return recommendations.sort_values('Rating', ascending=False).head(n_recommendations).set_index('MovieID')