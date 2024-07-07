import pandas as pd
from funk_svd import SVD
from src.models.base import BaseRecommender

class FunkSVDCollaborativeFiltering(BaseRecommender):
    def __init__(self, ml_movies_df: pd.DataFrame, ml_users_df: pd.DataFrame, n_factors=20, lr=0.005, reg=0.02, n_epochs=100):
        super().__init__(ml_movies_df, ml_users_df)
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.model = SVD(n_factors=self.n_factors, lr=self.lr, reg=self.reg, n_epochs=self.n_epochs)
        
    def fit(self, ml_ratings_train_df: pd.DataFrame):
        # Rename columns to match expected names by funk-svd
        ml_ratings_train_df = ml_ratings_train_df.rename(columns={"UserID": "u_id", "MovieID": "i_id", "Rating": "rating"})
        self.model.fit(ml_ratings_train_df)

    def predict(self, user_id: int, n_recommendations: int):
        # Get all movie IDs
        all_movie_ids = self.ml_movies_df['MovieID'].values
        
        # Create a DataFrame for prediction
        pred_df = pd.DataFrame({
            'u_id': [user_id] * len(all_movie_ids),
            'i_id': all_movie_ids
        })
        
        # Predict the ratings for all movies for the given user
        pred_df['Score'] = self.model.predict(pred_df)
        
        # Select only the 'i_id' and 'Score' columns and rename 'i_id' to 'MovieID'
        recommendations = pred_df[['i_id', 'Score']].rename(columns={'i_id': 'MovieID'})
        
        # Sort the DataFrame by the predicted ratings in descending order
        recommendations = recommendations.sort_values(by='Score', ascending=False).head(n_recommendations)
        
        return recommendations.set_index('MovieID')