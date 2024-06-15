import os
ROOT = os.path.join('..', '..')
import sys
sys.path.append(ROOT)
from src.models.base import BaseRecommender
#   
class PopularityBasedRecommender(BaseRecommender):
    def __init__(self, ml_movies_df, ml_users_df):
        super().__init__(ml_movies_df, ml_users_df)
    
    def fit(self, ml_ratings_train_df):
        self.ml_ratings_train_df = ml_ratings_train_df
        self.scores = ml_ratings_train_df[['MovieID', 'Rating']].groupby('MovieID').sum()\
            .sort_values('Rating', ascending=False)
        self.scores.columns = ["Score"]
        self.scores["Score"] = self.scores["Score"] / self.scores["Score"].max()
    
    def predict(self, user_id, n_recommendations):
        user_rated_movies_idx = self.ml_ratings_train_df[self.ml_ratings_train_df["UserID"] == user_id]["MovieID"].values
        scores = self.scores.drop(user_rated_movies_idx, errors='ignore').head(n_recommendations)
        return scores