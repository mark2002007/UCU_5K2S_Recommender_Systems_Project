import numpy as np
import pandas as pd
from src.models.base import BaseRecommender

class TwoStageRecommender(BaseRecommender):
    def __init__(self, ml_movies_df, ml_users_df, candidate_generator, ranker):
        super().__init__(ml_movies_df, ml_users_df)
        self.candidate_generator = candidate_generator
        self.ranker = ranker

    def fit(self, ml_ratings_train_df):
        self.candidate_generator.fit(ml_ratings_train_df)
        self.ranker.fit(ml_ratings_train_df)

    def predict(self, user_id, n_recommendations=10):
        predictions = self.candidate_generator.predict(user_id, n_recommendations=100).reset_index()
        predictions.columns = ["MovieID", "Score"]
        candidates, candidate_scores = predictions['MovieID'], predictions['Score']
        ranker_scores = self.ranker.predict(user_id, n_recommendations=len(candidates))
        
        ranker_scores = ranker_scores.reindex(candidates, fill_value=0)
        
        final_scores = 0.3 * candidate_scores + 0.7 * ranker_scores['Score'].values
        
        top_indices = np.argpartition(final_scores, -n_recommendations)[-n_recommendations:]
        top_candidates = candidates[top_indices]
        top_scores = final_scores[top_indices]
        
        return pd.DataFrame({
            'MovieID': top_candidates,
            'Score': top_scores
        }).set_index('MovieID')
