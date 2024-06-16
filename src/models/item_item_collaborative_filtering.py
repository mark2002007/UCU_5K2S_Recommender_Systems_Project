import pandas as pd
from src.models.base import BaseRecommender
from sklearn.metrics.pairwise import cosine_similarity


class ItemItemCollaborativeFiltering(BaseRecommender):
    def __init__(self, ml_movies_df, ml_users_df):
        super().__init__(ml_movies_df, ml_users_df)
        self.similarity_matrix = None
        self.ratings_matrix = None

    def fit(self, ml_ratings_train_df: pd.DataFrame):
        # Create a ratings matrix
        utility_df = pd.DataFrame(
            index=ml_ratings_train_df["UserID"].unique(), columns=ml_ratings_train_df["MovieID"].unique()
        )
        # Create a user-item dataframe from training set and update the full one
        utility_df.update(ml_ratings_train_df.pivot(values="Rating", index="UserID", columns="MovieID"))
        # Create user-user similarity df
        self.similarity_df = utility_df.corr(method="pearson")
        self.ml_ratings_train_df = ml_ratings_train_df

    def predict(self, user_id: int, n_recommendations=10, n_similar=10):
        # Get user info
        user_ratings = self.ml_ratings_train_df[self.ml_ratings_train_df['UserID'] == user_id]
        user_movies = user_ratings['MovieID'].values
        user_ratings = user_ratings['Rating'].values
        # For each movie, find n_simmilar movies and add to recommendations
        recommendations = None
        for movie_id in user_movies:
            similar_movies = self.similarity_df[movie_id].drop(user_movies).sort_values(ascending=False).head(n_similar)
            similar_movies = similar_movies.reset_index()
            similar_movies.columns = ['MovieID', 'Score']
            recommendations = pd.concat([recommendations, similar_movies], axis=0) if recommendations is not None else similar_movies
        # Group by movie and sum scores
        recommendations = recommendations.groupby('MovieID').sum().sort_values('Score', ascending=False).head(n_recommendations)

        return recommendations
