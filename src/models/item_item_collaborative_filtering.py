import pandas as pd
from src.models.base import BaseRecommender
from sklearn.metrics.pairwise import cosine_similarity


class ItemItemCollaborativeFiltering(BaseRecommender):
    def __init__(self, ml_movies_df, ml_users_df):
        super().__init__(ml_movies_df, ml_users_df)
        self.similarity_matrix = None
        self.ratings_matrix = None

    def fit(self, ml_ratings_train_df: pd.DataFrame):
        # Initialize a full ratings matrix with indices as UserIDs and columns as MovieIDs
        self.ratings_matrix = pd.DataFrame(
            index=self.ml_users_df['UserID'].unique(),
            columns=self.ml_movies_df['MovieID'].unique()
        )

        # Update the full matrix with the actual ratings from the training set
        training_matrix = ml_ratings_train_df.pivot(values="Rating", index="UserID", columns="MovieID")
        self.ratings_matrix.update(training_matrix)

        # Fill NaNs with zeros for computational purposes
        self.ratings_matrix.fillna(0, inplace=True)

        # Calculate the mean only for originally non-zero ratings (excluding zeros filled for NaN)
        mask = training_matrix.notna()
        means = training_matrix.where(mask).mean(axis=1).fillna(0)

        # Normalize the matrix by subtracting the mean ratings only for non-zero entries
        normalized_mat = self.ratings_matrix.where(mask).sub(means, axis=0).fillna(0)

        # Transpose the matrix to get a movie-user matrix for similarity calculation
        movie_user_mat = normalized_mat.T

        # Compute cosine similarity between movies
        similarity_matrix = cosine_similarity(movie_user_mat)

        # Create a DataFrame for the similarity matrix with movie IDs as both row and column indices
        self.similarity_matrix = pd.DataFrame(similarity_matrix, index=movie_user_mat.index, columns=movie_user_mat.index)

    def predict(self, user_id: int, n_recommendations=10):
        if user_id not in self.ratings_matrix.index:
            raise ValueError("User ID not found in the ratings matrix.")

        # Identify movies this user has rated
        rated_movies = self.ratings_matrix.loc[user_id][self.ratings_matrix.loc[user_id] > 0].index

        # Collect candidates based on similarity to rated movies
        candidate_movies = set()
        for movie_id in rated_movies:
            # Get the top N similar movies for each rated movie
            top_similar = self.similarity_matrix[movie_id].sort_values(ascending=False).head(n_recommendations + 1)
            candidate_movies.update(top_similar.index)

        # Exclude already rated movies from candidates
        candidate_movies.difference_update(rated_movies)

        # Predict ratings for candidate movies
        predictions = {}
        for movie_id in candidate_movies:
            similar_movies = self.similarity_matrix.loc[rated_movies, movie_id]
            similarities = similar_movies[similar_movies > 0]
            user_ratings = self.ratings_matrix.loc[user_id, similar_movies.index]

            # Compute the weighted average prediction
            weighted_scores = user_ratings * similarities
            if similarities.sum() > 0:
                predictions[movie_id] = weighted_scores.sum() / similarities.sum()
            else:
                predictions[movie_id] = 0

        # Calculate overall movie popularity as the mean rating for each movie
        movie_popularity = self.ratings_matrix.mean(axis=0)

        # Convert predictions to DataFrame, set MovieID as index, sort and return top N
        predictions_df = pd.DataFrame(list(predictions.items()), columns=['MovieID', 'PredictedRating'])
        predictions_df.set_index('MovieID', inplace=True)
        predictions_df.sort_values(by='PredictedRating', ascending=False, inplace=True)

        if predictions_df.empty:
            # Recommend the most popular movies as a fallback
            most_popular_movies = movie_popularity.sort_values(ascending=False).head(n_recommendations).index
            return pd.DataFrame({'PredictedRating': movie_popularity[most_popular_movies].values}, index=most_popular_movies)

        return predictions_df.head(n_recommendations)
