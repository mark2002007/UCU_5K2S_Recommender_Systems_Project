import os
ROOT = os.path.join('..', '..')
import sys
sys.path.append(ROOT)
import polars as pl
#
def ml_popularity_based_recommendation(ml_ratings_train_df, ml_movies_df, n_recommendations):
    """
    Recommends most popular movies. Thats it
    """
    return ml_ratings_train_df.join(ml_movies_df, on="MovieID") \
        .group_by("MovieID") \
        .agg(
            pl.sum("Rating").alias("score")
        ).join(ml_movies_df, on="MovieID") \
        .sort("score", descending=True) \
        .head(n_recommendations)