from src.constants import ML_SILVER_PATH, ML_SILVER_TABLES, BC_SILVER_PATH, BC_SILVER_TABLES
from pandas import read_parquet, DataFrame


def read_ml(root="../..") -> list[DataFrame]:
    """ml_complete, ml_users, ml_ratings, ml_movies, ml_genres"""
    return [read_parquet(f"{root}/{ML_SILVER_PATH}/{table}.parquet") for table in ML_SILVER_TABLES]


def read_bc(root="../..") -> list[DataFrame]:
    """bc_complete, bc_users, bc_ratings, bc_books"""
    return [read_parquet(f"{root}/{BC_SILVER_PATH}/{table}.parquet") for table in BC_SILVER_TABLES]


def ml_train_test_split(ml_ratings_df, min_user_test_samples, t_split_point=970_000_000):

    # Split data into train and test in T_SPLIT_POINT
    ml_ratings_train_df = ml_ratings_df[ml_ratings_df['Timestamp'] < t_split_point]
    ml_ratings_test_df = ml_ratings_df[ml_ratings_df['Timestamp'] >= t_split_point]

    # Keep only users in test that have at least MAX_K ratings
    users_with_enough_ratings = ml_ratings_test_df['UserID'].value_counts()
    users_with_enough_ratings = users_with_enough_ratings[users_with_enough_ratings >= min_user_test_samples]
    ml_ratings_test_df = ml_ratings_test_df[ml_ratings_test_df['UserID'].isin(users_with_enough_ratings.index)]

    # Filter the train data to include only those users who are also in the test data
    users_from_test = set(ml_ratings_test_df['UserID'].unique())
    users_from_train = set(ml_ratings_train_df['UserID'].unique())

    users_to_keep = users_from_test & users_from_train

    ml_ratings_train_df = ml_ratings_train_df[ml_ratings_train_df['UserID'].isin(users_to_keep)]
    ml_ratings_test_df = ml_ratings_test_df[ml_ratings_test_df['UserID'].isin(users_to_keep)]

    return ml_ratings_train_df, ml_ratings_test_df
