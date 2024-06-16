import os

ROOT = os.path.join("..", "..")
import sys

sys.path.append(ROOT)
#
import pandas as pd
from tqdm import tqdm
from src.metrics import ml_precision_at_k, ml_recall_at_k, ml_f1_at_k


class BaseRecommender:
    def __init__(self, ml_movies_df: pd.DataFrame, ml_users_df: pd.DataFrame):
        self.ml_movies_df = ml_movies_df
        self.ml_users_df = ml_users_df

    def fit(self, ml_ratings_train_df: pd.DataFrame):
        raise NotImplementedError

    def predict(self, user_id: int, n_recommendations: int):
        raise NotImplementedError

    def evaluate(self, ml_ratings_test_df: pd.DataFrame, k: int, verbose=False):
        users = ml_ratings_test_df["UserID"].unique()[0:2500]
        history = {"precision@k": [], "recall@k": [], "f1@k": []}
        with tqdm(total=len(users)) as pbar:
            sum_f1 = 0
            cnt_f1 = 0
            for user_id in users:
                recommendations = self.predict(user_id=user_id, n_recommendations=k)
                recommendations = recommendations.reset_index()
                recommendations.columns = ["MovieID", "Score"]
                precision_at_k = ml_precision_at_k(
                    k=k,
                    rec_df=recommendations,
                    ml_ratings_test_df=ml_ratings_test_df,
                    user_id=user_id,
                )
                recall_at_k = ml_recall_at_k(
                    k=k,
                    rec_df=recommendations,
                    ml_ratings_test_df=ml_ratings_test_df,
                    user_id=user_id,
                )
                f1_at_k = ml_f1_at_k(
                    k=k,
                    rec_df=recommendations,
                    ml_ratings_test_df=ml_ratings_test_df,
                    user_id=user_id,
                    precision_at_k=precision_at_k,
                    recall_at_k=recall_at_k,
                )
                history["precision@k"].append(precision_at_k)
                history["recall@k"].append(recall_at_k)
                history["f1@k"].append(f1_at_k)

                sum_f1 += f1_at_k
                cnt_f1 += 1
                if verbose:
                    pbar.set_postfix_str(f"Avg F1@K: {sum_f1 / cnt_f1:.2f}")
                pbar.update(1)
        return history
