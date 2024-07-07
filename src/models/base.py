from src.metrics import ml_precision_at_k, ml_recall_at_k, ml_f1_at_k, avg_precision_at_k, reciprocal_rank, NDCG
from tqdm import tqdm
import pandas as pd
import sys
import os

ROOT = os.path.join("..", "..")

sys.path.append(ROOT)
#


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
        history = {"precision@k": [], "recall@k": [],
                   "f1@k": [], "average_precision": [],
                   "reciprocal_rank": [], "NDCG": []}
        with tqdm(total=len(users)) as pbar:
            sum_f1 = 0
            cnt_f1 = 0
            for user_id in users:
                recommendations = self.predict(
                    user_id=user_id, n_recommendations=k)
                recommendations = recommendations.reset_index()
                recommendations.columns = ["MovieID", "Score"]
                precision_at_k = ml_precision_at_k(
                    k=k,
                    rec_df=recommendations,
                    ml_ratings_test_df=ml_ratings_test_df,
                    user_id=user_id,
                    verbose=verbose,
                )
                recall_at_k = ml_recall_at_k(
                    k=k,
                    rec_df=recommendations,
                    ml_ratings_test_df=ml_ratings_test_df,
                    user_id=user_id,
                    verbose=verbose,
                )
                f1_at_k = ml_f1_at_k(
                    k=k,
                    rec_df=recommendations,
                    ml_ratings_test_df=ml_ratings_test_df,
                    user_id=user_id,
                    precision_at_k=precision_at_k,
                    recall_at_k=recall_at_k,
                    verbose=verbose,
                )
                average_precision = avg_precision_at_k(
                    k,
                    rec_df=recommendations,
                    ml_ratings_test_df=ml_ratings_test_df,
                    user_id=user_id,
                    verbose=verbose,
                )
                RR = reciprocal_rank(
                    rec_df=recommendations,
                    ml_ratings_test_df=ml_ratings_test_df,
                    user_id=user_id
                )
                NDCG_metric = NDCG(
                    rec_df=recommendations,
                    ml_ratings_test_df=ml_ratings_test_df,
                    user_id=user_id
                )
                history["precision@k"].append(precision_at_k)
                history["recall@k"].append(recall_at_k)
                history["f1@k"].append(f1_at_k)
                history["average_precision"].append(average_precision)
                history['reciprocal_rank'].append(RR)
                history['NDCG'].append(NDCG_metric)

                sum_f1 += f1_at_k
                cnt_f1 += 1
                pbar.update(1)
        return history
