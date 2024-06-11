import polars as pl

def ml_precision_at_k(k, rec_df, ml_ratings_test_df, user_id):
    ml_user_ratings_test_df = ml_ratings_test_df.filter(pl.col("UserID") == user_id)
    topk = rec_df.head(k).with_columns([
        pl.col("MovieID").is_in(ml_user_ratings_test_df.select("MovieID").collect()).alias("is_relevant")
    ])
    top_k_len = topk.select(pl.len()).collect().item()
    ml_precision_at_k = topk.select(pl.col("is_relevant").sum()).collect().item() / top_k_len
    return ml_precision_at_k

def ml_recall_at_k(k, rec_df, ml_ratings_test_df, user_id):
    ml_user_ratings_test_df = ml_ratings_test_df.filter(pl.col("UserID") == user_id)
    topk = rec_df.head(k).with_columns([
        pl.col("MovieID").is_in(ml_user_ratings_test_df.select("MovieID").collect()).alias("is_relevant")
    ])
    ml_user_ratings_test_df_len = ml_user_ratings_test_df.select(pl.len()).collect().item()
    if ml_user_ratings_test_df_len == 0: # Not sure about this
        return 0
    recall_at_k = topk.select(pl.col("is_relevant").sum()).collect().item() / ml_user_ratings_test_df_len
    return recall_at_k

def ml_f1_at_k(k, rec_df, ml_ratings_test_df, user_id):
    precision = ml_precision_at_k(k, rec_df, ml_ratings_test_df, user_id)
    recall = ml_recall_at_k(k, rec_df, ml_ratings_test_df, user_id)
    return 2 * (precision * recall) / (precision + recall + 1e-12)