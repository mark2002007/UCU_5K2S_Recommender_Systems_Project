def ml_precision_at_k(k, rec_df, ml_ratings_test_df, user_id, verbose=True):
    ml_user_ratings_test_df = ml_ratings_test_df[ml_ratings_test_df["UserID"] == user_id]
    if len(ml_user_ratings_test_df) < k:
        if verbose:
            print(
                f"Warning: len(ml_user_ratings_test_df) < k for user with ID {user_id}!")
        if len(ml_user_ratings_test_df) == 0:
            if verbose:
                print(
                    f"Warning: len(ml_user_ratings_test_df) == 0 for user with ID {user_id}!")
            return 0
    topk = rec_df.head(k).copy()
    topk["is_relevant"] = topk["MovieID"].isin(
        ml_user_ratings_test_df["MovieID"])
    ml_precision_at_k = topk["is_relevant"].sum() / len(topk)
    return ml_precision_at_k


def ml_recall_at_k(k, rec_df, ml_ratings_test_df, user_id, verbose=True):
    ml_user_ratings_test_df = ml_ratings_test_df[ml_ratings_test_df["UserID"] == user_id]
    if len(ml_user_ratings_test_df) < k:
        if verbose:
            print(
                f"Warning: len(ml_user_ratings_test_df) < k for user with ID {user_id}!")
        if len(ml_user_ratings_test_df) == 0:
            if verbose:
                print(
                    f"Warning: len(ml_user_ratings_test_df) == 0 for user with ID {user_id}!")
            return 0
    topk = rec_df.head(k).copy()
    topk["is_relevant"] = topk["MovieID"].isin(
        ml_user_ratings_test_df["MovieID"])
    recall_at_k = topk["is_relevant"].sum() / len(ml_user_ratings_test_df)
    return recall_at_k


def ml_f1_at_k(k, rec_df, ml_ratings_test_df, user_id, precision_at_k=None, recall_at_k=None, verbose=True):
    if precision_at_k is None:
        precision_at_k = ml_precision_at_k(
            k, rec_df, ml_ratings_test_df, user_id, verbose)
    if recall_at_k is None:
        recall_at_k = ml_recall_at_k(
            k, rec_df, ml_ratings_test_df, user_id, verbose)
    return 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k + 1e-12)


def avg_precision_at_k(k, rec_df, ml_ratings_test_df, user_id, verbose=True):
    precision_sum = 0
    for i in range(1, k+1):
        precision_sum += ml_precision_at_k(i,
                                           rec_df, ml_ratings_test_df, user_id)
    return precision_sum/k
