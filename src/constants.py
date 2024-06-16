import os

ML_BRONZE_PATH = os.path.join("data", "bronze", "ml-1m")
BC_BRONZE_PATH = os.path.join("data", "bronze", "bc")

ML_SILVER_PATH = os.path.join("data", "silver", "ml-1m")
BC_SILVER_PATH = os.path.join("data", "silver", "bc")

ML_SILVER_TABLES = "ml_complete", "ml_users", "ml_ratings", "ml_movies", "ml_genres"
BC_SILVER_TABLES = "bc_complete", "bc_users", "bc_ratings", "bc_books"
