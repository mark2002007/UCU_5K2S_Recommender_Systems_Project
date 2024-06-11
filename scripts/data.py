import os
ROOT = os.path.join('..', '..')
import sys
sys.path.append(ROOT)
#
from src.constants import (
    ML_MOVIES_PATH, ML_RATINGS_PATH, ML_USERS_PATH,
    BC_BOOKS_PATH, BC_RATINGS_PATH, BC_USERS_PATH
)
import polars as pl

#############################################################################MOVIE LENS DATASET

# 1) Read lines and split them by "::"
# 2) Select the columns and cast them to the correct types
ml_ocupations = [
    "other",
	"academic/educator",
	"artist",
	"clerical/admin",
	"college/grad student",
	"customer service",
	"doctor/health care",
	"executive/managerial",
	"farmer",
	"homemaker",
	"K-12 student",
	"lawyer",
	"programmer",
	"retired",
	"sales/marketing",
	"scientist",
	"self-employed",
	"technician/engineer",
	"tradesman/craftsman",
	"unemployed",
	"writer",
]
ml_occupation_map = {i: occupation for i, occupation in enumerate(ml_ocupations)}
ml_users_df = pl.scan_csv(os.path.join(ROOT, ML_USERS_PATH), has_header=False, truncate_ragged_lines=True, encoding="utf8-lossy").select([
    pl.col("column_1").str.split("::")
]).select([
    pl.col("column_1").list.get(0).alias("UserID").cast(pl.Int32),
    pl.col("column_1").list.get(1).alias("Gender"),
    pl.col("column_1").list.get(2).alias("Age").cast(pl.Int32),
    pl.col("column_1").list.get(3).alias("Occupation").cast(pl.Int32).map_dict(ml_occupation_map),
    pl.col("column_1").list.get(4).alias("Zip-code"),
])

# 1) Read lines and split them by "::"
# 2) Select the columns and cast them to the correct types
# 3) Extract the year from the title column
# 4) If year is not null, remove it from the title
#   If the title is "Toy Story (1995)", the title should be "Toy Story" and the year should be 1995. 
#   If the title does not have a year, the year should be null.
ml_movies_df = pl.scan_csv(os.path.join(ROOT, ML_MOVIES_PATH), has_header=False, truncate_ragged_lines=True, encoding="utf8-lossy").select([
    pl.col("column_1").str.split("::")
]).select([
    pl.col("column_1").list.get(0).alias("MovieID").cast(pl.Int32),
    pl.col("column_1").list.get(1).alias("Title"),
    pl.col("column_1").list.get(2).str.split("|").alias("Genres")
]).with_columns([
    pl.col("Title").str.extract(r"\((\d{4})\)$").alias("Year").cast(pl.Int32),
]).with_columns([
    # If Year is not null, remove it from the title
    pl.when(pl.col("Year").is_null()).then(pl.col("Title")).otherwise(
        pl.col("Title").str.slice(0, pl.col("Title").str.find(" \((\d{4})\)$"))
    ).alias("Title")
])
# 4) Get possible genres
ml_genres = ml_movies_df.select(pl.col("Genres").explode().unique()).collect()
# 5) Make dummy variable for each genre
ml_movies_df = ml_movies_df.with_columns([
    pl.col("Genres").list.contains(genre[0]).alias(f"Is{genre[0]}")
    for genre in ml_genres.rows() if genre[0] is not None
])

# 1) Read lines and split them by "::"
# 2) Select columns and cast them to the correct type
ml_ratings_df = pl.scan_csv(os.path.join(ROOT, ML_RATINGS_PATH), has_header=False).select([
    pl.col("column_1").str.split("::")
]).select([
    pl.col("column_1").list.get(0).alias("UserID").cast(pl.Int32),
    pl.col("column_1").list.get(1).alias("MovieID").cast(pl.Int32),
    pl.col("column_1").list.get(2).alias("Rating").cast(pl.Int32),
    pl.col("column_1").list.get(3).alias("Timestamp").cast(pl.Int32)
])

# Join dataframes
ml_df = ml_ratings_df.join(ml_movies_df, on="MovieID").join(ml_users_df, on="UserID")

#############################################################################BOOK CROSSING

bc_books_df = pl.scan_csv(os.path.join(ROOT, BC_BOOKS_PATH), separator=";")

bc_users_df = pl.scan_csv(os.path.join(ROOT, BC_USERS_PATH), separator=";").filter(
    pl.col("Age").is_between(5, 100)
)

bc_ratings_df = pl.scan_csv(os.path.join(ROOT, BC_RATINGS_PATH), separator=";").filter(
    pl.col("User-ID").is_in(bc_users_df.select("User-ID").collect()),
)

bc_df = bc_ratings_df.join(bc_users_df, on='User-ID').join(bc_books_df, on='ISBN')