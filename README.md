
dataset ==> https://grouplens.org/datasets/movielens/1m/

## Setup Instructions

1. **Navigate to the Scripts Folder**

   Open your terminal and switch to the `scripts` directory:
   ```
   cd scripts
   ```

2. **Run the Data Preprocessing Script**

   Execute the following command to start the data preprocessing:
   ```
   python data_preprocessing.py
   ```

   This script processes the data and stores it in the `silver` bucket, ready for further experimentation.

### Next Steps
After preprocessing, you can proceed with your experiments, utilizing the data from the `silver` bucket.

## Overview

### Data

We performed a train-test split by time at point t_split_point and then adjusted the train and test sets to have at least K observations in the test set and at least one observation in the train set for each user from the test set. This was done as @K metrics require having at least K relevant items for the evaluated user.

### EDA

We decided to perform EDA on two datasets: MovieLens and BookCrossing and then proceeded with MovieLens as it has fewer outliers in users' ages and allows for larger sets to be introduced in the future.

### Baseline Recommender

As the baseline recommender, we chose a popularity-based recommender.

### Content-Based Filtering

In content-based filtering, we compute similarity between items using genres ('Is*' columns) and recommend similar movies to the ones the user watched.

### Item-Item Collaborative Filtering

Works similar to content based, but similarity is computed using utility matrix

### User-User Collaborative Filtering

Here we compute similarities between users based on their ratings, and then for unwatched movies unwatched by User we find their probable ratings as the sum of ratings of similar users weighted by their similarities to User.

### PageRank 

Similar to a popularity-based recommender, but scores are computed by the PageRank algorithm, where the adjacency matrix is constructed by the following rule:
If user rated movie i (r_i) lower than movie j (r_j) ==> A[i, j] += r_j - r_i