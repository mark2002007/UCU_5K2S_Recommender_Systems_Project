
dataset ==> https://grouplens.org/datasets/movielens/1m/

### Instructions

1. **Use python 3.9**
   ```
   python3.9 -m venv venv
   ```

2. **Activate Environment**
   ```
   source venv/bin/activate
   ```

3. **Install Requirements**
   ```
   pip install -r requirements.txt
   ```

4. **Install FunkSVD**
   ```
   pip install git+https://github.com/gbolmier/funk-svd
   ```

5. **Run the Data Preprocessing Script (Optional)**

   Execute the following command to start the data preprocessing:
   ```
   cd scripts
   python data_preprocessing.py
   ```

---

### Project Structure

#### `/artifacts`
Contains useful project artifacts, such as a PDF file with the task description.

#### `/data`
Stores data from the BookCrossing and MovieLens datasets across two subfolders:
- `/data/bronze` - Contains the raw, downloaded data.
- `/data/silver` - Contains refined and parsed data in Parquet file format.

#### `/scripts`
Includes scripts that are not used regularly. For example, the `data_preprocessing.py` script is used once to preprocess data and populate the `/data/silver` folder.

#### `/src`
A package containing all the recommendation models and utilities for common data operations, such as splitting or reading data.

#### `/experiments`
Contains Jupyter notebooks used for exploring and evaluating different approaches, as well as for conducting Exploratory Data Analysis (EDA) of the datasets.

---

### Next Steps
After preprocessing, you can proceed with your experiments, utilizing the data from the `silver` bucket.

### Overview

#### Data

We performed a train-test split by time at point t_split_point and then adjusted the train and test sets to have at least K observations in the test set and at least one observation in the train set for each user from the test set. This was done as @K metrics require having at least K relevant items for the evaluated user.

#### EDA

We decided to perform EDA on two datasets: MovieLens and BookCrossing and then proceeded with MovieLens as it has fewer outliers in users' ages and allows for larger sets to be introduced in the future.

#### Baseline Recommender

As the baseline recommender, we chose a popularity-based recommender.

#### Content-Based Filtering

In content-based filtering, we compute similarity between items using genres ('Is*' columns) and recommend similar movies to the ones the user watched.

#### Item-Item Collaborative Filtering

Works similar to content based, but similarity is computed using utility matrix

#### User-User Collaborative Filtering

Here we compute similarities between users based on their ratings, and then for unwatched movies unwatched by User we find their probable ratings as the sum of ratings of similar users weighted by their similarities to User.

#### PageRank 

Similar to a popularity-based recommender, but scores are computed by the PageRank algorithm, where the adjacency matrix is constructed by the following rule:
If user rated movie i (r_i) lower than movie j (r_j) then A[i, j] += r_j - r_i

#### SVD

Decomposes ratings matrix onto user-feature and feature-item matrix, treating missing values as mean ratings.

#### SVD (ALS)

Similar to SVD, but uses alternating least squares to approximate U and V.

#### SVD (Gradient Descent)

Similar to SVD, but uses gradient descent to simultaneously approximate U and V.

#### SVD (Funk)

Similar to SVD (Gradient Descent) but with additional features such as biases and regularization.

#### Neural Collaborative Filtering

Uses embeddings and neural network layers to predict relevant movies.

#### Multi-Armed Bandits

Determines which movie would be the best rated on test set by recommending different movies and observing users ratings on train set.


### Results

| Algorithm | AVG Precision@K | AVG Recall@K | AVG F1@K | AVG Average Precision | Best Precision@K | Best Recall@K | Best F1@K | Best Average Precision | MRR | AVG NDCG |
| ---------------                                                             | ------ | ------ |  ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Popularity-Based                                                            | 0.3381 | 0.0655 | 0.0983 | 0.3772 | 1.0000 | 0.3043 | 0.3273 | 1.0000 | 0.6467 | 0.6505 |
| Content-Based Filtering                                                     | 0.0573 | 0.0095 | 0.0145 | 0.0521 | 0.5500 | 0.2069 | 0.2449 | 0.6248 | 0.5896 | 0.2184 |
| Item-Item Colaborative Filtering                                            | 0.0108 | 0.0013 | 0.0020 | 0.0102 | 0.4500 | 0.0476 | 0.0526 | 0.5330 | 0.8830 | 0.0545 |
| User-User Colaborative Filtering                                            | 0.1473 | 0.0266 | 0.0401 | 0.1719 | 0.8000 | 0.1724 | 0.2041 | 0.8991 | 0.5861 | 0.4455 |
| PageRank                                                                    | 0.3172 | 0.0629 | 0.0937 | 0.3573 | 1.0000 | 0.3478 | 0.3721 | 1.0000 | 0.6494 | 0.6355 |
| SVD                                                                         | 0.3460 | 0.0706 | 0.1046 | 0.3907 | 1.0000 | 0.3600 | 0.4000 | 1.0000 | 0.6665 | 0.6692 |
| SVD (ALS)                                                                   | 0.3648 | 0.0762 | 0.1126 | 0.4101 | 1.0000 | 0.3750 | 0.4091 | 1.0000 | 0.6769 | 0.6813 |
| SVD (Gradient Descent)                                                      | 0.1510 | 0.0298 | 0.0442 | 0.1796 | 0.9500 | 0.2609 | 0.3390 | 0.9975 | 0.5649 | 0.4657 |
| SVD (Funk)                                                                  | 0.1054 | 0.0193 | 0.0289 | 0.1071 | 0.7000 | 0.3913 | 0.4186 | 0.7465 | 0.5147 | 0.3434 |
| Neural Colaborative Filtering                                               | 0.1043 | 0.0214 | 0.0313 | 0.0999 | 0.7500 | 0.2609 | 0.2791 | 0.6467 | 0.4972 | 0.3316 |
| Multi-Armed Bandits (Epsilon-Greedy)                                        | 0.0714 | 0.0088 | 0.0142 | 0.0923 | 0.6500 | 0.0909 | 0.0984 | 0.7669 | 0.7410 | 0.3052 |
| Multi-Armed Bandits (UCB)                                                   | 0.0714 | 0.0088 | 0.0142 | 0.0923 | 0.6500 | 0.0909 | 0.0984 | 0.7669 | 0.7410 | 0.3052 |
| Multi-Armed Bandits (Thompson)                                              | 0.0714 | 0.0088 | 0.0142 | 0.0923 | 0.6500 | 0.0909 | 0.0984 | 0.7669 | 0.7410 | 0.3052 |
| LTR Linear Regression                                                       | 0.0027 | 0.0003 | 0.0005 | 0.0023 | 0.2000 | 0.0370 | 0.0426 | 0.2608 | 0.9608 | 0.0145 |
| RankNet                                                                     | 0.1233 | 0.0241 | 0.0358 | 0.1388 | 0.7000 | 0.2174 | 0.2326 | 0.7877 | 0.5238 | 0.3914 |
| TwoStage (PopularityBasedRecommender -> RankNetRecommender)                 | 0.3067 | 0.0595 | 0.0891 | 0.2880 | 1.0000 | 0.3478 | 0.3721 | 1.0000 | 0.5263 | 0.5683 |
| TwoStage (PopularityBasedRecommender -> LTRLinearRegressionRecommender)     | 0.3034 | 0.0603 | 0.0898 | 0.3166 | 1.0000 | 0.3913 | 0.4186 | 1.0000 | 0.5600 | 0.6019 |
| TwoStage (ItemItemCollaborativeFiltering -> RankNetRecommender)             | 0.0124 | 0.0018 | 0.0026 | 0.0131 | 0.4500 | 0.1000 | 0.1000 | 0.4358 | 0.8762 | 0.0640 |
| TwoStage (ItemItemCollaborativeFiltering -> LTRLinearRegressionRecommender) | 0.0115 | 0.0017 | 0.0024 | 0.0124 | 0.4500 | 0.1000 | 0.1143 | 0.5895 | 0.8774 | 0.0659 |
| TwoStage (UserUserColaborativeFiltering -> RankNetRecommender)              | 0.1782 | 0.0347 | 0.0515 | 0.1449 | 0.9000 | 0.2143 | 0.2500 | 0.8620 | 0.4228 | 0.4106 |
| TwoStage (UserUserColaborativeFiltering -> LTRLinearRegressionRecommender)  | 0.1592 | 0.0304 | 0.0454 | 0.0454 | 0.8500 | 0.2069 | 0.2500 | 0.8613 | 0.4626 | 0.4137 |
| TwoStage (NeuralCollaborativeFiltering -> RankNetRecommender)               | 0.1404 | 0.0280 | 0.0414 | 0.1234 | 0.7500 | 0.2609 | 0.2791 | 0.7431 | 0.4618 | 0.3618 |
| TwoStage (NeuralCollaborativeFiltering -> LTRLinearRegressionRecommender)   | 0.0676 | 0.0138 | 0.0202 | 0.0989 | 0.4500 | 0.1739 | 0.1860 | 0.6971 | 0.6261 | 0.3288 |
| TwoStage (FunkSVDCollaborativeFiltering -> RankNetRecommender)              | 0.1263 | 0.0252 | 0.0374 | 0.1143 | 0.7500 | 0.2609 | 0.2791 | 0.8203 | 0.4646 | 0.3550 |
| TwoStage (FunkSVDCollaborativeFiltering -> LTRLinearRegressionRecommender)  | 0.1127 | 0.0216 | 0.0320 | 0.1098 | 0.8500 | 0.2609 | 0.2791 | 0.8698 | 0.5032 | 0.3418 |
| TwoStage (SVDALSCollaborativeFiltering -> RankNetRecommender)               | 0.3386 | 0.0698 | 0.1033 | 0.3341 | 1.0000 | 0.3478 | 0.3721 | 1.0000 | 0.5726 | 0.6138 |
| TwoStage (SVDALSCollaborativeFiltering -> LTRLinearRegressionRecommender)   | 0.3343 | 0.0707 | 0.1041 | 0.3357 | 1.0000 | 0.3333 | 0.3636 | 1.0000 | 0.5556 | 0.6189 |
| TwoStage (SVDCollaborativeFiltering -> RankNetRecommender)                  | 0.3062 | 0.0613 | 0.0911 | 0.3083 | 1.0000 | 0.3226 | 0.3922 | 1.0000 | 0.5516 | 0.5887 |
| TwoStage (SVDCollaborativeFiltering -> LTRLinearRegressionRecommender)      | 0.2824 | 0.0588 | 0.0866 | 0.2902 | 0.9500 | 0.3143 | 0.4000 | 0.9949 | 0.5483 | 0.5754 |
| TwoStage (SVDGradientDescentRecommender -> RankNetRecommender)              | 0.1967 | 0.0396 | 0.0586 | 0.1595 | 0.9500 | 0.3913 | 0.4186 | 0.9921 | 0.4018 | 0.4295 |
| TwoStage (SVDGradientDescentRecommender -> LTRLinearRegressionRecommender)  | 0.1200 | 0.0236 | 0.0349 | 0.1227 | 0.8000 | 0.2609 | 0.2791 | 0.9058 | 0.5459 | 0.3737 |
| Markov Chain                                                                | 0.1235 | 0.0240 | 0.0356 | 0.1368 | 0.7000 | 0.2083 | 0.2667 | 0.7863 | 0.5304 | 0.3843 |

AVG Precision@K, AVG Recall@k, and AVG F1@K show moderate results for all algorithms so far. The reason of low performance on this metrics is due to large amount of relevant items in test data (more than 100 movies on average are rated by user in the test data). It leads to very low recall (less than 0.1 in almost all cases), which downgrades f1. Also, AVG Recall@K grows as K grows. We decided to enhance evaluation approach by adding 2 Ranking Metrics: MRR and NDCG. Both of them show quite good results for most of the algorithms. MRR > 0.5 means that for large amount of user the first relevant movie in recommendations is either first or second one. NDCG @ 20 > 0.5 is also a excellent result, as it means that more than half of the list is sorted well.