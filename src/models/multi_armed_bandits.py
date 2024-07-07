from tqdm import tqdm
import pandas as pd
import numpy as np
from src.models.base import BaseRecommender
import scipy.stats as stats

class MultiArmedBanditsRecommender(BaseRecommender):
    def __init__(self, ml_movies_df, ml_users_df, epsilon=0.1):
        super().__init__(ml_movies_df, ml_users_df)

    def fit(self, ml_ratings_train_df, iterations=1000, strategy="epsilon", epsilon=None, c=None, alpha_prior=None, beta_prior=None, top_k=5):
        assert strategy in ['epsilon', 'ucb', 'thompson'], f"Unknown strategy: {strategy}"
        assert epsilon is not None or strategy != 'epsilon', "Epsilon must be provided for epsilon-greedy strategy"
        assert c is not None or strategy != 'ucb', "C must be provided for UCB strategy"
        assert alpha_prior is not None or strategy != 'thompson', "Alpha prior must be provided for Thompson sampling strategy"
        assert beta_prior is not None or strategy != 'thompson', "Beta prior must be provided for Thompson sampling strategy"
        
        self.ml_ratings_train_df = ml_ratings_train_df
        self.movie_ids = self.ml_movies_df['MovieID'].unique()
        self.num_movies = len(self.movie_ids)
        self.counts = pd.Series(np.zeros(self.num_movies), index=self.movie_ids)
        self.sums = pd.Series(np.zeros(self.num_movies), index=self.movie_ids)
        self.ratings = pd.Series(np.zeros(self.num_movies), index=self.movie_ids)
        self.successes = pd.Series(np.zeros(self.num_movies), index=self.movie_ids)
        self.failures = pd.Series(np.zeros(self.num_movies), index=self.movie_ids)
        
        self.movie_ratings = self.ml_ratings_train_df.groupby('MovieID')['Rating'].apply(list).to_dict()
        
        with tqdm(total=iterations) as pbar:
            for iteration in tqdm(range(1, iterations+1)):
                rating = None
                while rating is None:
                    movie_id = self.recommend_movie(iteration=iteration, strategy=strategy, epsilon=epsilon, c=c, alpha_prior=alpha_prior, beta_prior=beta_prior)
                    if movie_id not in self.movie_ids:
                        continue  # Skip invalid movie IDs
                    
                    ratings = self.movie_ratings.get(movie_id, [])
                    if not ratings:
                        rating = 2.5
                    else:
                        rating = np.random.choice(ratings)
                self.counts[movie_id] += 1
                self.sums[movie_id] += rating
                self.ratings[movie_id] = self.sums[movie_id] / self.counts[movie_id] if self.counts[movie_id] > 0 else 0
                
                if rating >= 3.0:
                    self.successes[movie_id] += 1
                else:
                    self.failures[movie_id] += 1
                    
                pbar.update(1)
                if (iteration + 1) % 1000 == 0 or iteration == iterations - 1:
                    top_k_movies = self.ratings.sort_values(ascending=False).head(top_k)
                    top_k_string = ", ".join([f"{movie_id:>5}: {rating:.2f}" for movie_id, rating in top_k_movies.items()])
                    pbar.set_postfix_str(f"top movies (id: rating): {top_k_string}")

    def recommend_movie(self, iteration, strategy, epsilon, c, alpha_prior, beta_prior):
        if strategy == 'epsilon':
            return self.recommend_movie_epsilon_greedy(iteration=iteration, epsilon=epsilon)
        elif strategy == 'ucb':
            return self.recommend_movie_ucb(iteration=iteration, c=c)
        elif strategy == 'thompson':
            return self.recommend_movie_thompson_sampling(iteration=iteration, alpha_prior=alpha_prior, beta_prior=beta_prior)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
    def recommend_movie_epsilon_greedy(self, iteration, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.movie_ids)
        else:
            return self.movie_ids[np.argmax(self.sums / self.counts)]
        
    def recommend_movie_ucb(self, iteration, c):
        total_counts = np.sum(self.counts)
        if total_counts == 0:
            return np.random.choice(self.movie_ids)
        ucb_values = self.ratings + c * np.sqrt(np.log(total_counts) / self.counts)
        return ucb_values.idxmax()
    
    def recommend_movie_thompson_sampling(self, iteration, alpha_prior, beta_prior):
        alpha_params = alpha_prior + self.successes
        beta_params = beta_prior + self.failures
        beta_samples = np.random.beta(alpha_params, beta_params)
        max_index = np.argmax(beta_samples)
        return self.movie_ids[max_index]
        
    def predict(self, user_id: int, n_recommendations: int):
        user_rated_movies = self.ml_ratings_train_df[self.ml_ratings_train_df['UserID'] == user_id]['MovieID']
        recommendations = self.ratings[~self.ratings.index.isin(user_rated_movies)][:n_recommendations]
        return recommendations