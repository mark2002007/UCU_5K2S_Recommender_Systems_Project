import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from src.models.base import BaseRecommender

class RatingDataset(Dataset):
    def __init__(self, ratings_df):
        self.user_ids = ratings_df['UserID'].values
        self.movie_ids = ratings_df['MovieID'].values
        self.ratings = ratings_df['Rating'].values

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]

class NeuralCollaborativeFilteringModel(nn.Module):
    def __init__(self, n_users, n_movies, n_factors, hidden_layers):
        super(NeuralCollaborativeFilteringModel, self).__init__()
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.movie_embedding = nn.Embedding(n_movies, n_factors)

        self.hidden_layers = nn.Sequential()
        input_size = n_factors * 2
        for i, units in enumerate(hidden_layers):
            self.hidden_layers.add_module(f"layer_{i}", nn.Linear(input_size, units))
            self.hidden_layers.add_module(f"relu_{i}", nn.ReLU())
            self.hidden_layers.add_module(f"dropout_{i}", nn.Dropout(0.2))
            input_size = units

        self.output_layer = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, movie_ids):
        user_embedding = self.user_embedding(user_ids)
        movie_embedding = self.movie_embedding(movie_ids)
        input_vecs = torch.cat([user_embedding, movie_embedding], dim=-1)
        x = self.hidden_layers(input_vecs)
        output = self.output_layer(x)
        return self.sigmoid(output)

class NeuralCollaborativeFiltering(BaseRecommender):
    def __init__(self, ml_movies_df, ml_users_df, n_factors=50, hidden_layers=[64, 32, 16], device='cpu'):
        super().__init__(ml_movies_df, ml_users_df)
        self.n_factors = n_factors
        self.hidden_layers = hidden_layers
        self.device = torch.device(device)
        self.model = None

        # Create mappings for user and movie IDs
        self.user2idx = {user_id: idx for idx, user_id in enumerate(ml_users_df['UserID'].unique())}
        self.movie2idx = {movie_id: idx for idx, movie_id in enumerate(ml_movies_df['MovieID'].unique())}

    def fit(self, ml_ratings_train_df, epochs=10, batch_size=64, lr=0.001):
        self.ml_ratings_train_df = ml_ratings_train_df.copy()
        n_users = self.ml_users_df['UserID'].nunique()
        n_movies = self.ml_movies_df['MovieID'].nunique()

        # Normalize ratings
        self.ml_ratings_train_df['Rating'] = self.ml_ratings_train_df['Rating'] / self.ml_ratings_train_df['Rating'].max()

        # Map user and movie IDs to indices
        self.ml_ratings_train_df['UserID'] = self.ml_ratings_train_df['UserID'].map(self.user2idx)
        self.ml_ratings_train_df['MovieID'] = self.ml_ratings_train_df['MovieID'].map(self.movie2idx)

        dataset = RatingDataset(self.ml_ratings_train_df)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = NeuralCollaborativeFilteringModel(n_users, n_movies, self.n_factors, self.hidden_layers).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
                for user_ids, movie_ids, ratings in dataloader:
                    user_ids = user_ids.to(self.device).long()
                    movie_ids = movie_ids.to(self.device).long()
                    ratings = ratings.to(self.device).float()

                    optimizer.zero_grad()
                    outputs = self.model(user_ids, movie_ids).squeeze()
                    loss = criterion(outputs, ratings)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    pbar.set_postfix(loss=epoch_loss / (pbar.n + 1))
                    pbar.update(1)

    def predict(self, user_id, n_recommendations):
        self.model.eval()
        n_movies = self.ml_movies_df['MovieID'].nunique()

        # Map user_id to index
        user_idx = self.user2idx[user_id]

        movie_ids = torch.arange(n_movies).to(self.device).long()
        user_ids = torch.tensor([user_idx] * n_movies).to(self.device).long()

        with torch.no_grad():
            predictions = self.model(user_ids, movie_ids).squeeze().cpu().numpy()
        recommendations = pd.DataFrame({'MovieID': self.ml_movies_df['MovieID'].values, 'Score': predictions})
        recommendations = recommendations[['MovieID', 'Score']]  # Ensure only the necessary columns
        recommendations = recommendations.sort_values('Score', ascending=False).head(n_recommendations)

        return recommendations.set_index('MovieID')