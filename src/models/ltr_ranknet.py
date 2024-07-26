import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import random  
from src.models.base import BaseRecommender

class RankNetDataset(Dataset):
    def __init__(self, ml_ratings_df, user_encoder, movie_encoder):
        self.user_encoder = user_encoder
        self.movie_encoder = movie_encoder
        
        self.user_movie_ratings = {}
        for row in ml_ratings_df.itertuples():
            user = row.UserID
            movie = row.MovieID
            rating = row.Rating
            if user not in self.user_movie_ratings:
                self.user_movie_ratings[user] = {}
            self.user_movie_ratings[user][movie] = rating
        
        self.users = list(self.user_movie_ratings.keys())
    
    def __len__(self):
        return len(self.users) * 10
    
    def __getitem__(self, idx):
        user_id = self.users[idx % len(self.users)]
        user_ratings = self.user_movie_ratings[user_id]
        
        movie_id_1, rating_1 = random.choice(list(user_ratings.items()))
        movie_id_2, rating_2 = random.choice(list(user_ratings.items()))
        
        label = 1 if rating_1 > rating_2 else 0
        
        user_id_encoded = self.user_encoder.transform([user_id])[0]
        movie_id_1_encoded = self.movie_encoder.transform([movie_id_1])[0]
        movie_id_2_encoded = self.movie_encoder.transform([movie_id_2])[0]
        
        return (torch.tensor(user_id_encoded),
                torch.tensor(movie_id_1_encoded),
                torch.tensor(movie_id_2_encoded),
                torch.tensor(label))

class RankNet(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim, hidden_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, user_ids, movie_ids_1, movie_ids_2):
        user_embed = self.user_embedding(user_ids)
        movie_embed_1 = self.movie_embedding(movie_ids_1)
        movie_embed_2 = self.movie_embedding(movie_ids_2)
        
        x1 = torch.cat([user_embed, movie_embed_1], dim=1)
        x2 = torch.cat([user_embed, movie_embed_2], dim=1)
        
        score1 = self.fc2(torch.relu(self.fc1(x1)))
        score2 = self.fc2(torch.relu(self.fc1(x2)))
        
        return score1 - score2
    
    def predict(self, user_ids, movie_ids):
        user_embed = self.user_embedding(user_ids)
        movie_embed = self.movie_embedding(movie_ids)
        
        x = torch.cat([user_embed, movie_embed], dim=1)
        
        return self.fc2(torch.relu(self.fc1(x)))
        
class RankNetRecommender(BaseRecommender):
    def __init__(self, ml_movies_df, ml_users_df, embedding_dim=50, hidden_dim=100, learning_rate=0.001, batch_size=64, epochs=10):
        super().__init__(ml_movies_df, ml_users_df)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        
        self.n_users = len(ml_users_df)
        self.n_movies = len(ml_movies_df)
        
        self.model = None
        
    def fit(self, ml_ratings_train_df):
        # Encode user and movie IDs
        self.user_encoder.fit(ml_ratings_train_df['UserID'])
        self.movie_encoder.fit(ml_ratings_train_df['MovieID'])
        
        # Create the RankNet model
        self.model = RankNet(self.n_users, self.n_movies, self.embedding_dim, self.hidden_dim)
        
        # Prepare the dataset and dataloader
        dataset = RankNetDataset(ml_ratings_train_df, self.user_encoder, self.movie_encoder)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                user_ids, movie_ids_1, movie_ids_2, labels = batch
                
                optimizer.zero_grad()
                outputs = self.model(user_ids, movie_ids_1, movie_ids_2)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    def predict(self, user_id, n_recommendations):
        self.model.eval()
        user_id_encoded = self.user_encoder.transform([user_id])[0]
        user_tensor = torch.tensor([user_id_encoded])
        
        all_movie_ids = self.ml_movies_df['MovieID'].values
        known_movie_ids = [mid for mid in all_movie_ids if mid in self.movie_encoder.classes_]
        all_movie_ids_encoded = self.movie_encoder.transform(known_movie_ids)
        movie_tensors = torch.tensor(all_movie_ids_encoded)
        
        with torch.no_grad():
            scores = self.model.predict(user_tensor.repeat(len(movie_tensors)), movie_tensors)
        
        scores = scores.numpy().flatten()
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        
        recommendations = pd.DataFrame({
            'MovieID': np.array(known_movie_ids)[top_indices],
            'Score': scores[top_indices]
        })
        
        return recommendations.set_index('MovieID')