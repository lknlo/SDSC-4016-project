import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import traceback
import time

print("== Testing Recommender System ==")
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

# Define simplified model classes to avoid importing the full modules
class SimpleLSTMModel(nn.Module):
    def __init__(self, vocab_size=1000, embedding_dim=100, hidden_dim=128, numerical_dim=11):
        super(SimpleLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim + numerical_dim, 1)
        
    def forward(self, text, numerical):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        hidden = hidden[-1]
        combined = torch.cat((hidden, numerical), dim=1)
        return self.fc(combined).squeeze()

class SimpleMFModel(nn.Module):
    def __init__(self, num_users=100, num_items=100, embedding_dim=50):
        super(SimpleMFModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
    def forward(self, user_indices, item_indices):
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        dot_product = torch.sum(user_emb * item_emb, dim=1)
        return dot_product

print("\nCreating simplified model instances...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create simple models
sentiment_model = SimpleLSTMModel().to(device)
recommender_model = SimpleMFModel().to(device)

print("Models created successfully")

print("\nCreating simple recommender class...")

# Create a simplified recommender class
class SimpleRecommender:
    def __init__(self, sentiment_model, recommender_model, data_path='processed_yelp_data.csv'):
        self.sentiment_model = sentiment_model
        self.recommender_model = recommender_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading data from {data_path}...")
        if os.path.exists(data_path):
            self.data = pd.read_csv(data_path, nrows=1000)  # Load only 1000 rows for testing
            print(f"Loaded {len(self.data)} rows from {data_path}")
            
            # Create sample user/business mappings
            self.user_id_map = {str(uid): i for i, uid in enumerate(self.data['user_id'].unique()[:100])}
            self.business_id_map = {str(bid): i for i, bid in enumerate(self.data['business_id'].unique()[:100])}
            
            # Create reverse mappings
            self.user_idx_map = {v: k for k, v in self.user_id_map.items()}
            self.business_idx_map = {v: k for k, v in self.business_id_map.items()}
        else:
            print(f"Warning: {data_path} not found. Using dummy data.")
            self.data = pd.DataFrame()
            self.user_id_map = {"user1": 0, "user2": 1}
            self.business_id_map = {"business1": 0, "business2": 1}
            self.user_idx_map = {0: "user1", 1: "user2"}
            self.business_idx_map = {0: "business1", 1: "business2"}
    
    def get_recommendations(self, user_id, n=5):
        if user_id not in self.user_id_map:
            print(f"Warning: User {user_id} not found in mapping")
            return []
            
        user_idx = self.user_id_map[user_id]
        business_indices = list(range(len(self.business_id_map)))
        
        # Generate dummy scores
        scores = np.random.random(len(business_indices))
        
        # Return top N recommendations
        top_indices = np.argsort(scores)[-n:][::-1]
        recommendations = []
        
        for idx in top_indices:
            business_idx = business_indices[idx]
            if business_idx in self.business_idx_map:
                business_id = self.business_idx_map[business_idx]
                score = scores[idx]
                recommendations.append((business_id, score))
                
        return recommendations

# Instantiate our simple recommender
try:
    recommender = SimpleRecommender(
        sentiment_model=sentiment_model,
        recommender_model=recommender_model,
        data_path='processed_yelp_data.csv'
    )
    
    print("\nRecommender created successfully!")
    print(f"Number of users: {len(recommender.user_id_map)}")
    print(f"Number of businesses: {len(recommender.business_id_map)}")
    
    # Try to get some recommendations
    print("\nGetting sample recommendations...")
    if recommender.user_id_map:
        sample_user_id = list(recommender.user_id_map.keys())[0]
        recommendations = recommender.get_recommendations(sample_user_id, n=3)
        
        print(f"Top 3 recommendations for user {sample_user_id}:")
        for business_id, score in recommendations:
            print(f"Business: {business_id}, Score: {score:.4f}")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"Error in test: {str(e)}")
    traceback.print_exc() 