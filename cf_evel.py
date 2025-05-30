import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime

# set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load the processed data
print("Loading data...")
data = pd.read_csv('processed_yelp_data.csv')
print(f"Data loaded with shape: {data.shape}")

# Define default column names
user_idx_col = 'user_idx' if 'user_idx' in data.columns else None
business_idx_col = 'business_idx' if 'business_idx' in data.columns else None

# Reduce dataset size to speed up training
sample_size = 10000  # Use only 10K samples
if len(data) > sample_size:
    print(f"Sampling {sample_size} records to reduce runtime...")
    # First get unique users and businesses to sample from
    unique_users = data[user_idx_col].unique() if user_idx_col else data['user_id'].unique()
    unique_businesses = data[business_idx_col].unique() if business_idx_col else data['business_id'].unique()
    
    # Randomly sample a subset of users and businesses
    np.random.seed(RANDOM_SEED)
    sample_users = np.random.choice(unique_users, size=min(1000, len(unique_users)), replace=False)
    sample_businesses = np.random.choice(unique_businesses, size=min(1000, len(unique_businesses)), replace=False)
    
    # Filter data to only include these users and businesses
    if user_idx_col and business_idx_col:
        data = data[data[user_idx_col].isin(sample_users) & data[business_idx_col].isin(sample_businesses)]
    else:
        data = data[data['user_id'].isin(sample_users) & data['business_id'].isin(sample_businesses)]
        
    # If still too large, take a random sample
    if len(data) > sample_size:
        data = data.sample(sample_size, random_state=RANDOM_SEED)
        
    print(f"Reduced data shape: {data.shape}")
    print(f"Number of unique users: {data[user_idx_col].nunique() if user_idx_col else data['user_id'].nunique()}")
    print(f"Number of unique businesses: {data[business_idx_col].nunique() if business_idx_col else data['business_id'].nunique()}")

# define features for collaborative filtering
cf_features = ['user_id', 'business_id', 'stars']
context_features = ['year', 'month', 'day_of_week', 'latitude', 'longitude']

# ensure all necessary columns exist
required_columns = cf_features + context_features
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    print(f"Warning: The following columns are missing: {missing_columns}")
    for col in missing_columns:
        if col in ['year', 'month', 'day_of_week']:
            # create dummy temporal features
            if col == 'year':
                data[col] = 2020
            elif col == 'month':
                data[col] = 6
            elif col == 'day_of_week':
                data[col] = 3
        elif col in ['latitude', 'longitude']:
            # create dummy location features
            data[col] = 0.0
        else:
            print(f"Error: Critical column {col} is missing. Cannot proceed.")
            exit(1)

# ensure user_id and business_id are available as integer indices
if 'user_idx' in data.columns and 'business_idx' in data.columns:
    print("Using existing user_idx and business_idx columns...")
    user_idx_col = 'user_idx'
    business_idx_col = 'business_idx'
    
    # Remap indices to be contiguous after sampling
    print("Remapping indices to be contiguous...")
    unique_users = data[user_idx_col].unique()
    unique_businesses = data[business_idx_col].unique()
    
    user_to_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_users)}
    business_to_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_businesses)}
    
    # Create remapped index columns
    data['user_idx_new'] = data[user_idx_col].map(user_to_idx)
    data['business_idx_new'] = data[business_idx_col].map(business_to_idx)
    
    # Replace old columns with remapped ones
    data[user_idx_col] = data['user_idx_new']
    data[business_idx_col] = data['business_idx_new']
    data.drop(['user_idx_new', 'business_idx_new'], axis=1, inplace=True)
else:
    print("Creating user and business indices...")
    # create mappings from IDs to indices
    unique_users = data['user_id'].unique()
    unique_businesses = data['business_id'].unique()

    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    business_to_idx = {business: idx for idx, business in enumerate(unique_businesses)}

    # create new columns with indices
    data['user_idx'] = data['user_id'].map(user_to_idx)
    data['business_idx'] = data['business_id'].map(business_to_idx)

    user_idx_col = 'user_idx'
    business_idx_col = 'business_idx'

    # store the mapping sizes for embedding dimensions
    num_users = len(unique_users)
    num_businesses = len(unique_businesses)
    print(f"Number of unique users: {num_users}")
    print(f"Number of unique businesses: {num_businesses}")

# get the mapping sizes for embedding dimensions
num_users = data[user_idx_col].nunique()
num_businesses = data[business_idx_col].nunique()

# time-based split
print("Splitting data chronologically...")
if 'date' in data.columns:
    # convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'], errors='coerce')

    # sort by date
    data = data.sort_values('date')
else:
    print("No date column found. Creating dummy dates for chronological split.")
    data['date'] = pd.date_range(start='2020-01-01', periods=len(data))

# split data: 70% train, 15% validation, 15% test
train_size = 0.7
val_size = 0.15

# Sort data if date is available
if 'date' in data.columns:
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.sort_values('date')

# Split into train, validation and test sets
train_end_idx = int(len(data) * train_size)
val_end_idx = int(len(data) * (train_size + val_size))

train_data = data.iloc[:train_end_idx].copy()
val_data = data.iloc[train_end_idx:val_end_idx].copy()
test_data = data.iloc[val_end_idx:].copy()

# Ensure all indices in validation and test sets are also in training set
train_users = set(train_data[user_idx_col].unique())
train_businesses = set(train_data[business_idx_col].unique())

# Filter validation data
val_data_filtered = val_data[
    val_data[user_idx_col].isin(train_users) & 
    val_data[business_idx_col].isin(train_businesses)
]

# Use filtered validation data only if it's non-empty
if len(val_data_filtered) > 0:
    val_data = val_data_filtered
    print(f"Filtered validation data size: {len(val_data)}")
else:
    print("Warning: Filtered validation data is empty. Using original validation split.")

# Filter test data
test_data_filtered = test_data[
    test_data[user_idx_col].isin(train_users) & 
    test_data[business_idx_col].isin(train_businesses)
]

# Use filtered test data only if it's non-empty
if len(test_data_filtered) > 0:
    test_data = test_data_filtered
    print(f"Filtered test data size: {len(test_data)}")
else:
    print("Warning: Filtered test data is empty. Using original test split.")

# If validation or test set is still empty, take a portion from the training set
if len(val_data) == 0:
    print("Creating validation set from training data")
    train_val_split = int(len(train_data) * 0.85)
    val_data = train_data[train_val_split:].copy()
    train_data = train_data[:train_val_split].copy()

if len(test_data) == 0:
    print("Creating test set from training data")
    train_test_split = int(len(train_data) * 0.85)
    test_data = train_data[train_test_split:].copy()
    train_data = train_data[:train_test_split].copy()

print(f"Train data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")
print(f"Test data size: {len(test_data)}")
print(f"Unique users in training: {len(train_data[user_idx_col].unique())}")
print(f"Unique businesses in training: {len(train_data[business_idx_col].unique())}")

# scale context features
print("Scaling context features...")
context_scaler = MinMaxScaler()
train_data[context_features] = context_scaler.fit_transform(train_data[context_features])
val_data[context_features] = context_scaler.transform(val_data[context_features])
test_data[context_features] = context_scaler.transform(test_data[context_features])


# define PyTorch Dataset
class RecommendationDataset(Dataset):
    def __init__(self, dataframe, user_col, item_col, rating_col, context_features=None):
        self.dataframe = dataframe
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.context_features = context_features

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        user_idx = self.dataframe.iloc[idx][self.user_col]
        item_idx = self.dataframe.iloc[idx][self.item_col]
        rating = self.dataframe.iloc[idx][self.rating_col]

        # convert to tensors
        user_idx_tensor = torch.tensor(user_idx, dtype=torch.long)
        item_idx_tensor = torch.tensor(item_idx, dtype=torch.long)
        rating_tensor = torch.tensor(rating, dtype=torch.float)

        # include context features if specified
        if self.context_features is not None and len(self.context_features) > 0:
            context_data = self.dataframe.iloc[idx][self.context_features].values.astype(np.float32)
            context_tensor = torch.tensor(context_data, dtype=torch.float)
            return {
                'user_idx': user_idx_tensor,
                'item_idx': item_idx_tensor,
                'context': context_tensor,
                'rating': rating_tensor
            }
        else:
            return {
                'user_idx': user_idx_tensor,
                'item_idx': item_idx_tensor,
                'rating': rating_tensor
            }


# define Model Architectures

# 1. Basic Matrix Factorization Model (Baseline)
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50):
        super(MatrixFactorization, self).__init__()

        # user and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        # global bias, user bias, and item bias
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # initialize biases
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_indices, item_indices):
        # get embeddings and biases
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)

        user_b = self.user_bias(user_indices).squeeze()
        item_b = self.item_bias(item_indices).squeeze()

        # compute dot product
        dot_product = torch.sum(user_emb * item_emb, dim=1)

        # prediction = global bias + user bias + item bias + user-item interaction
        prediction = self.global_bias + user_b + item_b + dot_product

        return prediction


# 2. Neural Collaborative Filtering (NCF) Model
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50, hidden_layers=[100, 50]):
        super(NeuralCollaborativeFiltering, self).__init__()

        # user and item embeddings for Generalized Matrix Factorization (GMF)
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)

        # user and item embeddings for Multi-Layer Perceptron (MLP)
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

        # initialize embeddings
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

        # mLP layers
        self.mlp_layers = nn.ModuleList()
        input_size = 2 * embedding_dim

        for i, hidden_size in enumerate(hidden_layers):
            self.mlp_layers.append(nn.Linear(input_size, hidden_size))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.BatchNorm1d(hidden_size))
            self.mlp_layers.append(nn.Dropout(0.2))
            input_size = hidden_size

        # output layer
        self.output_layer = nn.Linear(embedding_dim + hidden_layers[-1], 1)

    def forward(self, user_indices, item_indices):
        # gMF part
        user_emb_gmf = self.user_embedding_gmf(user_indices)
        item_emb_gmf = self.item_embedding_gmf(item_indices)
        gmf_output = user_emb_gmf * item_emb_gmf

        # mLP part
        user_emb_mlp = self.user_embedding_mlp(user_indices)
        item_emb_mlp = self.item_embedding_mlp(item_indices)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=1)

        # apply MLP layers
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)

        # concatenate GMF and MLP outputs
        ncf_output = torch.cat([gmf_output, mlp_input], dim=1)

        # final prediction
        prediction = self.output_layer(ncf_output).squeeze()

        return prediction


# 3. Context-Aware Recommendation Model
class ContextAwareRecommender(nn.Module):
    def __init__(self, num_users, num_items, context_dim,
                 embedding_dim=50, context_hidden_dims=[64, 32],
                 final_hidden_dims=[128, 64]):
        super(ContextAwareRecommender, self).__init__()

        # user and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        # context feature encoder
        context_layers = []
        input_dim = context_dim

        for hidden_dim in context_hidden_dims:
            context_layers.append(nn.Linear(input_dim, hidden_dim))
            context_layers.append(nn.ReLU())
            context_layers.append(nn.BatchNorm1d(hidden_dim))
            context_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        self.context_encoder = nn.Sequential(*context_layers)
        context_output_dim = context_hidden_dims[-1] if context_hidden_dims else context_dim

        # final MLP
        mlp_layers = []
        combined_dim = 2 * embedding_dim + context_output_dim  # user_emb + item_emb + context_features

        for hidden_dim in final_hidden_dims:
            mlp_layers.append(nn.Linear(combined_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.Dropout(0.2))
            combined_dim = hidden_dim

        mlp_layers.append(nn.Linear(combined_dim, 1))

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, user_indices, item_indices, context_features):
        # get embeddings
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)

        # process context features
        context_encoded = self.context_encoder(context_features)

        # concatenate all features
        combined = torch.cat([user_emb, item_emb, context_encoded], dim=1)

        # final prediction
        prediction = self.mlp(combined).squeeze()

        return prediction


# create datasets
train_dataset_basic = RecommendationDataset(
    train_data, user_idx_col, business_idx_col, 'stars'
)
val_dataset_basic = RecommendationDataset(
    val_data, user_idx_col, business_idx_col, 'stars'
)
test_dataset_basic = RecommendationDataset(
    test_data, user_idx_col, business_idx_col, 'stars'
)

train_dataset_context = RecommendationDataset(
    train_data, user_idx_col, business_idx_col, 'stars', context_features
)
val_dataset_context = RecommendationDataset(
    val_data, user_idx_col, business_idx_col, 'stars', context_features
)
test_dataset_context = RecommendationDataset(
    test_data, user_idx_col, business_idx_col, 'stars', context_features
)

# create dataloaders
batch_size = 1024

train_loader_basic = DataLoader(train_dataset_basic, batch_size=batch_size, shuffle=True)
val_loader_basic = DataLoader(val_dataset_basic, batch_size=batch_size)
test_loader_basic = DataLoader(test_dataset_basic, batch_size=batch_size)

train_loader_context = DataLoader(train_dataset_context, batch_size=batch_size, shuffle=True)
val_loader_context = DataLoader(val_dataset_context, batch_size=batch_size)
test_loader_context = DataLoader(test_dataset_context, batch_size=batch_size)


# define training function for basic models
def train_basic_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # training
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training'):
            # zero gradients
            optimizer.zero_grad()

            # get data
            user_indices = batch['user_idx'].to(device)
            item_indices = batch['item_idx'].to(device)
            ratings = batch['rating'].to(device)

            # forward pass
            outputs = model(user_indices, item_indices)

            # calculate loss
            loss = criterion(outputs, ratings)

            # backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(ratings)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # validation
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation'):
                # get data
                user_indices = batch['user_idx'].to(device)
                item_indices = batch['item_idx'].to(device)
                ratings = batch['rating'].to(device)

                # forward pass
                outputs = model(user_indices, item_indices)

                # calculate loss
                loss = criterion(outputs, ratings)
                running_val_loss += loss.item() * len(ratings)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {epoch_train_loss:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f}')

        # save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()

    # load best model
    model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


# define training function for context-aware model
def train_context_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # training
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training'):
            # zero gradients
            optimizer.zero_grad()

            # get data
            user_indices = batch['user_idx'].to(device)
            item_indices = batch['item_idx'].to(device)
            context = batch['context'].to(device)
            ratings = batch['rating'].to(device)

            # forward pass
            outputs = model(user_indices, item_indices, context)

            # calculate loss
            loss = criterion(outputs, ratings)

            # backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(ratings)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # validation
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation'):
                # get data
                user_indices = batch['user_idx'].to(device)
                item_indices = batch['item_idx'].to(device)
                context = batch['context'].to(device)
                ratings = batch['rating'].to(device)

                # forward pass
                outputs = model(user_indices, item_indices, context)

                # calculate loss
                loss = criterion(outputs, ratings)
                running_val_loss += loss.item() * len(ratings)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {epoch_train_loss:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f}')

        # save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()

    # load best model
    model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


# define evaluation function for basic models
def evaluate_basic_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            # get data
            user_indices = batch['user_idx'].to(device)
            item_indices = batch['item_idx'].to(device)
            ratings = batch['rating'].to(device)

            # forward pass
            outputs = model(user_indices, item_indices)

            # calculate loss
            loss = criterion(outputs, ratings)
            running_loss += loss.item() * len(ratings)

            # save predictions and targets for metrics
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(ratings.cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)

    # convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # clip predictions to rating range [1, 5]
    all_preds = np.clip(all_preds, 1.0, 5.0)

    # calculate metrics
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test RMSE: {rmse:.4f}')
    print(f'Test MAE: {mae:.4f}')

    # plot prediction distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(all_preds, bins=20, alpha=0.5, label='Predicted', color='blue')
    sns.histplot(all_targets, bins=20, alpha=0.5, label='Actual', color='red')
    plt.title('Distribution of Actual vs Predicted Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'rating_distribution_{model.__class__.__name__}.png')
    plt.close()

    # plot scatter plot of actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets, all_preds, alpha=0.1)
    plt.plot([1, 5], [1, 5], 'r--')
    plt.title('Actual vs Predicted Ratings')
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.xlim(1, 5)
    plt.ylim(1, 5)
    plt.tight_layout()
    plt.savefig(f'actual_vs_predicted_{model.__class__.__name__}.png')
    plt.close()

    return test_loss, rmse, mae, all_preds, all_targets


# define evaluation function for context model
def evaluate_context_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            # get data
            user_indices = batch['user_idx'].to(device)
            item_indices = batch['item_idx'].to(device)
            context = batch['context'].to(device)
            ratings = batch['rating'].to(device)

            # forward pass
            outputs = model(user_indices, item_indices, context)

            # calculate loss
            loss = criterion(outputs, ratings)
            running_loss += loss.item() * len(ratings)

            # save predictions and targets for metrics
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(ratings.cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)

    # convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # clip predictions to rating range [1, 5]
    all_preds = np.clip(all_preds, 1.0, 5.0)

    # calculate metrics
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test RMSE: {rmse:.4f}')
    print(f'Test MAE: {mae:.4f}')

    # plot prediction distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(all_preds, bins=20, alpha=0.5, label='Predicted', color='blue')
    sns.histplot(all_targets, bins=20, alpha=0.5, label='Actual', color='red')
    plt.title('Distribution of Actual vs Predicted Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'rating_distribution_{model.__class__.__name__}.png')
    plt.close()

    # plot scatter plot of actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets, all_preds, alpha=0.1)
    plt.plot([1, 5], [1, 5], 'r--')
    plt.title('Actual vs Predicted Ratings')
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.xlim(1, 5)
    plt.ylim(1, 5)
    plt.tight_layout()
    plt.savefig(f'actual_vs_predicted_{model.__class__.__name__}.png')
    plt.close()

    return test_loss, rmse, mae, all_preds, all_targets


# define function to generate top-n recommendations
def generate_recommendations(model, user_idx, business_indices, top_n=10, use_context=False, context_data=None):
    model.eval()

    # convert user_idx to tensor and expand to match the size of business_indices
    user_indices = torch.tensor([user_idx] * len(business_indices), dtype=torch.long).to(device)
    business_indices_tensor = torch.tensor(business_indices, dtype=torch.long).to(device)

    with torch.no_grad():
        if use_context:
            # repeat context data for each business
            context_tensor = torch.tensor(np.repeat(context_data.reshape(1, -1), len(business_indices), axis=0),
                                          dtype=torch.float).to(device)

            # get predictions
            predictions = model(user_indices, business_indices_tensor, context_tensor)
        else:
            # get predictions
            predictions = model(user_indices, business_indices_tensor)

    # get top-n businesses
    _, indices = torch.topk(predictions, top_n)
    top_business_indices = business_indices_tensor[indices].cpu().numpy()

    return top_business_indices, predictions[indices].cpu().numpy()


# create results directory
results_dir = 'cf_results'
os.makedirs(results_dir, exist_ok=True)

# initialize models
# 1. Matrix Factorization Model (Baseline)
print("\n--- Training Matrix Factorization Model ---")
mf_model = MatrixFactorization(
    num_users=num_users,
    num_items=num_businesses,
    embedding_dim=50
).to(device)

# 2. Neural Collaborative Filtering Model
print("\n--- Training Neural Collaborative Filtering Model ---")
ncf_model = NeuralCollaborativeFiltering(
    num_users=num_users,
    num_items=num_businesses,
    embedding_dim=50,
    hidden_layers=[100, 50]
).to(device)

# 3. Context-Aware Recommendation Model
print("\n--- Training Context-Aware Recommendation Model ---")
context_model = ContextAwareRecommender(
    num_users=num_users,
    num_items=num_businesses,
    context_dim=len(context_features),
    embedding_dim=50,
    context_hidden_dims=[32, 16],
    final_hidden_dims=[128, 64]
).to(device)

# define loss function and optimizers
criterion = nn.MSELoss()

optimizer_mf = optim.Adam(mf_model.parameters(), lr=0.001, weight_decay=1e-6)
optimizer_ncf = optim.Adam(ncf_model.parameters(), lr=0.001, weight_decay=1e-6)
optimizer_context = optim.Adam(context_model.parameters(), lr=0.001, weight_decay=1e-6)

# apply L2 regularization through weight decay

# train models
print("\nTraining Matrix Factorization Model...")
mf_model, mf_train_losses, mf_val_losses = train_basic_model(
    mf_model, train_loader_basic, val_loader_basic,
    criterion, optimizer_mf, num_epochs=3
)

print("\nTraining Neural Collaborative Filtering Model...")
ncf_model, ncf_train_losses, ncf_val_losses = train_basic_model(
    ncf_model, train_loader_basic, val_loader_basic,
    criterion, optimizer_ncf, num_epochs=3
)

print("\nTraining Context-Aware Recommendation Model...")
context_model, context_train_losses, context_val_losses = train_context_model(
    context_model, train_loader_context, val_loader_context,
    criterion, optimizer_context, num_epochs=3
)

# evaluate models
print("\nEvaluating Matrix Factorization Model...")
mf_results = evaluate_basic_model(mf_model, test_loader_basic, criterion)

print("\nEvaluating Neural Collaborative Filtering Model...")
ncf_results = evaluate_basic_model(ncf_model, test_loader_basic, criterion)

print("\nEvaluating Context-Aware Recommendation Model...")
context_results = evaluate_context_model(context_model, test_loader_context, criterion)

# plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(mf_train_losses, label='MF Train')
plt.plot(mf_val_losses, label='MF Val')
plt.plot(ncf_train_losses, label='NCF Train')
plt.plot(ncf_val_losses, label='NCF Val')
plt.plot(context_train_losses, label='Context Train')
plt.plot(context_val_losses, label='Context Val')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'training_curves.png'))
plt.close()

# save model comparison results
results = {
    'Matrix Factorization': {
        'test_loss': float(mf_results[0]),
        'rmse': float(mf_results[1]),
        'mae': float(mf_results[2])
    },
    'Neural Collaborative Filtering': {
        'test_loss': float(ncf_results[0]),
        'rmse': float(ncf_results[1]),
        'mae': float(ncf_results[2])
    },
    'Context-Aware Recommendation': {
        'test_loss': float(context_results[0]),
        'rmse': float(context_results[1]),
        'mae': float(context_results[2])
    }
}

# save results to JSON
with open(os.path.join(results_dir, 'model_comparison.json'), 'w') as f:
    json.dump(results, f, indent=4)

# create a summary table
summary_df = pd.DataFrame({
    'Model': ['Matrix Factorization', 'Neural Collaborative Filtering', 'Context-Aware Recommendation'],
    'Test Loss': [results['Matrix Factorization']['test_loss'],
                  results['Neural Collaborative Filtering']['test_loss'],
                  results['Context-Aware Recommendation']['test_loss']],
    'RMSE': [results['Matrix Factorization']['rmse'],
             results['Neural Collaborative Filtering']['rmse'],
             results['Context-Aware Recommendation']['rmse']],
    'MAE': [results['Matrix Factorization']['mae'],
            results['Neural Collaborative Filtering']['mae'],
            results['Context-Aware Recommendation']['mae']]
})

print("\nModel Comparison Summary:")
print(summary_df)

# plot comparison bar chart
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='RMSE', data=summary_df)
plt.title('Model Performance Comparison (RMSE)')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, summary_df['RMSE'].max() * 1.1)

plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='MAE', data=summary_df)
plt.title('Model Performance Comparison (MAE)')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, summary_df['MAE'].max() * 1.1)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'model_comparison_metrics.png'))
plt.close()

# generate sample recommendations for a few users
print("\nGenerating sample recommendations...")

# get a few sample users
sample_users = test_data[user_idx_col].sample(5).tolist()
all_business_indices = np.sort(test_data[business_idx_col].unique())

# for each sample user, get top 10 recommendations from each model
sample_recommendations = {}

for user_idx in sample_users:
    user_recommendations = {}

    # get recommendations from Matrix Factorization model
    mf_business_indices, mf_scores = generate_recommendations(
        mf_model, user_idx, all_business_indices, top_n=10
    )

    # get recommendations from Neural Collaborative Filtering model
    ncf_business_indices, ncf_scores = generate_recommendations(
        ncf_model, user_idx, all_business_indices, top_n=10
    )

    # get recommendations from Context-Aware model
    # use average context features for simplicity
    avg_context = test_data[context_features].mean().values
    context_business_indices, context_scores = generate_recommendations(
        context_model, user_idx, all_business_indices, top_n=10,
        use_context=True, context_data=avg_context
    )

    # store recommendations
    user_recommendations['Matrix Factorization'] = {
        'business_indices': mf_business_indices.tolist(),
        'scores': mf_scores.tolist()
    }

    user_recommendations['Neural Collaborative Filtering'] = {
        'business_indices': ncf_business_indices.tolist(),
        'scores': ncf_scores.tolist()
    }

    user_recommendations['Context-Aware Recommendation'] = {
        'business_indices': context_business_indices.tolist(),
        'scores': context_scores.tolist()
    }

    sample_recommendations[str(user_idx)] = user_recommendations

# save sample recommendations
with open(os.path.join(results_dir, 'sample_recommendations.json'), 'w') as f:
    json.dump(sample_recommendations, f, indent=4)

# compute model novelty and diversity metrics
print("\nComputing recommendation diversity and novelty metrics...")


def compute_recommendation_diversity(recommendations_list, all_items):
    """
    Compute diversity as the average pairwise Jaccard distance between users' recommendation lists
    """
    if len(recommendations_list) <= 1:
        return 0.0

    total_distance = 0.0
    pairs_count = 0

    for i in range(len(recommendations_list)):
        for j in range(i + 1, len(recommendations_list)):
            set_i = set(recommendations_list[i])
            set_j = set(recommendations_list[j])

            # jaccard distance = 1 - Jaccard similarity
            # jaccard similarity = |intersection| / |union|
            intersection = len(set_i.intersection(set_j))
            union = len(set_i.union(set_j))

            jaccard_distance = 1.0 - (intersection / union if union > 0 else 0.0)

            total_distance += jaccard_distance
            pairs_count += 1

    return total_distance / pairs_count if pairs_count > 0 else 0.0


def compute_recommendation_novelty(recommendations_list, item_popularity, total_items):
    """
    Compute novelty as the average information content of recommended items
    Higher values indicate more novel (less popular) items being recommended
    """
    if not recommendations_list:
        return 0.0

    novelty_scores = []

    for recommendations in recommendations_list:
        user_novelty = 0.0
        for item in recommendations:
            # calculate information content: -log2(popularity)
            popularity = item_popularity.get(item, 1.0 / total_items)
            user_novelty += -np.log2(popularity)

        # average novelty for this user's recommendations
        novelty_scores.append(user_novelty / len(recommendations) if recommendations else 0.0)

    # return average novelty across all users
    return np.mean(novelty_scores)


# calculate item popularity
item_popularity = {}
total_interactions = len(train_data)

for item_idx in train_data[business_idx_col]:
    item_popularity[item_idx] = item_popularity.get(item_idx, 0) + 1

# convert to probabilities
for item_idx in item_popularity:
    item_popularity[item_idx] /= total_interactions

# extract recommendation lists from sample_recommendations
mf_recommendations = [sample_recommendations[user]['Matrix Factorization']['business_indices']
                      for user in sample_recommendations]
ncf_recommendations = [sample_recommendations[user]['Neural Collaborative Filtering']['business_indices']
                       for user in sample_recommendations]
context_recommendations = [sample_recommendations[user]['Context-Aware Recommendation']['business_indices']
                           for user in sample_recommendations]

# calculate diversity and novelty
total_businesses = len(all_business_indices)

mf_diversity = compute_recommendation_diversity(mf_recommendations, total_businesses)
ncf_diversity = compute_recommendation_diversity(ncf_recommendations, total_businesses)
context_diversity = compute_recommendation_diversity(context_recommendations, total_businesses)

mf_novelty = compute_recommendation_novelty(mf_recommendations, item_popularity, total_businesses)
ncf_novelty = compute_recommendation_novelty(ncf_recommendations, item_popularity, total_businesses)
context_novelty = compute_recommendation_novelty(context_recommendations, item_popularity, total_businesses)

# add metrics to results
diversity_novelty_results = {
    'Matrix Factorization': {
        'diversity': float(mf_diversity),
        'novelty': float(mf_novelty)
    },
    'Neural Collaborative Filtering': {
        'diversity': float(ncf_diversity),
        'novelty': float(ncf_novelty)
    },
    'Context-Aware Recommendation': {
        'diversity': float(context_diversity),
        'novelty': float(context_novelty)
    }
}

# save diversity and novelty results
with open(os.path.join(results_dir, 'diversity_novelty_metrics.json'), 'w') as f:
    json.dump(diversity_novelty_results, f, indent=4)

# update the summary table
summary_df['Diversity'] = [
    diversity_novelty_results['Matrix Factorization']['diversity'],
    diversity_novelty_results['Neural Collaborative Filtering']['diversity'],
    diversity_novelty_results['Context-Aware Recommendation']['diversity']
]

summary_df['Novelty'] = [
    diversity_novelty_results['Matrix Factorization']['novelty'],
    diversity_novelty_results['Neural Collaborative Filtering']['novelty'],
    diversity_novelty_results['Context-Aware Recommendation']['novelty']
]

print("\nUpdated Model Comparison Summary with Diversity and Novelty:")
print(summary_df)

# plot diversity and novelty comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='Diversity', data=summary_df)
plt.title('Model Recommendation Diversity')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.1)

plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='Novelty', data=summary_df)
plt.title('Model Recommendation Novelty')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'diversity_novelty_comparison.png'))
plt.close()

# save models
print("\nSaving models...")
torch.save(mf_model.state_dict(), os.path.join(results_dir, 'matrix_factorization_model.pt'))
torch.save(ncf_model.state_dict(), os.path.join(results_dir, 'neural_collaborative_filtering_model.pt'))
torch.save(context_model.state_dict(), os.path.join(results_dir, 'context_aware_recommendation_model.pt'))

# save DataFrame with business information for easier recommendation lookup
business_info = data[[business_idx_col, 'business_id']].drop_duplicates()
if 'name' in data.columns:
    business_info = pd.merge(business_info, data[['business_id', 'name']].drop_duplicates(), on='business_id')

business_info.to_csv(os.path.join(results_dir, 'business_info.csv'), index=False)

# save context scaler
import joblib

joblib.dump(context_scaler, os.path.join(results_dir, 'context_scaler.joblib'))

# Create recommendation script content
recommendation_script = '''
import torch
import numpy as np
import pandas as pd
from cf_evel import MatrixFactorization, NeuralCollaborativeFiltering, ContextAwareRecommender
import joblib
import os

class YelpRecommender:
    def __init__(self, model_path, model_type='ContextAware', num_users=1000, num_items=1000, scaler_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create the appropriate model instance
        if model_type == 'ContextAware':
            self.model = ContextAwareRecommender(num_users=num_users, num_items=num_items, context_dim=5)
        elif model_type == 'NCF':
            self.model = NeuralCollaborativeFiltering(num_users=num_users, num_items=num_items)
        else:  # MF
            self.model = MatrixFactorization(num_users=num_users, num_items=num_items)
            
        # Load the saved weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        if scaler_path:
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = None
            
    def get_recommendations(self, user_idx, business_indices, context_data=None, top_n=10):
        with torch.no_grad():
            # Convert to tensors
            user_indices = torch.tensor([user_idx] * len(business_indices), dtype=torch.long).to(self.device)
            business_indices = torch.tensor(business_indices, dtype=torch.long).to(self.device)
            
            if context_data is not None and self.scaler is not None:
                context_data = self.scaler.transform(context_data)
                context_data = torch.tensor(np.repeat(context_data.reshape(1, -1), len(business_indices), axis=0), 
                                           dtype=torch.float).to(self.device)
                predictions = self.model(user_indices, business_indices, context_data)
            else:
                predictions = self.model(user_indices, business_indices)
                
            # Get top-n businesses
            _, indices = torch.topk(predictions, min(top_n, len(predictions)))
            return business_indices[indices].cpu().numpy()
'''

# Save the recommendation script
with open(os.path.join(results_dir, 'recommend.py'), 'w') as f:
    f.write(recommendation_script)

print("\nAll models trained, evaluated, and saved.")
print(f"Results saved to {results_dir} directory.")
print("A recommendation script has been created for easy future use.")

# print final summary
print("\n=== Recommendation System Summary ===")
print(f"Number of users: {num_users}")
print(f"Number of businesses: {num_businesses}")
print(f"Training examples: {len(train_data)}")
print(f"Validation examples: {len(val_data)}")
print(f"Testing examples: {len(test_data)}")
print("\nModel Performance:")
for model_name in results:
    print(f"  {model_name}:")
    print(f"    RMSE: {results[model_name]['rmse']:.4f}")
    print(f"    MAE: {results[model_name]['mae']:.4f}")

print("\nRecommendation Diversity and Novelty:")
for model_name in diversity_novelty_results:
    print(f"  {model_name}:")
    print(f"    Diversity: {diversity_novelty_results[model_name]['diversity']:.4f}")
    print(f"    Novelty: {diversity_novelty_results[model_name]['novelty']:.4f}")

# add additional visualizations for model comparison
def create_enhanced_visualizations(results, summary_df, results_dir):
    """
    Create enhanced visualizations for model comparison
    """
    print("\nGenerating enhanced visualizations...")
    
    # 1. Radar chart for model metrics
    plt.figure(figsize=(10, 8))
    
    # Prepare radar chart
    categories = ['RMSE', 'MAE', 'Diversity', 'Novelty']
    N = len(categories)
    
    # Scale values to be between 0 and 1 for better radar chart visualization
    # Lower RMSE and MAE are better, so invert those
    max_rmse = summary_df['RMSE'].max()
    max_mae = summary_df['MAE'].max()
    
    radar_values = []
    for _, row in summary_df.iterrows():
        values = [
            1 - (row['RMSE'] / max_rmse),  # Lower RMSE is better, so invert
            1 - (row['MAE'] / max_mae),    # Lower MAE is better, so invert
            row['Diversity'],              # Higher diversity is better
            row['Novelty'] / summary_df['Novelty'].max()  # Normalize novelty
        ]
        radar_values.append(values)
    
    # Calculate angles for radar chart
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Create radar chart
    ax = plt.subplot(111, polar=True)
    
    # Plot each model
    colors = ['blue', 'green', 'red']
    models = summary_df['Model'].tolist()
    
    for i, (values, model, color) in enumerate(zip(radar_values, models, colors)):
        values += values[:1]  # Close the polygon
        ax.plot(angles, values, color=color, linewidth=2, label=model)
        ax.fill(angles, values, color=color, alpha=0.1)
    
    # Set category labels
    plt.xticks(angles[:-1], categories)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_radar_comparison.png'))
    plt.close()
    
    # 2. Training loss curves with smoothing
    plt.figure(figsize=(12, 6))
    
    # Apply simple moving average for smoothing
    def smooth(y, box_pts=3):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    # Plot original and smoothed training loss
    plt.subplot(1, 2, 1)
    plt.plot(mf_train_losses, 'b-', alpha=0.3, label='MF Train (raw)')
    plt.plot(smooth(mf_train_losses), 'b-', label='MF Train (smoothed)')
    plt.plot(ncf_train_losses, 'g-', alpha=0.3, label='NCF Train (raw)')
    plt.plot(smooth(ncf_train_losses), 'g-', label='NCF Train (smoothed)')
    plt.plot(context_train_losses, 'r-', alpha=0.3, label='Context Train (raw)') 
    plt.plot(smooth(context_train_losses), 'r-', label='Context Train (smoothed)')
    plt.title('Training Loss (Smoothed)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot original and smoothed validation loss
    plt.subplot(1, 2, 2)
    plt.plot(mf_val_losses, 'b-', alpha=0.3, label='MF Val (raw)')
    plt.plot(smooth(mf_val_losses), 'b-', label='MF Val (smoothed)')
    plt.plot(ncf_val_losses, 'g-', alpha=0.3, label='NCF Val (raw)')
    plt.plot(smooth(ncf_val_losses), 'g-', label='NCF Val (smoothed)')
    plt.plot(context_val_losses, 'r-', alpha=0.3, label='Context Val (raw)')
    plt.plot(smooth(context_val_losses), 'r-', label='Context Val (smoothed)')
    plt.title('Validation Loss (Smoothed)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'smoothed_training_curves.png'))
    plt.close()
    
    # 3. Heatmap of recommendation overlap between models
    def compute_overlap(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    # Get sample recommendations for overlap calculation
    sample_users = list(sample_recommendations.keys())
    if sample_users:
        overlap_matrix = np.zeros((3, 3))
        model_names = ['Matrix Factorization', 'Neural Collaborative Filtering', 'Context-Aware Recommendation']
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    overlap_matrix[i, j] = 1.0  # Perfect overlap with self
                else:
                    overlaps = []
                    for user in sample_users:
                        recs1 = sample_recommendations[user][model1]['business_indices']
                        recs2 = sample_recommendations[user][model2]['business_indices']
                        overlaps.append(compute_overlap(recs1, recs2))
                    overlap_matrix[i, j] = np.mean(overlaps)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(overlap_matrix, annot=True, fmt=".2f", xticklabels=model_names, yticklabels=model_names, cmap="YlGnBu")
        plt.title('Recommendation Overlap Between Models')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'model_overlap_heatmap.png'))
        plt.close()
    
    # 4. User-Business distribution
    plt.figure(figsize=(12, 6))
    
    # Plot ratings distribution
    plt.subplot(1, 2, 1)
    sns.countplot(x='stars', data=data)
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    
    # Plot items per user histogram
    plt.subplot(1, 2, 2)
    user_counts = data[user_idx_col].value_counts()
    sns.histplot(user_counts, bins=20)
    plt.title('Items Rated per User')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'data_distribution.png'))
    plt.close()
    
    print("Enhanced visualizations have been generated.")

# Generate enhanced visualizations
create_enhanced_visualizations(results, summary_df, results_dir)