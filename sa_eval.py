import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import re
import os
import json
from datetime import datetime
import torch.nn.functional as F

# download NLTK resources
print("Downloading NLTK resources...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("NLTK resources downloaded successfully.")
except Exception as e:
    print(f"Warning: Could not download NLTK resources: {e}")
    print("Will use a simple tokenization method instead.")

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

# define features for sentiment analysis
text_features = [
    'text_length', 'word_count',
    'avg_word_length', 'exclamation_count',
    'question_count', 'uppercase_ratio'
]

# context features
context_features = [
    'latitude', 'longitude',
    'year', 'month', 'day_of_week'
]

# all features and target
feature_columns = text_features + context_features
target_column = 'stars'

# ensure all feature columns exist and convert to numeric
for col in feature_columns:
    if col not in data.columns:
        print(f"Warning: Column {col} not found in data. Creating with zeros.")
        data[col] = 0.0
    else:
        # convert to numeric, coerce errors to NaN
        data[col] = pd.to_numeric(data[col], errors='coerce')
        # fill NaN values with 0
        data[col] = data[col].fillna(0.0)

# ensure target column is numeric
data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
data[target_column] = data[target_column].fillna(data[target_column].mean())

# function to preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    try:
        # convert to lowercase
        text = text.lower()
        # remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # tokenize
        tokens = word_tokenize(text)
        return " ".join(tokens)
    except Exception as e:
        print(f"Warning: Error in text preprocessing: {e}")
        return text.lower()  # fallback to simple lowercase if tokenization fails

# apply text preprocessing
print("Preprocessing text...")
data['processed_text'] = data['text'].apply(preprocess_text)

# ensure the date column exists and is datetime
if 'date' in data.columns:
    try:
        # convert date to datetime
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        # sort by date
        data = data.sort_values('date')
    except Exception as e:
        print(f"Error converting date column: {e}")
        # create a dummy date column for chronological splitting
        data['date'] = pd.date_range(start='2010-01-01', periods=len(data), freq='D')
else:
    print("Date column not found. Creating dummy dates.")
    data['date'] = pd.date_range(start='2010-01-01', periods=len(data), freq='D')

# time-based split
print("Splitting data chronologically...")
# split data: 70% train, 15% validation, 15% test
train_size = 0.7
val_size = 0.15
test_size = 0.15

train_end_idx = int(len(data) * train_size)
val_end_idx = int(len(data) * (train_size + val_size))

train_data = data.iloc[:train_end_idx].copy()
val_data = data.iloc[train_end_idx:val_end_idx].copy()
test_data = data.iloc[val_end_idx:].copy()

print(f"Train data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")
print(f"Test data size: {len(test_data)}")

# build vocabulary from training data
print("Building vocabulary...")


def build_vocabulary(texts, max_vocab_size=1000):
    """build a vocabulary from a list of texts."""
    word_counts = Counter()
    for text in tqdm(texts):
        if isinstance(text, str):
            tokens = text.split()
            word_counts.update(tokens)

    # get most common words
    vocab = ['<pad>', '<unk>'] + [word for word, _ in word_counts.most_common(max_vocab_size - 2)]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word_to_idx


vocab, word_to_idx = build_vocabulary(train_data['processed_text'], max_vocab_size=1000)
print(f"Vocabulary size: {len(vocab)}")


# function to convert text to indices
def text_to_indices(text, word_to_idx, max_length=100):
    """convert text to a list of indices."""
    if not isinstance(text, str):
        return [0] * max_length  # Return all padding if text is not a string

    tokens = text.split()
    indices = [word_to_idx.get(token, word_to_idx['<unk>']) for token in tokens[:max_length]]
    # pad sequence
    if len(indices) < max_length:
        indices += [word_to_idx['<pad>']] * (max_length - len(indices))
    return indices


# apply text to indices conversion
print("Converting text to indices...")
train_data.loc[:, 'text_indices'] = train_data['processed_text'].apply(lambda x: text_to_indices(x, word_to_idx))
val_data.loc[:, 'text_indices'] = val_data['processed_text'].apply(lambda x: text_to_indices(x, word_to_idx))
test_data.loc[:, 'text_indices'] = test_data['processed_text'].apply(lambda x: text_to_indices(x, word_to_idx))

# scale numerical features
print("Scaling numerical features...")
scaler = StandardScaler()
train_data.loc[:, feature_columns] = scaler.fit_transform(train_data[feature_columns])
val_data.loc[:, feature_columns] = scaler.transform(val_data[feature_columns])
test_data.loc[:, feature_columns] = scaler.transform(test_data[feature_columns])


# define PyTorch Dataset
class YelpDataset(Dataset):
    def __init__(self, dataframe, numerical_features, target_column, use_text=True):
        self.dataframe = dataframe
        self.numerical_features = numerical_features
        self.target_column = target_column
        self.use_text = use_text

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # numerical features
        numerical_data = torch.tensor(
            self.dataframe.iloc[idx][self.numerical_features].values.astype(np.float32),
            dtype=torch.float32
        )

        # target
        target = torch.tensor(self.dataframe.iloc[idx][self.target_column], dtype=torch.float32)

        # text indices
        if self.use_text:
            # convert string or list representation of indices to actual list of integers
            text_indices = self.dataframe.iloc[idx]['text_indices']
            if isinstance(text_indices, str):
                # if stored as string (e.g., "[1, 2, 3]"), convert to list using json.loads
                text_indices = json.loads(text_indices)
            text_indices = torch.tensor(text_indices, dtype=torch.long)
            return {'numerical': numerical_data, 'text': text_indices, 'target': target}
        else:
            return {'numerical': numerical_data, 'target': target}


# define Model Architectures

# 1. Baseline MLP Model (only numerical features)
class MLPBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
        super(MLPBaseline, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # output layer - regression for star rating
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()


# 2. CNN Text Model with Numerical Features
class CNNTextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, numerical_dim,
                 hidden_dims=[64, 32], dropout=0.3):
        super(CNNTextModel, self).__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # conv layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        # fully connected layers for text features
        text_output_dim = num_filters * len(filter_sizes)

        # combined layers for text + numerical
        combined_input_dim = text_output_dim + numerical_dim

        # mLP layers
        layers = []
        prev_dim = combined_input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # output layer - regression for star rating
        layers.append(nn.Linear(prev_dim, 1))

        self.fc = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, numerical):
        # text processing
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(text)

        # (batch_size, seq_len, embedding_dim) -> (batch_size, embedding_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)

        # apply convolutions and max pooling
        # [(batch_size, num_filters), ...] for each filter size
        conved = [F.relu(conv(embedded)).max(dim=2)[0] for conv in self.convs]

        # concatenate convolutional outputs
        # (batch_size, num_filters * len(filter_sizes))
        conved = torch.cat(conved, dim=1)

        # combine with numerical features
        combined = torch.cat((conved, numerical), dim=1)

        # apply dropout
        combined = self.dropout(combined)

        # final fully connected layers
        output = self.fc(combined)

        return output.squeeze()


# 3. LSTM Text Model with Numerical Features
class LSTMTextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, numerical_dim,
                 num_layers=1, bidirectional=True, hidden_dims=[64, 32], dropout=0.3):
        super(LSTMTextModel, self).__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # lSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # determine LSTM output dim
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # combined layers for LSTM + numerical
        combined_input_dim = lstm_output_dim + numerical_dim

        # mLP layers
        layers = []
        prev_dim = combined_input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # output layer - regression for star rating
        layers.append(nn.Linear(prev_dim, 1))

        self.fc = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, numerical):
        # text processing
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(text)

        # apply dropout to embeddings
        embedded = self.dropout(embedded)

        # pass through LSTM
        # output: (batch_size, seq_len, hidden_dim*num_directions)
        # hidden: (num_layers*num_directions, batch_size, hidden_dim)
        output, (hidden, cell) = self.lstm(embedded)

        # use last hidden state
        if self.lstm.bidirectional:
            # concatenate the final forward and backward hidden states
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        # combine with numerical features
        combined = torch.cat((hidden, numerical), dim=1)

        # apply dropout
        combined = self.dropout(combined)

        # final fully connected layers
        output = self.fc(combined)

        return output.squeeze()


# create datasets
train_dataset = YelpDataset(train_data, feature_columns, target_column, use_text=True)
val_dataset = YelpDataset(val_data, feature_columns, target_column, use_text=True)
test_dataset = YelpDataset(test_data, feature_columns, target_column, use_text=True)

# also create a version without text for baseline model
train_dataset_notext = YelpDataset(train_data, feature_columns, target_column, use_text=False)
val_dataset_notext = YelpDataset(val_data, feature_columns, target_column, use_text=False)
test_dataset_notext = YelpDataset(test_data, feature_columns, target_column, use_text=False)

# create dataloaders
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

train_loader_notext = DataLoader(train_dataset_notext, batch_size=batch_size, shuffle=True)
val_loader_notext = DataLoader(val_dataset_notext, batch_size=batch_size)
test_loader_notext = DataLoader(test_dataset_notext, batch_size=batch_size)


# define training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, use_text=True):
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
            if use_text:
                numerical_data = batch['numerical'].to(device)
                text_data = batch['text'].to(device)
                targets = batch['target'].to(device)

                # forward pass
                outputs = model(text_data, numerical_data)
            else:
                numerical_data = batch['numerical'].to(device)
                targets = batch['target'].to(device)

                # forward pass
                outputs = model(numerical_data)

            # calculate loss
            loss = criterion(outputs, targets)

            # backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(targets)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # validation
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation'):
                # get data
                if use_text:
                    numerical_data = batch['numerical'].to(device)
                    text_data = batch['text'].to(device)
                    targets = batch['target'].to(device)

                    # forward pass
                    outputs = model(text_data, numerical_data)
                else:
                    numerical_data = batch['numerical'].to(device)
                    targets = batch['target'].to(device)

                    # forward pass
                    outputs = model(numerical_data)

                # calculate loss
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * len(targets)

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


# define evaluation function
def evaluate_model(model, test_loader, criterion, use_text=True):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            # get data
            if use_text:
                numerical_data = batch['numerical'].to(device)
                text_data = batch['text'].to(device)
                targets = batch['target'].to(device)

                # forward pass
                outputs = model(text_data, numerical_data)
            else:
                numerical_data = batch['numerical'].to(device)
                targets = batch['target'].to(device)

                # forward pass
                outputs = model(numerical_data)

            # calculate loss
            loss = criterion(outputs, targets)
            running_loss += loss.item() * len(targets)

            # save predictions and targets for metrics
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)

    # convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # calculate metrics
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

    # convert to classification for classification metrics
    # round to nearest integer and clip to range [1, 5]
    all_preds_class = np.round(all_preds).clip(1, 5).astype(int)
    all_targets_class = np.round(all_targets).astype(int)
    accuracy = accuracy_score(all_targets_class, all_preds_class)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test RMSE: {rmse:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')
    print('\nClassification Report:')
    print(classification_report(all_targets_class, all_preds_class))

    # create confusion matrix
    plt.figure(figsize=(8, 6))
    conf_mat = pd.crosstab(
        pd.Series(all_targets_class, name='Actual'),
        pd.Series(all_preds_class, name='Predicted')
    )
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model.__class__.__name__}.png')
    plt.close()

    # plot prediction distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(all_preds, bins=20, alpha=0.5, label='Predicted')
    sns.histplot(all_targets, bins=20, alpha=0.5, label='Actual')
    plt.title('Distribution of Actual vs Predicted Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'rating_distribution_{model.__class__.__name__}.png')
    plt.close()

    return test_loss, rmse, accuracy, all_preds, all_targets


# create results directory
results_dir = 'sa_results'
os.makedirs(results_dir, exist_ok=True)

# initialize models
# 1. Baseline MLP Model
print("\n--- Training Baseline MLP Model ---")
mlp_baseline = MLPBaseline(
    input_dim=len(feature_columns),
    hidden_dims=[128, 64, 32],
    dropout=0.3
).to(device)

# 2. CNN Text Model
print("\n--- Training CNN Text Model ---")
cnn_text_model = CNNTextModel(
    vocab_size=len(vocab),
    embedding_dim=100,
    num_filters=100,
    filter_sizes=[3, 4, 5],
    numerical_dim=len(feature_columns),
    hidden_dims=[128, 64],
    dropout=0.3
).to(device)

# 3. LSTM Text Model
print("\n--- Training LSTM Text Model ---")
lstm_text_model = LSTMTextModel(
    vocab_size=len(vocab),
    embedding_dim=100,
    hidden_dim=128,
    numerical_dim=len(feature_columns),
    num_layers=2,
    bidirectional=True,
    hidden_dims=[128, 64],
    dropout=0.3
).to(device)

# define loss function and optimizers
criterion = nn.MSELoss()

optimizer_mlp = optim.Adam(mlp_baseline.parameters(), lr=0.001)
optimizer_cnn = optim.Adam(cnn_text_model.parameters(), lr=0.001)
optimizer_lstm = optim.Adam(lstm_text_model.parameters(), lr=0.001)

# train models
print("\nTraining Baseline MLP Model...")
mlp_baseline, mlp_train_losses, mlp_val_losses = train_model(
    mlp_baseline, train_loader_notext, val_loader_notext,
    criterion, optimizer_mlp, num_epochs=3, use_text=False
)

print("\nTraining CNN Text Model...")
cnn_text_model, cnn_train_losses, cnn_val_losses = train_model(
    cnn_text_model, train_loader, val_loader,
    criterion, optimizer_cnn, num_epochs=3, use_text=True
)

print("\nTraining LSTM Text Model...")
lstm_text_model, lstm_train_losses, lstm_val_losses = train_model(
    lstm_text_model, train_loader, val_loader,
    criterion, optimizer_lstm, num_epochs=3, use_text=True
)

# evaluate models
print("\nEvaluating Baseline MLP Model...")
mlp_results = evaluate_model(mlp_baseline, test_loader_notext, criterion, use_text=False)

print("\nEvaluating CNN Text Model...")
cnn_results = evaluate_model(cnn_text_model, test_loader, criterion, use_text=True)

print("\nEvaluating LSTM Text Model...")
lstm_results = evaluate_model(lstm_text_model, test_loader, criterion, use_text=True)

# plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(mlp_train_losses, label='MLP Train')
plt.plot(mlp_val_losses, label='MLP Val')
plt.plot(cnn_train_losses, label='CNN Train')
plt.plot(cnn_val_losses, label='CNN Val')
plt.plot(lstm_train_losses, label='LSTM Train')
plt.plot(lstm_val_losses, label='LSTM Val')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'training_curves.png'))
plt.close()

# save model comparison results
results = {
    'MLP Baseline': {
        'test_loss': float(mlp_results[0]),
        'rmse': float(mlp_results[1]),
        'accuracy': float(mlp_results[2])
    },
    'CNN Text Model': {
        'test_loss': float(cnn_results[0]),
        'rmse': float(cnn_results[1]),
        'accuracy': float(cnn_results[2])
    },
    'LSTM Text Model': {
        'test_loss': float(lstm_results[0]),
        'rmse': float(lstm_results[1]),
        'accuracy': float(lstm_results[2])
    },
}

# save results to JSON
with open(os.path.join(results_dir, 'model_comparison.json'), 'w') as f:
    json.dump(results, f, indent=4)

# create a summary table
summary_df = pd.DataFrame({
    'Model': ['MLP Baseline', 'CNN Text Model', 'LSTM Text Model'],
    'Test Loss': [results['MLP Baseline']['test_loss'],
                  results['CNN Text Model']['test_loss'],
                  results['LSTM Text Model']['test_loss']],
    'RMSE': [results['MLP Baseline']['rmse'],
             results['CNN Text Model']['rmse'],
             results['LSTM Text Model']['rmse']],
    'Accuracy': [results['MLP Baseline']['accuracy'],
                 results['CNN Text Model']['accuracy'],
                 results['LSTM Text Model']['accuracy']]
})

print("\nModel Comparison Summary:")
print(summary_df)

# plot comparison bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='RMSE', data=summary_df)
plt.title('Model Performance Comparison (RMSE)')
plt.ylim(0, summary_df['RMSE'].max() * 1.1)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'model_comparison_rmse.png'))
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=summary_df)
plt.title('Model Performance Comparison (Accuracy)')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'model_comparison_accuracy.png'))
plt.close()

# save models
print("\nSaving models...")
torch.save(mlp_baseline.state_dict(), os.path.join(results_dir, 'mlp_baseline.pt'))
torch.save(cnn_text_model.state_dict(), os.path.join(results_dir, 'cnn_text_model.pt'))
torch.save(lstm_text_model.state_dict(), os.path.join(results_dir, 'lstm_text_model.pt'))

# save preprocessing resources
with open(os.path.join(results_dir, 'word_to_idx.json'), 'w') as f:
    json.dump(word_to_idx, f)

# save scaler
import joblib

joblib.dump(scaler, os.path.join(results_dir, 'feature_scaler.joblib'))

print("\nAll models trained, evaluated, and saved.")
print(f"Results saved to {results_dir} directory.")