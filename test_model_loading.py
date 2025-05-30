import torch
import numpy as np
import os
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Try importing the model classes
print("\nImporting model classes...")
try:
    from sa_eval import LSTMTextModel, CNNTextModel, MLPBaseline
    print("Successfully imported sentiment models")
except Exception as e:
    print(f"Error importing sentiment models: {str(e)}")
    
try:
    from cf_evel import MatrixFactorization, NeuralCollaborativeFiltering, ContextAwareRecommender
    print("Successfully imported recommender models")
except Exception as e:
    print(f"Error importing recommender models: {str(e)}")

# Try instantiating the models
print("\nInstantiating models...")
try:
    lstm_model = LSTMTextModel(
        vocab_size=1000,
        embedding_dim=100,
        hidden_dim=128,
        numerical_dim=11
    )
    print("Successfully instantiated LSTM model")
except Exception as e:
    print(f"Error instantiating LSTM model: {str(e)}")
    
try:
    cnn_model = CNNTextModel(
        vocab_size=1000,
        embedding_dim=100,
        num_filters=100,
        filter_sizes=[3, 4, 5],
        numerical_dim=11
    )
    print("Successfully instantiated CNN model")
except Exception as e:
    print(f"Error instantiating CNN model: {str(e)}")
    
try:
    mlp_model = MLPBaseline(input_dim=11)
    print("Successfully instantiated MLP model")
except Exception as e:
    print(f"Error instantiating MLP model: {str(e)}")
    
try:
    mf_model = MatrixFactorization(
        num_users=100,
        num_items=100
    )
    print("Successfully instantiated MF model")
except Exception as e:
    print(f"Error instantiating MF model: {str(e)}")
    
try:
    ncf_model = NeuralCollaborativeFiltering(
        num_users=100,
        num_items=100
    )
    print("Successfully instantiated NCF model")
except Exception as e:
    print(f"Error instantiating NCF model: {str(e)}")
    
try:
    context_model = ContextAwareRecommender(
        num_users=100,
        num_items=100,
        context_dim=5
    )
    print("Successfully instantiated Context model")
except Exception as e:
    print(f"Error instantiating Context model: {str(e)}")

print("\nTest completed!") 