import torch
import numpy as np
import pandas as pd
import os
import sys
import traceback
import time

# Set a module-level name to prevent running the main script in imported files
__name__ = "run_recommender_test"

print("Testing the recommender system...")
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

print("\nImporting classes...")
try:
    # Override __name__ in imported modules to prevent training
    import builtins
    original_import = builtins.__import__
    
    def custom_import(name, *args, **kwargs):
        module = original_import(name, *args, **kwargs)
        if name in ['sa_eval', 'cf_evel']:
            module.__name__ = "__imported_module__"
        return module
    
    builtins.__import__ = custom_import
    
    # Now import our recommender
    from sentiment_aware_recommender import SentimentAwareRecommender
    print("Successfully imported SentimentAwareRecommender class")
except Exception as e:
    print(f"Error importing SentimentAwareRecommender: {str(e)}")
    traceback.print_exc()
    sys.exit(1)
    
# Restore original import
builtins.__import__ = original_import

print("\nChecking for data files...")
if os.path.exists('processed_yelp_data.csv'):
    print("Found processed_yelp_data.csv")
else:
    print("Warning: processed_yelp_data.csv not found")

if os.path.exists('sa_results/best_model.pt'):
    print("Found sa_results/best_model.pt")
else:
    print("Warning: sa_results/best_model.pt not found")

if os.path.exists('cf_results/best_model.pt'):
    print("Found cf_results/best_model.pt")
else:
    print("Warning: cf_results/best_model.pt not found")

print("\nCreating recommender instance...")
start_time = time.time()
try:
    recommender = SentimentAwareRecommender(
        data_path='processed_yelp_data.csv',
        sentiment_model_path='sa_results/best_model.pt',
        recommender_model_path='cf_results/best_model.pt'
    )
    print(f"Successfully created recommender in {time.time() - start_time:.2f} seconds")
    
    print("\nSome basic info about the recommender:")
    print(f"Number of users: {len(recommender.user_id_map)}")
    print(f"Number of businesses: {len(recommender.business_id_map)}")
    print(f"Sentiment model type: {recommender.sentiment_model_type}")
    print(f"Recommender model type: {recommender.recommender_model_type}")
    
    print("\nTest completed successfully!")
except Exception as e:
    print(f"Error creating recommender: {str(e)}")
    traceback.print_exc() 