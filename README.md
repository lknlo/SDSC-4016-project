# Sentiment-Aware Recommender System

This project implements a hybrid recommender system that combines collaborative filtering with sentiment analysis to provide personalized business recommendations using Yelp data. The system leverages both user ratings and review text to enhance recommendation quality.

## Features

- **Hybrid Recommendation**: Integrates collaborative filtering (Matrix Factorization, Neural Collaborative Filtering, Context-Aware models) with sentiment analysis (LSTM, CNN, MLP).
- **Sentiment Analysis**: Predicts review sentiment using deep learning models.
- **Context Awareness**: Supports recommendations based on temporal and spatial context.
- **Visualization**: Generates visualizations for model performance and sentiment trends.
- **Extensive Preprocessing**: Includes scripts for data cleaning, feature extraction, and exploratory analysis.

## Project Structure

- `sentiment_aware_recommender.py`: Main class for the hybrid recommender system.
- `recommendation_script.py`: Standalone collaborative filtering recommender.
- `sa_eval.py`: Sentiment analysis models, training, and evaluation utilities.
- `preprocess.py`: Data preprocessing and feature extraction.
- `cf_results/`: Contains collaborative filtering results and business info.
- `run_recommender.py`: Script to test and run the recommender system.
- `final_test.py`, `test_model_loading.py`: Testing and validation scripts.

## Setup

### Prerequisites

- Python 3.7+
- PyTorch
- pandas, numpy, scikit-learn, matplotlib, seaborn, tqdm, nltk, polars, cartopy

Install dependencies with:

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn tqdm nltk polars cartopy