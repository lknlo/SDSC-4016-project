import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict, Tuple

class YelpRecommender:
    def __init__(self, data_path: str = 'processed_yelp_data.csv'):
        """
        Initialize the recommender with the preprocessed data.
        
        Args:
            data_path (str): Path to the processed data CSV file
        """
        # Load the processed data
        self.data = pd.read_csv(data_path)
        
        # Load the ID mappings
        with open('user_id_map.json', 'r') as f:
            self.user_id_map = json.load(f)
        with open('business_id_map.json', 'r') as f:
            self.business_id_map = json.load(f)
            
        # Create reverse mappings for easier lookup
        self.user_idx_map = {v: k for k, v in self.user_id_map.items()}
        self.business_idx_map = {v: k for k, v in self.business_id_map.items()}
        
        # Create user-business rating matrix
        self.rating_matrix = self._create_rating_matrix()
        
        # Calculate user similarities
        self.user_similarities = self._calculate_user_similarities()
        
    def _create_rating_matrix(self) -> pd.DataFrame:
        """Create a user-business rating matrix."""
        # Pivot the data to create a user-business rating matrix
        rating_matrix = self.data.pivot(
            index='user_idx',
            columns='business_idx',
            values='stars'
        ).fillna(0)
        return rating_matrix
    
    def _calculate_user_similarities(self) -> np.ndarray:
        """Calculate cosine similarities between users."""
        # Convert rating matrix to numpy array
        ratings = self.rating_matrix.values
        
        # Calculate cosine similarities
        similarities = cosine_similarity(ratings)
        return similarities
    
    def get_user_recommendations(self, user_id: str, n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """
        Get recommendations for a specific user.
        
        Args:
            user_id (str): The user ID to get recommendations for
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            List of tuples containing (business_id, predicted_rating)
        """
        # Convert user_id to index
        user_idx = self.user_id_map.get(user_id)
        if user_idx is None:
            raise ValueError(f"User ID {user_id} not found in the dataset")
            
        # Get user's ratings
        user_ratings = self.rating_matrix.loc[user_idx]
        
        # Find businesses the user hasn't rated
        unrated_businesses = user_ratings[user_ratings == 0].index
        
        # Get similar users
        user_similarity = self.user_similarities[user_idx]
        
        # Calculate predicted ratings for unrated businesses
        predictions = []
        for business_idx in unrated_businesses:
            # Get ratings for this business from similar users
            business_ratings = self.rating_matrix[business_idx]
            rated_by_similar = business_ratings[business_ratings > 0]
            
            if len(rated_by_similar) > 0:
                # Calculate weighted average rating
                similar_users = rated_by_similar.index
                weights = user_similarity[similar_users]
                weighted_ratings = business_ratings[similar_users] * weights
                predicted_rating = weighted_ratings.sum() / weights.sum()
                
                predictions.append((business_idx, predicted_rating))
        
        # Sort predictions by rating and get top n
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:n_recommendations]
        
        # Convert business indices back to IDs
        return [(self.business_idx_map[biz_idx], rating) for biz_idx, rating in top_predictions]
    
    def get_business_recommendations(self, business_id: str, n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """
        Get similar businesses based on user ratings.
        
        Args:
            business_id (str): The business ID to find similar businesses for
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            List of tuples containing (business_id, similarity_score)
        """
        # Convert business_id to index
        business_idx = self.business_id_map.get(business_id)
        if business_idx is None:
            raise ValueError(f"Business ID {business_id} not found in the dataset")
            
        # Calculate business similarities
        business_similarities = cosine_similarity(self.rating_matrix.T)
        
        # Get similarities for the target business
        business_similarity = business_similarities[business_idx]
        
        # Sort businesses by similarity (excluding the business itself)
        similar_businesses = []
        for idx, similarity in enumerate(business_similarity):
            if idx != business_idx:
                similar_businesses.append((idx, similarity))
        
        similar_businesses.sort(key=lambda x: x[1], reverse=True)
        top_similar = similar_businesses[:n_recommendations]
        
        # Convert business indices back to IDs
        return [(self.business_idx_map[biz_idx], similarity) for biz_idx, similarity in top_similar]

def main():
    # Initialize the recommender
    recommender = YelpRecommender()
    
    # Example usage
    try:
        # Get a sample user ID from the dataset
        sample_user_id = list(recommender.user_id_map.keys())[0]
        print(f"\nGetting recommendations for user: {sample_user_id}")
        
        # Get user recommendations
        user_recommendations = recommender.get_user_recommendations(sample_user_id)
        print("\nTop 5 business recommendations for this user:")
        for business_id, predicted_rating in user_recommendations:
            print(f"Business ID: {business_id}, Predicted Rating: {predicted_rating:.2f}")
        
        # Get a sample business ID from the dataset
        sample_business_id = list(recommender.business_id_map.keys())[0]
        print(f"\nGetting similar businesses to: {sample_business_id}")
        
        # Get business recommendations
        business_recommendations = recommender.get_business_recommendations(sample_business_id)
        print("\nTop 5 similar businesses:")
        for business_id, similarity in business_recommendations:
            print(f"Business ID: {business_id}, Similarity Score: {similarity:.2f}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 