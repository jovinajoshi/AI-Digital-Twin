"""
Data Processing Module for AI Digital Twin Creator

This module handles data collection, preprocessing, and feature extraction
for user behavior analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json


class DataProcessor:
    """Processes and prepares user data for ML models."""
    
    def __init__(self):
        self.user_data = None
        self.processed_data = None
        
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Load user activity data from file or generate sample data.
        
        Args:
            filepath: Path to CSV file with user data
            
        Returns:
            DataFrame with user activity data
        """
        if filepath:
            self.user_data = pd.read_csv(filepath)
        else:
            # Generate sample data if no file provided
            self.user_data = self._generate_sample_data()
        
        return self.user_data
    
    def _generate_sample_data(self, days: int = 30) -> pd.DataFrame:
        """
        Generate sample user activity data for demonstration.
        
        Args:
            days: Number of days of data to generate
            
        Returns:
            DataFrame with sample activity data
        """
        np.random.seed(42)
        activities = [
            'Morning Exercise', 'Breakfast', 'Work Meeting', 'Lunch',
            'Coding Session', 'Reading', 'Dinner', 'Relaxation',
            'Social Media', 'Shopping', 'Gaming', 'Study',
            'Commute', 'Coffee Break', 'Gym', 'Sleep'
        ]
        
        categories = {
            'Morning Exercise': 'Health',
            'Breakfast': 'Meal',
            'Work Meeting': 'Work',
            'Lunch': 'Meal',
            'Coding Session': 'Work',
            'Reading': 'Learning',
            'Dinner': 'Meal',
            'Relaxation': 'Leisure',
            'Social Media': 'Leisure',
            'Shopping': 'Errands',
            'Gaming': 'Leisure',
            'Study': 'Learning',
            'Commute': 'Transport',
            'Coffee Break': 'Break',
            'Gym': 'Health',
            'Sleep': 'Rest'
        }
        
        data = []
        start_date = datetime.now() - timedelta(days=days)
        
        # Time distribution throughout the day - probabilities normalized to sum to 1.0
        hour_probs = [0.1, 0.1, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02, 0.01, 0.01]
        hour_probs = np.array(hour_probs) / sum(hour_probs)  # Normalize to ensure sum equals 1.0
        hour_values = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Generate 5-10 activities per day
            num_activities = np.random.randint(5, 11)
            
            for i in range(num_activities):
                # Time distribution throughout the day
                hour = np.random.choice(hour_values, p=hour_probs)
                minute = np.random.randint(0, 60)
                timestamp = current_date.replace(hour=hour, minute=minute, second=0)
                
                activity = np.random.choice(activities)
                category = categories[activity]
                
                # Duration in minutes
                duration = np.random.randint(15, 120)
                
                # Energy level (1-10)
                energy = np.random.randint(3, 10)
                
                # Location
                locations = ['Home', 'Office', 'Gym', 'Cafe', 'Outdoors']
                location = np.random.choice(locations, p=[0.4, 0.3, 0.1, 0.1, 0.1])
                
                data.append({
                    'timestamp': timestamp,
                    'activity': activity,
                    'category': category,
                    'duration': duration,
                    'energy_level': energy,
                    'location': location,
                    'day_of_week': timestamp.strftime('%A'),
                    'hour': hour
                })
        
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    def preprocess_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Preprocess user data for ML models.
        
        Args:
            df: Input DataFrame (uses self.user_data if None)
            
        Returns:
            Preprocessed DataFrame
        """
        if df is None:
            df = self.user_data.copy()
        
        # Extract temporal features
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week_num'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.day_name()  # Add day name
        df['is_weekend'] = df['day_of_week_num'].isin([5, 6]).astype(int)
        
        # Create activity frequency features
        activity_counts = df.groupby('activity').size().to_dict()
        df['activity_frequency'] = df['activity'].map(activity_counts)
        
        # Create time-based features
        df['time_of_day'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening']
        )
        
        self.processed_data = df
        return df
    
    def extract_features(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract features for clustering and prediction.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with extracted features
        """
        if df is None:
            df = self.processed_data.copy()
        
        # Create feature matrix
        features = pd.DataFrame()
        
        # Temporal features
        features['hour'] = df['hour']
        features['day_of_week'] = df['day_of_week_num']
        features['is_weekend'] = df['is_weekend']
        
        # Activity features (one-hot encoded)
        activity_dummies = pd.get_dummies(df['category'], prefix='category')
        features = pd.concat([features, activity_dummies], axis=1)
        
        # Duration and energy
        features['duration'] = df['duration']
        features['energy_level'] = df['energy_level']
        
        # Location features
        location_dummies = pd.get_dummies(df['location'], prefix='location')
        features = pd.concat([features, location_dummies], axis=1)
        
        return features
    
    def get_user_preferences(self, df: pd.DataFrame = None) -> Dict:
        """
        Extract user preferences from activity data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with user preferences
        """
        if df is None:
            df = self.processed_data.copy()
        
        preferences = {
            'favorite_activities': df['activity'].value_counts().head(5).to_dict(),
            'favorite_categories': df['category'].value_counts().to_dict(),
            'preferred_times': df.groupby('time_of_day').size().to_dict(),
            'average_duration': df['duration'].mean(),
            'peak_energy_hours': df.groupby('hour')['energy_level'].mean().sort_values(ascending=False).head(3).index.tolist(),
            'most_common_location': df['location'].mode()[0] if len(df['location'].mode()) > 0 else 'Home'
        }
        
        return preferences
    
    def save_data(self, filepath: str):
        """Save processed data to CSV."""
        if self.processed_data is not None:
            self.processed_data.to_csv(filepath, index=False)
        else:
            raise ValueError("No processed data available. Run preprocess_data() first.")

