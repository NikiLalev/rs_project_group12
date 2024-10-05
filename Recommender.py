import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

class RecommenderStrategy(ABC):
    """Abstract base class for recommender strategies."""

    def __init__(self, model):
        """
        Initialize the strategy with a model.

        Args:
            model: The recommendation model.
        """
        self.model = model

    @abstractmethod
    def recommend(self, user_id: int, n: int) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.

        Args:
            user_id (int): The ID of the user to recommend for.
            n (int): The number of recommendations to generate.

        Returns:
            List[Tuple[int, float]]: A list of tuples, each containing an item ID and its score.
        """
        pass

class UserUserRecommender(RecommenderStrategy):
    """User-User collaborative filtering recommender strategy."""

    def recommend(self, user_id: int, n: int) -> List[Tuple[int, float]]:
        return self.model.recommend(user_id, n)
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(df)

class ItemItemRecommender(RecommenderStrategy):
    """Item-Item collaborative filtering recommender strategy."""

    def recommend(self, user_id: int, n: int) -> List[Tuple[int, float]]:
        return self.model.recommend(user_id, n)
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(df)

class Recommender:
    """Singleton class for managing recommendation models and strategies."""

    _instance = None  # Class-level attribute for Singleton instance
    MODEL_PATHS = {
        "user-user": "models/simple_uu_slim.pkl",
        "item-item": "models/simple_ii_slim.pkl",
    }

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance is created (Singleton)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        """Initialize the Recommender instance."""
        if not hasattr(self, 'models'):
            self.models = {}
            self.current_strategy = None

    def create_strategy(self, model_type: str):
        """
        Create and return a strategy based on the model type.

        Args:
            model_type (str): The type of model to create a strategy for.

        Returns:
            RecommenderStrategy: An instance of the appropriate strategy.

        Raises:
            ValueError: If the strategy is not implemented for the given model type.
        """
        model = self.models[model_type]
        if model_type == "user-user":
            return UserUserRecommender(model)
        elif model_type == "item-item":
            return ItemItemRecommender(model)
        else:
            raise ValueError(f"Strategy not implemented for model type: {model_type}")

    def load_model(self, model_type: str):
        """
        Load a model of the specified type and set it as the current strategy.

        Args:
            model_type (str): The type of model to load.

        Raises:
            ValueError: If the model type is invalid.
        """
        logging.info(f"Loading model: {model_type}")
        if model_type not in self.models:
            path = self.MODEL_PATHS.get(model_type)
            if not path:
                logging.error(f"Invalid model type: {model_type}")
                raise ValueError(f"Invalid model type: {model_type}")
            with open(path, 'rb') as file:
                self.models[model_type] = pickle.load(file)
                logging.info(f"Model {model_type} loaded successfully.")
        self.current_strategy = self.create_strategy(model_type)

    def recommend(self, user_id: int, n: int) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.

        Args:
            user_id (int): The ID of the user to recommend for.
            n (int): The number of recommendations to generate.

        Returns:
            List[Tuple[int, float]]: A list of tuples, each containing an item ID and its score.

        Raises:
            ValueError: If no model has been loaded.
        """
        if self.current_strategy is None:
            raise ValueError("No model loaded. Call load_model first.")
        return self.current_strategy.recommend(user_id, n)
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the ratings for all user-item pairs in the input DataFrame.

        Args:
            df (pd.DataFrame): A DataFrame with 'user' and 'item' columns.
        
        Returns:
            pd.DataFrame: A DataFrame with 'user', 'item', and 'prediction' columns.

        Raises:
            ValueError: If no model has been loaded.
        """
        if self.current_strategy is None:
            raise ValueError("No model loaded. Call load_model first.")
        return self.current_strategy.predict(df)