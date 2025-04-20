import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class StructuredDataset:
    def __init__(self, data_path: str, target_column: str, test_size: float = 0.2):
        """
        Initialize the structured dataset handler
        
        Args:
            data_path (str): Path to the dataset file
            target_column (str): Name of the target column
            test_size (float): Proportion of data to use for testing
        """
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.scaler = StandardScaler()
        
        # Load and prepare data
        self._load_data()
        
    def _load_data(self):
        """Load and prepare the dataset"""
        # Load data
        self.df = pd.read_csv(self.data_path)
        
        # Separate features and target
        self.feature_columns = [col for col in self.df.columns if col != self.target_column]
        self.X = self.df[self.feature_columns].values
        self.y = self.df[self.target_column].values
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=42
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
    def get_random_sample(self, split: str = 'test') -> Tuple[List[float], int]:
        """
        Get a random sample from the dataset
        
        Args:
            split (str): Which split to sample from ('train' or 'test')
            
        Returns:
            Tuple[List[float], int]: Features and target label
        """
        if split == 'train':
            idx = np.random.randint(0, len(self.X_train))
            return self.X_train[idx].tolist(), self.y_train[idx]
        else:
            idx = np.random.randint(0, len(self.X_test))
            return self.X_test[idx].tolist(), self.y_test[idx]
            
    def get_feature_names(self) -> List[str]:
        """Get the names of the feature columns"""
        return self.feature_columns 