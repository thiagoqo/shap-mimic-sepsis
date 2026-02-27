import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Optional

class RandomForestModel:
    def __init__(self, 
                 task_type: str = 'classification', 
                 random_state: Optional[int] = None,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2):
        """
        Wrapper for Scikit-Learn RandomForest models to handle time-series data formatting.
        
        Args:
            task_type: 'classification' or 'regression'
            random_state: Random seed for reproducibility
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the tree
        """
        self.task_type = task_type
        self.random_state = random_state
        
        if task_type == 'classification':
             self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state,
                n_jobs=-1  # Use all available cores
            )
        else:  # regression
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state,
                n_jobs=-1
            )
    
    def flatten_features(self, data: np.ndarray) -> np.ndarray:
        """Reshape (N, T, X) data to (N, T*X)"""
        if len(data.shape) == 2:
            return data
            
        N, T, X = data.shape
        return data.reshape(N, T * X)
    
    def fit(self, train_data: np.ndarray, train_targets: np.ndarray) -> None:
        X = self.flatten_features(train_data)
        # Handle NaN values (RandomForest in sklearn does not support NaNs natively)
        # Simple imputation: replace with 0 or mean. Given standardization, 0 is mean.
        # Ideally, we should use SimpleImputer, but to keep it simple and consistent:
        # Assuming data is already normalized/standardized (0 mean), replacing NaN with 0 is reasonable.
        if np.isnan(X).any():
             # print("Warning: NaNs found in input data for RandomForest. Imputing with 0.")
             X = np.nan_to_num(X, nan=0.0)
             
        self.model.fit(X, train_targets)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        X = self.flatten_features(data)
        if np.isnan(X).any():
             X = np.nan_to_num(X, nan=0.0)

        if self.task_type == 'classification':
            # Return probabilities of the positive class
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """Percentage prediction for classification"""
        X = self.flatten_features(data)
        if np.isnan(X).any():
             X = np.nan_to_num(X, nan=0.0)
             
        if self.task_type == 'classification':
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("predict_proba is not available for regression tasks")
