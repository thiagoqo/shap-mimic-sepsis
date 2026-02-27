import numpy as np
import lightgbm as lgb
from typing import Optional

class LightGBMModel:
    def __init__(self, 
                 task_type: str = 'classification', 
                 random_state: Optional[int] = None,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 num_leaves: int = 31,
                 use_gpu: bool = False):
        """
        Wrapper for LightGBM models to handle time-series data formatting.
        
        Args:
            task_type: 'classification' or 'regression'
            random_state: Random seed for reproducibility
            n_estimators: Number of boosting iterations
            learning_rate: Learning rate
            use_gpu: Whether to use GPU acceleration
        """
        self.task_type = task_type
        self.random_state = random_state
        
        # Configure GPU params
        extra_params = {}
        if use_gpu:
            extra_params['device'] = 'gpu'
            # Note: LightGBM requires OpenCL for GPU support usually,
            # sometimes 'cuda' works depending on build. Sticking to 'gpu' which is standard.
        
        if task_type == 'classification':
             self.model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1, # Silence warnings
                **extra_params
            )
        else:  # regression
            self.model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
                **extra_params
            )
    
    def flatten_features(self, data: np.ndarray) -> np.ndarray:
        """Reshape (N, T, X) data to (N, T*X)"""
        if len(data.shape) == 2:
            return data
            
        N, T, X = data.shape
        return data.reshape(N, T * X)
    
    def fit(self, train_data: np.ndarray, train_targets: np.ndarray) -> None:
        X = self.flatten_features(train_data)
        self.model.fit(X, train_targets)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        X = self.flatten_features(data)
        if self.task_type == 'classification':
            # Return probabilities of the positive class
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """Percentage prediction for classification"""
        X = self.flatten_features(data)
        if self.task_type == 'classification':
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("predict_proba is not available for regression tasks")
