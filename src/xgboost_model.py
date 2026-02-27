import numpy as np
import xgboost as xgb
from typing import Optional

class XGBoostModel:
    def __init__(self, 
                 task_type: str = 'classification', 
                 random_state: Optional[int] = None,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 use_gpu: bool = False):
        """
        Wrapper for XGBoost models to handle time-series data formatting.
        
        Args:
            task_type: 'classification' or 'regression'
            random_state: Random seed for reproducibility
            n_estimators: Number of gradient boosted trees
            max_depth: Maximum tree depth for base learners
            learning_rate: Boosting learning rate
            use_gpu: Whether to use GPU acceleration
        """
        self.task_type = task_type
        self.random_state = random_state
        
        # Configure GPU params
        extra_params = {}
        if use_gpu:
            extra_params['tree_method'] = 'gpu_hist'
            extra_params['device'] = 'cuda'
        
        if task_type == 'classification':
             self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                n_jobs=-1,  # Use all available cores
                eval_metric='logloss', # Avoid warning
                **extra_params
            )
        else:  # regression
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                n_jobs=-1,
                eval_metric='rmse',
                **extra_params
            )
    
    def flatten_features(self, data: np.ndarray) -> np.ndarray:
        """Reshape (N, T, X) data to (N, T*X)"""
        # If data is already 2D (e.g. static task with T=1 collapsed?), handle it
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
        """Percentage prediction for classification (compatible with LIME/Sklearn)"""
        X = self.flatten_features(data)
        if self.task_type == 'classification':
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("predict_proba is not available for regression tasks")
