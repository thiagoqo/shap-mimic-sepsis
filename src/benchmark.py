import argparse
import pandas as pd
import numpy as np
from src.data_processor import TimeSeriesDataProcessor
from src.linear_model import LinearTimeSeriesModel
from src.lstm_model import LSTMModel
from src.transformer_model import TimeSeriesTransformer
from src.xgboost_model import XGBoostModel
from src.random_forest_model import RandomForestModel
from src.lightgbm_model import LightGBMModel
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, brier_score_loss
from sklearn.calibration import calibration_curve, CalibrationDisplay
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import torch


# Set random seeds at the top of your file
def set_random_seeds(seed=42):
    """Set random seeds for reproducibility across all libraries used"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    # Make TensorFlow deterministic
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set NumPy print options for consistent output
    np.set_printoptions(precision=3, suppress=True)
    
    print(f"Random seeds set to {seed} for reproducibility")



def load_data(data_path: str) -> pd.DataFrame:
    """Load the patient timeseries data"""
    print("Loading patient timeseries data...")
    return pd.read_csv(data_path)

def get_feature_columns(df: pd.DataFrame, target_col: str) -> list:
    """Get feature columns by excluding specific columns"""
    exclude_columns = [
        'morta_hosp',  # future information, exclude to avoid data leakage
        'morta_90',    # future information, exclude to avoid data leakage
        'timestep',   # temporal index
        'stay_id',    # identifier
        target_col,   # target variable
        'los',      # future information, exclude to avoid data leakage

        # FIX This code comment below could be exclude important 
        # columns in a mortality prediction task, for example

        #'mechvent' if target_col != 'mechvent' else None,
        #'septic_shock' if target_col != 'septic_shock' else None,
        #'vasopressor' if target_col != 'vasopressor' else None,

        #'vaso_median' if target_col != 'vasopressor' else None,
        #'vaso_max' if target_col != 'vasopressor' else None,

        #Only auxiliary columns that would give a "spoiler" of the target 
        # should be excluded.
        #For example: if I want to predict whether it uses a 
        # vasopressor (binary), I can't see the dose (vaso_median)
        'vaso_median' if target_col == 'vasopressor' else None,
        'vaso_max' if target_col == 'vasopressor' else None,
        
        # --- DEFINITIONAL ABLATION (Data Leakage Prevention) ---
        # Reviewer Critique: Septic Shock is defined by vasopressors, fluids, and lactate/MAP.
        # Predicting septic shock using these variables creates circular prediction.
        'vaso_median' if target_col == 'septic_shock' else None,
        'vaso_max' if target_col == 'septic_shock' else None,
        'fluid_step' if target_col == 'septic_shock' else None,
        'fluid_total' if target_col == 'septic_shock' else None,
        'lactate_min' if target_col == 'septic_shock' else None,
        'lactate_max' if target_col == 'septic_shock' else None,
        'lactate_median' if target_col == 'septic_shock' else None,
        'map_min' if target_col == 'septic_shock' else None,
        'map_max' if target_col == 'septic_shock' else None,
        'map_median' if target_col == 'septic_shock' else None,
    ]
    return [col for col in df.columns if col not in exclude_columns and col is not None]

def split_data(df: pd.DataFrame, train_ratio: float = 0.8, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and validation sets"""
    patient_ids = df['stay_id'].unique()
    train_size = int(len(patient_ids) * train_ratio)
    
    # Use np.random.RandomState with fixed seed for shuffling
    rs = np.random.RandomState(random_state)
    shuffled_ids = patient_ids.copy()
    rs.shuffle(shuffled_ids)
    
    train_ids = shuffled_ids[:train_size]
    val_ids = shuffled_ids[train_size:]
    
    train_df = df[df['stay_id'].isin(train_ids)]
    val_df = df[df['stay_id'].isin(val_ids)]
    
    print(f"\nTrain set: {len(train_ids)} patients")
    print(f"Val set: {len(val_ids)} patients")
    
    return train_df, val_df

def evaluate_model(targets: np.ndarray, predictions: np.ndarray, task_type: str, n_bootstrap: int = 1000) -> Dict[str, Dict[str, float]]:
    """Calculate performance metrics with 95% confidence intervals using bootstrapping"""
    def calc_metrics(t, p):
        if task_type == 'classification':
            bp = (p >= 0.5).astype(int)
            # Avoid ValueError if a bootstrap sample only has one class
            try:
                auroc = roc_auc_score(t, p)
            except ValueError:
                auroc = np.nan
                
            # Expected Calibration Error (ECE)
            fraction_of_positives, mean_predicted_value = calibration_curve(t, p, n_bins=10)
            ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
            return {
                'auroc': auroc,
                'auprc': average_precision_score(t, p),
                'brier': brier_score_loss(t, p),
                'ece': ece,
                'accuracy': np.mean(bp == t),
                'precision': precision_score(t, bp, zero_division=0),
                'recall': recall_score(t, bp, zero_division=0),
                'f1': f1_score(t, bp, zero_division=0)
            }
        else:
            mse = np.mean((t - p) ** 2)
            return {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': np.mean(np.abs(t - p))
            }

    # Calculate actual metrics on full data
    actual_metrics = calc_metrics(targets, predictions)
    
    # Bootstrapping
    bootstrapped_metrics = {k: [] for k in actual_metrics.keys()}
    n_samples = len(targets)
    
    # Use fixed seed for reproducibility in bootstrap if possible
    rs = np.random.RandomState(42)
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = rs.randint(0, n_samples, n_samples)
        sample_targets = targets[indices]
        sample_predictions = predictions[indices]
        
        # Calculate metrics for sample
        sample_metrics = calc_metrics(sample_targets, sample_predictions)
        for k, v in sample_metrics.items():
            if not np.isnan(v):
                bootstrapped_metrics[k].append(v)
    
    # Calculate confidence intervals
    result = {}
    for k, v in actual_metrics.items():
        if len(bootstrapped_metrics[k]) > 0:
            lower = np.percentile(bootstrapped_metrics[k], 2.5)
            upper = np.percentile(bootstrapped_metrics[k], 97.5)
        else:
            lower, upper = np.nan, np.nan
            
        result[k] = {
            'mean': v,
            'lower_ci': lower,
            'upper_ci': upper
        }
    return result

def get_baseline_metrics(train_targets: np.ndarray, val_targets: np.ndarray, task_type: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Calculate baseline performance based on task type"""
    if task_type == 'classification':
        majority_pred = train_targets.mean() > 0.5
        train_baseline = np.ones_like(train_targets) * majority_pred
        val_baseline = np.ones_like(val_targets) * majority_pred
    else:  # regression
        mean_pred = np.mean(train_targets)
        train_baseline = np.ones_like(train_targets) * mean_pred
        val_baseline = np.ones_like(val_targets) * mean_pred
    
    return {
        'train': evaluate_model(train_targets, train_baseline, task_type),
        'val': evaluate_model(val_targets, val_baseline, task_type)
    }

def print_results(model_metrics: Dict[str, Dict[str, Dict[str, float]]], baseline_metrics: Dict[str, Dict[str, Dict[str, float]]], task_type: str):
    """Print model and baseline performance metrics based on task type"""
    def format_metric(metrics_dict, split, metric_name):
        m = metrics_dict[split][metric_name]
        return f"{m['mean']:.3f} (95% CI: {m['lower_ci']:.3f} - {m['upper_ci']:.3f})"

    print("\nResults:")
    print("Model Performance:")
    if task_type == 'classification':
        print(f"Train Brier: {format_metric(model_metrics, 'train', 'brier')}")
        print(f"Train ECE: {format_metric(model_metrics, 'train', 'ece')}")
        print(f"Train Accuracy: {format_metric(model_metrics, 'train', 'accuracy')}")
        print(f"Train AUROC: {format_metric(model_metrics, 'train', 'auroc')}")
        print(f"Train AUPRC: {format_metric(model_metrics, 'train', 'auprc')}")
        print(f"Val Brier: {format_metric(model_metrics, 'val', 'brier')}")
        print(f"Val ECE: {format_metric(model_metrics, 'val', 'ece')}")
        print(f"Val Accuracy: {format_metric(model_metrics, 'val', 'accuracy')}")
        print(f"Val AUROC: {format_metric(model_metrics, 'val', 'auroc')}")
        print(f"Val AUPRC: {format_metric(model_metrics, 'val', 'auprc')}")
        
        print("\nBaseline (Majority Class) Performance:")
        print(f"Train Brier: {format_metric(baseline_metrics, 'train', 'brier')}")
        print(f"Train ECE: {format_metric(baseline_metrics, 'train', 'ece')}")
        print(f"Train Accuracy: {format_metric(baseline_metrics, 'train', 'accuracy')}")
        print(f"Train AUROC: {format_metric(baseline_metrics, 'train', 'auroc')}")
        print(f"Train AUPRC: {format_metric(baseline_metrics, 'train', 'auprc')}")
        print(f"Val Brier: {format_metric(baseline_metrics, 'val', 'brier')}")
        print(f"Val ECE: {format_metric(baseline_metrics, 'val', 'ece')}")
        print(f"Val Accuracy: {format_metric(baseline_metrics, 'val', 'accuracy')}")
        print(f"Val AUROC: {format_metric(baseline_metrics, 'val', 'auroc')}")
        print(f"Val AUPRC: {format_metric(baseline_metrics, 'val', 'auprc')}")
    else:  # regression
        print(f"Train RMSE: {format_metric(model_metrics, 'train', 'rmse')}")
        print(f"Train MAE: {format_metric(model_metrics, 'train', 'mae')}")
        print(f"Val RMSE: {format_metric(model_metrics, 'val', 'rmse')}")
        print(f"Val MAE: {format_metric(model_metrics, 'val', 'mae')}")
        
        print("\nBaseline (Mean Prediction) Performance:")
        print(f"Train RMSE: {format_metric(baseline_metrics, 'train', 'rmse')}")
        print(f"Train MAE: {format_metric(baseline_metrics, 'train', 'mae')}")
        print(f"Val RMSE: {format_metric(baseline_metrics, 'val', 'rmse')}")
        print(f"Val MAE: {format_metric(baseline_metrics, 'val', 'mae')}")

def run_benchmark(task: str, model_type: str, include_treatments: bool = True, 
                 prediction_horizon: int = None, random_state: int = 42,
                 regularization: str = 'ridge', alpha: float = 1.0,
                 use_gpu: bool = False):
    # Determine task type based on target column
    task_type = 'classification' if task in ['mechvent', 'morta_hosp', 'septic_shock', 'sepsis', 'vasopressor'] else 'regression'
    
    # Load and prepare data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "processed_files", "patient_timeseries_v4.csv")
    df = load_data(data_path)
    features = get_feature_columns(df, task)
    
    # Filter out treatment variables if specified
    if not include_treatments:
        treatment_vars = ['mechvent', 'vaso_median', 'vaso_max', 'abx_given',
       'hours_since_first_abx', 'num_abx', 'fluid_total', 'fluid_step', 'peep', 'tidal_volume', 'minute_volume', 'peak_inspiratory_pressure', 'mean_airway_pressure']
        features = [f for f in features if f not in treatment_vars]
    
    # Initialize processor
    print("\nInitializing data processor...")
    processor = TimeSeriesDataProcessor(
        features=features,
        task=task,
        window_size=6,
        prediction_horizon=prediction_horizon if task in ['septic_shock', 'mechvent', 'sepsis', 'vasopressor'] else None
    )

    
    # Split data
    train_df, val_df = split_data(df)
    
    # Process and normalize data
    print("\nProcessing data...")
    train_features, train_targets, train_indices = processor.prepare_data(train_df)
    val_features, val_targets, val_indices = processor.prepare_data(val_df)
    train_features_norm, val_features_norm = processor.normalize_features(train_features, val_features)
    
    # Configure batch size based on model type
    batch_size = 32 if model_type in ['lstm', 'transformer'] else None
    
    # Train model
    print(f"\nTraining {model_type} model (GPU={use_gpu})...")
    if model_type == 'linear':
        # Print regularization information for regression tasks
        if task_type == 'regression':
            if regularization == 'ridge':
                print(f"Using Ridge regression with alpha={alpha} (L2 regularization)")
            elif regularization == 'lasso':
                print(f"Using Lasso regression with alpha={alpha} (L1 regularization)")
            elif regularization == 'elasticnet':
                print(f"Using ElasticNet regression with alpha={alpha}, l1_ratio=0.5 (combined L1/L2 regularization)")
            else:
                print("Using standard Linear Regression (no regularization)")
        else:
            print("Using Logistic Regression for classification task")
            
        model = LinearTimeSeriesModel(
            task_type=task_type, 
            random_state=random_state,
            regularization=regularization if task_type == 'regression' else None,
            alpha=alpha
        )
    elif model_type == 'lstm':
        input_dim = train_features_norm.shape[2]
        model = LSTMModel(task_type=task_type, input_dim=input_dim)
    elif model_type == 'transformer':
        model = TimeSeriesTransformer(task_type=task_type)
    elif model_type == 'xgboost':
        model = XGBoostModel(task_type=task_type, random_state=random_state, use_gpu=use_gpu)
    elif model_type == 'random_forest':
        model = RandomForestModel(task_type=task_type, random_state=random_state)
    elif model_type == 'lightgbm':
        model = LightGBMModel(task_type=task_type, random_state=random_state, use_gpu=use_gpu)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    if batch_size:
        # For LSTM and Transformer models, use batched training
        model.fit(train_features_norm, train_targets, batch_size=batch_size)
        train_preds = model.predict(train_features_norm, batch_size=batch_size)
        val_preds = model.predict(val_features_norm, batch_size=batch_size)
    else:
        # For linear model, use regular training
        model.fit(train_features_norm, train_targets)
        train_preds = model.predict(train_features_norm)
        val_preds = model.predict(val_features_norm)
    
    # Confusion Matrix (for Classification)
    if task_type == 'classification':
        try:
            print("Generating Confusion Matrix...")
            val_preds_binary = (val_preds >= 0.5).astype(int)
            cm = confusion_matrix(val_targets, val_preds_binary)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
            disp.plot(cmap='Blues', values_format='d', ax=ax)
            ax.set_title(f'Confusion Matrix: {task} ({model_type})')
            
            # Save
            cm_dir = os.path.join(project_root, "results", "confusion_matrices")
            os.makedirs(cm_dir, exist_ok=True)
            cm_path_png = os.path.join(cm_dir, f"cm_{task}_{model_type}.png")
            cm_path_eps = os.path.join(cm_dir, f"cm_{task}_{model_type}.eps")
            plt.savefig(cm_path_png)
            plt.savefig(cm_path_eps, format='eps')
            plt.close() # Close plot to free memory
            print(f"Saved Confusion Matrix to {cm_path_png} and {cm_path_eps}")
            
            # (Calibration Curves are now generated in a unified plot by compare_models.py)

            # Save logits/predictions for DeLong Test
            print("Exporting predictions for statistical significance testing...")
            preds_dir = os.path.join(project_root, "results", "predictions")
            os.makedirs(preds_dir, exist_ok=True)
            
            # Extract SOFA/NEWS aligned with val_df indices for comparison
            # Notice we kept val_df uncorrupted before extraction
            aligned_val_df = val_df.loc[val_indices]
            export_df = pd.DataFrame({
                'stay_id': aligned_val_df['stay_id'].values,
                'target': val_targets,
                'pred_prob': val_preds,
            })
            
            # Append scores if they exist in dataset
            if 'sofa_score' in aligned_val_df.columns: export_df['sofa_score'] = aligned_val_df['sofa_score'].values
            if 'news_score' in aligned_val_df.columns: export_df['news_score'] = aligned_val_df['news_score'].values
            
            preds_path = os.path.join(preds_dir, f"preds_{task}_{model_type}_{trt_suffix}.csv")
            export_df.to_csv(preds_path, index=False)
            print(f"Saved predictions to {preds_path}")
            
        except Exception as e:
            print(f"Error plotting or exporting: {e}")

    # Calculate metrics
    model_metrics = {
        'train': evaluate_model(train_targets, train_preds, task_type),
        'val': evaluate_model(val_targets, val_preds, task_type)
    }
    
    baseline_metrics = get_baseline_metrics(train_targets, val_targets, task_type)
    
    # Print results
    print_results(model_metrics, baseline_metrics, task_type)
    
    # Return metrics for saving to CSV
    result = {
        'task': task,
        'model_type': model_type,
        'include_treatments': include_treatments,
        'prediction_horizon': prediction_horizon,
        'regularization': regularization,
        'alpha': alpha
    }
    
    # Add model metrics
    for split in ['train', 'val']:
        for metric, values in model_metrics[split].items():
            result[f'{split}_{metric}_mean'] = values['mean']
            result[f'{split}_{metric}_lower_ci'] = values['lower_ci']
            result[f'{split}_{metric}_upper_ci'] = values['upper_ci']
    
    # Add baseline metrics
    for split in ['train', 'val']:
        for metric, values in baseline_metrics[split].items():
            result[f'{split}_baseline_{metric}_mean'] = values['mean']
            result[f'{split}_baseline_{metric}_lower_ci'] = values['lower_ci']
            result[f'{split}_baseline_{metric}_upper_ci'] = values['upper_ci']
            
    return result

def run_all_experiments(use_gpu: bool = False):
    """Run experiments with different configurations and save results to CSV"""
    # Define tasks and their types
    tasks = {
        'morta_hosp': 'static',  # Static outcome
        'los': 'static',         # Static outcome
        'septic_shock': 'temporal',  # Time-varying outcome
        'mechvent': 'temporal',       # Time-varying outcome
        'vasopressor': 'temporal'       # Time-varying outcome
    }
    
    model_types = ['linear', 'lstm', 'transformer', 'xgboost', 'random_forest', 'lightgbm']
    treatment_options = [True, False]
    
    # Set fixed prediction horizon for temporal tasks
    fixed_prediction_horizon = 6  # Hours ahead to predict
    
    results = []
    
    # Calculate total experiments
    total_experiments = 0
    for task, task_type in tasks.items():
        if task_type == 'static':
            total_experiments += len(model_types) * len(treatment_options)
        else:  # temporal
            total_experiments += len(model_types) * len(treatment_options)
    
    experiment_count = 0
    
    # Run experiments for all tasks
    for task, task_type in tasks.items():
        for model_type in model_types:
            for include_treatments in treatment_options:
                if task_type == 'static':
                    # For static tasks, run once with no prediction horizon
                    experiment_count += 1
                    print(f"\n\n{'='*80}")
                    print(f"Experiment {experiment_count}/{total_experiments}")
                    print(f"Task: {task}, Model: {model_type}, Include Treatments: {include_treatments}")
                    print(f"{'='*80}\n")
                    
                    result = run_benchmark(
                        task=task,
                        model_type=model_type,
                        include_treatments=include_treatments,
                        prediction_horizon=None,
                        use_gpu=use_gpu
                    )
                    results.append(result)
                    
                    # Save intermediate results after each experiment
                    results_df = pd.DataFrame(results)
                    for trt_val in [True, False]:
                        df_trt = results_df[results_df['include_treatments'] == trt_val]
                        if not df_trt.empty:
                            trt_suffix = "with_treatments" if trt_val else "no_treatments"
                            df_trt.to_csv(f"benchmark_results_{trt_suffix}.csv", index=False)
                    print(f"Results saved to benchmark_results_with_treatments.csv and benchmark_results_no_treatments.csv")
                else:
                    # For temporal tasks, use fixed prediction horizon
                    experiment_count += 1
                    print(f"\n\n{'='*80}")
                    print(f"Experiment {experiment_count}/{total_experiments}")
                    print(f"Task: {task}, Model: {model_type}, Include Treatments: {include_treatments}")
                    print(f"Prediction Horizon: {fixed_prediction_horizon} hours")
                    print(f"{'='*80}\n")
                    
                    result = run_benchmark(
                        task=task,
                        model_type=model_type,
                        include_treatments=include_treatments,
                        prediction_horizon=fixed_prediction_horizon,
                        use_gpu=use_gpu
                    )
                    results.append(result)
                    
                    # Save intermediate results after each experiment
                    results_df = pd.DataFrame(results)
                    for trt_val in [True, False]:
                        df_trt = results_df[results_df['include_treatments'] == trt_val]
                        if not df_trt.empty:
                            trt_suffix = "with_treatments" if trt_val else "no_treatments"
                            df_trt.to_csv(f"benchmark_results_{trt_suffix}.csv", index=False)
                    print(f"Results saved to benchmark_results_with_treatments.csv and benchmark_results_no_treatments.csv")
    
    return results

def run_selected_experiments(task: str, include_treatments: bool = False, use_gpu: bool = False):
    """Run experiments with all models for a specific task and treatment setting"""
    model_types = ['linear', 'lstm', 'transformer', 'xgboost', 'random_forest', 'lightgbm']
    
    # Determine if this is a temporal task
    temporal_tasks = ['septic_shock', 'mechvent', 'sepsis', 'vasopressor']
    is_temporal = task in temporal_tasks
    
    # Define prediction horizons for temporal tasks
    prediction_horizons = [1, 2, 3, 4, 5, 6] if is_temporal else [None]
    
    results = []
    
    total_experiments = len(model_types) * len(prediction_horizons)
    experiment_count = 0
    
    for model_type in model_types:
        for horizon in prediction_horizons:
            experiment_count += 1
            print(f"\n\n{'='*80}")
            print(f"Experiment {experiment_count}/{total_experiments}")
            print(f"Task: {task}, Model: {model_type}, Include Treatments: {include_treatments}")
            if is_temporal:
                print(f"Prediction Horizon: {horizon} hours")
            print(f"{'='*80}\n")
            
            result = run_benchmark(
                task=task,
                model_type=model_type,
                include_treatments=include_treatments,
                prediction_horizon=horizon,
                use_gpu=use_gpu
            )
            results.append(result)
            
            # Save intermediate results after each experiment
            trt_suffix = "with_treatments" if include_treatments else "no_treatments"
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"{task}_benchmark_results_{trt_suffix}.csv", index=False)
            print(f"Results saved to {task}_benchmark_results_{trt_suffix}.csv")
    
    return results

if __name__ == "__main__":
    # Set random seeds at the beginning
    set_random_seeds()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--run_all", action="store_true", help="Run all experiments")
    parser.add_argument("--run_selected", action="store_true", help="Run all models for a specific task")
    parser.add_argument("--task", type=str, default="mechvent", help="Target column name")
    parser.add_argument("--model_type", type=str, default="lstm", help="Model type")
    parser.add_argument("--include_treatments", type=bool, default=False, help="Whether to include treatment variables")
    parser.add_argument("--prediction_horizon", type=int, default=6, help="Prediction horizon for temporal tasks (hours)")
    parser.add_argument("--regularization", type=str, default="ridge", choices=["ridge", "lasso", "elasticnet", "none"], 
                        help="Regularization type for linear models")
    parser.add_argument("--alpha", type=float, default=1.0, help="Regularization strength")
    parser.add_argument("--use_gpu", action="store_true", help="Whether to use GPU for XGBoost/LightGBM")
    
    args = parser.parse_args()

    # Call this function at the beginning of your main function or script
    set_random_seeds(args.random_state)
        
    if args.run_all:
        run_all_experiments(use_gpu=args.use_gpu)
    elif args.run_selected:
        run_selected_experiments(task=args.task, include_treatments=args.include_treatments, use_gpu=args.use_gpu)
    else:
        # For single runs, use the specified prediction horizon for temporal tasks
        temporal_tasks = ['septic_shock', 'mechvent', 'sepsis', 'vasopressor']
        
        result = run_benchmark(
            args.task, 
            args.model_type,
            args.include_treatments,
            prediction_horizon=args.prediction_horizon if args.task in temporal_tasks else None,
            random_state=args.random_state,
            regularization=args.regularization,
            alpha=args.alpha,
            use_gpu=args.use_gpu
        )
        # Save single result to CSV
        trt_suffix = "with_treatments" if args.include_treatments else "no_treatments"
        pd.DataFrame([result]).to_csv(f"single_benchmark_result_{trt_suffix}.csv", index=False)
        print(f"Result saved to single_benchmark_result_{trt_suffix}.csv")
