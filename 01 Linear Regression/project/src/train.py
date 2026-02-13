import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_predict

import config
from preprocessing import get_processor


def remove_outliers(df):
    outliers = df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index
    return df.drop(outliers)


def train_model(X, y, model_name, model, param_grid):
    """
    Train a single model with cross-validation and hyperparameter tuning.

    Args:
        X: Features
        y: Target (log-transformed)
        model_name: Name of the model (for display)
        model: Scikit-learn model instance
        param_grid: Dictionary of hyperparameters to search

    Returns:
        tuple: (best_estimator, best_params, best_rmse)
    """
    # Create pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', get_processor()),
        ('model', model)
    ])

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1  # Use all CPU cores
    )

    grid_search.fit(X, y)

    # Calculate RMSE from MSE
    best_rmse = np.sqrt(-grid_search.best_score_)

    return grid_search.best_estimator_, grid_search.best_params_, best_rmse


def evaluate_ensemble(X, y, models_dict):
    """
    Evaluate the ensemble by averaging predictions from all models.
    Uses cross-validation to get unbiased estimate.

    Args:
        X: Features
        y: Target (log-transformed)
        models_dict: Dictionary of trained models

    Returns:
        float: Ensemble CV RMSE
    """
    # Collect cross-validated predictions from each model
    cv_predictions = []

    for model_name, model_pipeline in models_dict.items():
        # Get out-of-fold predictions using same CV splits
        cv_pred = cross_val_predict(
            model_pipeline, X, y, cv=5,
            method='predict', n_jobs=-1
        )
        cv_predictions.append(cv_pred)

    # Average predictions across all models
    ensemble_pred = np.mean(cv_predictions, axis=0)

    # Calculate RMSE
    ensemble_rmse = np.sqrt(np.mean((y - ensemble_pred) ** 2))

    return ensemble_rmse


def run_training():
    """
    Compare multiple linear regression models and return both individual models
    and ensemble performance.
    """
    # Load data
    df = pd.read_csv(config.TRAIN_PATH)

    # Remove outliers
    df = remove_outliers(df)

    X = df.drop([config.TARGET, 'Id'], axis=1)
    y = np.log1p(df[config.TARGET])

    # Define models and their hyperparameter grids
    models_config = {
        'Linear Regression': {
            'model': LinearRegression(),
            'param_grid': {}  # No hyperparameters to tune
        },
        'Ridge': {
            'model': Ridge(),
            'param_grid': {
                'model__alpha': [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
            }
        },
        'Lasso': {
            'model': Lasso(max_iter=10000),
            'param_grid': {
                'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
            }
        },
        'ElasticNet': {
            'model': ElasticNet(max_iter=10000),
            'param_grid': {
                'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'model__l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
            }
        }
    }


    # Store results
    results = {}
    trained_models = {}

    # Train each model
    for model_name, config_dict in models_config.items():
        best_model, best_params, best_rmse = train_model(
            X, y,
            model_name,
            config_dict['model'],
            config_dict['param_grid']
        )

        results[model_name] = {
            'model': best_model,
            'params': best_params,
            'rmse': best_rmse
        }
        trained_models[model_name] = best_model

    # Evaluate ensemble
    ensemble_rmse = evaluate_ensemble(X, y, trained_models)

    for model_name, result in results.items():
        params_str = str(result['params']) if result['params'] else 'None'
        print(f"{model_name:<20} {result['rmse']:<15.5f} {params_str}")

    print(f"{'-' * 60}")
    print(f"{'Ensemble (Average)':<20} {ensemble_rmse:<15.5f} {'All models averaged'}")

    return trained_models, results, ensemble_rmse


def make_submission_ensemble(models_dict):
    """
    Generate predictions using ensemble of all models (simple average).

    Args:
        models_dict: Dictionary of trained model pipelines
    """
    test_df = pd.read_csv(config.TEST_PATH)
    test_X = test_df.drop(['Id'], axis=1)

    # Collect predictions from all models
    all_predictions = []

    for model_name, model_pipeline in models_dict.items():
        log_predictions = model_pipeline.predict(test_X)
        predictions = np.expm1(log_predictions)
        all_predictions.append(predictions)

    # Average all predictions
    ensemble_predictions = np.mean(all_predictions, axis=0)

    # Format according to Kaggle requirement
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'SalePrice': ensemble_predictions
    })

    submission.to_csv(config.SUBMISSION_PATH, index=False)


def make_submission_single(model_pipeline):
    """
    Generate predictions using a single model.

    Args:
        model_pipeline: Trained pipeline
    """
    test_df = pd.read_csv(config.TEST_PATH)
    test_X = test_df.drop(['Id'], axis=1)

    # Predict and transform back from log-scale to original price
    log_predictions = model_pipeline.predict(test_X)
    predictions = np.expm1(log_predictions)

    # Format according to Kaggle requirement
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'SalePrice': predictions
    })

    submission.to_csv(config.SUBMISSION_PATH, index=False)


if __name__ == "__main__":
    trained_models, all_results, ensemble_rmse = run_training()
    make_submission_ensemble(trained_models)

    # Or if you prefer single best model, uncomment below:
    # best_model_name = min(all_results.items(), key=lambda x: x[1]['rmse'])[0]
    # make_submission_single(all_results[best_model_name]['model'])