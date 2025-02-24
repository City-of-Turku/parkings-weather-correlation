from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from itertools import product
import pandas as pd
import numpy as np


def get_forecast_total_sum(forecast, start_date, end_date):
    filtered_df = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)]
    total_sum = filtered_df['yhat'].sum()
    return total_sum


def cross_validate(model, initial, period, horizon):
    return cross_validation(model=model, 
                            initial=initial, 
                            period=period, 
                            horizon=horizon)


def optimize_prophet_params(df, target_metric='rmse', print_status=False, incl_rain=False, incl_temp=False):
    """
    Perform grid search to find optimal Prophet parameters.
    
    Parameters:
    df: DataFrame with 'ds' and 'y' columns
    target_metric: Metric to optimize ('rmse', 'mae', 'mape', 'mdape', 'smape')
    
    Returns:
    dict: Best parameters and their score
    DataFrame: All parameters and their scores
    """
    
    # Define parameter grid
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5, 1.0],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0, 100.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'changepoint_range': [0.8, 0.85, 0.9, 0.95],
        'interval_width': [0.95, 0.99]
    }
    
    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    results = []
    
    # Progress tracking
    total_combinations = len(all_params)
    print(f"Testing {total_combinations} combinations")
    
    for i, params in enumerate(all_params, 1):
        if print_status:
            print(f"Testing combination {i}/{total_combinations}")
        
        try:
            # Create and fit model
            model = Prophet(**params)
            model.add_country_holidays(country_name='FI')
            if incl_rain:
                model.add_regressor('rain')
            if incl_temp:
                model.add_regressor('temp')
            model.fit(df)
            
            # Cross validation
            df_cv = cross_validation(
                model,
                initial='120 days',
                period='7 days',
                horizon='14 days',
                parallel="processes"
            )
            
            # Calculate metrics
            df_metrics = performance_metrics(df_cv)
            
            # Store results
            params['score'] = df_metrics[target_metric].mean()
            results.append(params)
            
        except Exception as e:
            print(f"Error with parameters {params}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best parameters
    best_params = results_df.loc[results_df['score'].idxmin()].to_dict()
    
    # Sort results by score
    results_df = results_df.sort_values('score')
    
    return best_params, results_df

def print_best_params(best_params, results_df, n_best=5):
    """Print the best parameters and top N results"""
    print("\nBest Parameters:")
    for param, value in best_params.items():
        if param != 'score':
            print(f"{param}: {value}")
    print(f"Best score: {best_params['score']:.2f}")
    
    print("\nTop", n_best, "Parameter Combinations:")
    print(results_df.head(n_best).to_string())
