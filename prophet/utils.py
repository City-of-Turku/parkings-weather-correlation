from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric


def get_forecast_total_sum(forecast, start_date, end_date):
    filtered_df = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)]
    total_sum = filtered_df['yhat'].sum()
    return total_sum


def cross_validate(model, initial, period, horizon):
    return cross_validation(model=model, 
                            initial=initial, 
                            period=period, 
                            horizon=horizon)