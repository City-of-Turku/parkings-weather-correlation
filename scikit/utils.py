import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from IPython.display import display


YEAR = "year"
MONTH = "month"
MONTH_SIN = "month_sin"
MONTH_COS = "month_cos"
DAY_OF_WEEK="dayofweek"
DAY_OF_WEEK_SIN = "day_of_week_sin"
DAY_OF_YEAR="day_of_year"
HOUR="hour"
HOUR_SIN = "hour_sin"
RAIN="rain"
RAIN_BOOL="rain boolean"
TEMPERATURE="temperature"
BASE = "base"
DRY_WEATHER ="dry weather"
MINUS_20 = "-20 degrees temperature"
MINUS_10 = "-10 degrees temperature"
ZERO = "0 degrees temperature"
PLUS_10 = "10 degrees temperature"
PLUS_20 = "20 degrees temperature"
ONEHOT_SCALER_COLUMNS = [HOUR, MONTH, DAY_OF_WEEK]
STANDARD_SCALER_COLUMNS = [RAIN, TEMPERATURE, YEAR]
SEASONABILITY_COLUMNS = [YEAR, MONTH, DAY_OF_WEEK, HOUR]
RAIN_VALUE = 3


def add_seasonability_columns(df: pd.DataFrame) -> pd.DataFrame:
    if YEAR in SEASONABILITY_COLUMNS:
        df[YEAR] = df.index.year
    if MONTH in SEASONABILITY_COLUMNS:
        df[MONTH] = df.index.month
    if MONTH_SIN in SEASONABILITY_COLUMNS:
        df[MONTH_SIN] = np.sin(2 * np.pi * df["month"] / 12)
    if MONTH_COS in SEASONABILITY_COLUMNS:
        df[MONTH_COS] = np.cos(2 * np.pi * df["month"] / 12)
    if DAY_OF_WEEK in SEASONABILITY_COLUMNS:
        df[DAY_OF_WEEK] = df.index.dayofweek
    if DAY_OF_WEEK_SIN in SEASONABILITY_COLUMNS:
        df[DAY_OF_WEEK_SIN] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    if DAY_OF_YEAR in SEASONABILITY_COLUMNS:
        df[DAY_OF_YEAR] = df.index.dayofyear
    if HOUR in SEASONABILITY_COLUMNS:
        df[HOUR] = df.index.hour
    if HOUR_SIN in SEASONABILITY_COLUMNS:
        df[HOUR_SIN] = np.sin(2 * np.pi * df["hour"] / 24)
    return df


def get_parkings_df(file_name:str) -> pd.DataFrame:
    df_parkings = pd.read_csv(file_name)
    df_parkings["time_start"] = pd.to_datetime(df_parkings["time_start"], format="ISO8601")
    df_parkings["time_end"] = pd.to_datetime(df_parkings["time_end"], format="ISO8601")
    return df_parkings


def get_parkings_in_zone(df: pd.DataFrame, zone: int) -> pd.DataFrame:
    return df[df["zone_id"] == zone]


def get_parkings_in_area(df: pd.DataFrame, area: str) -> pd.DataFrame:
    return df[df["parking_area_id"] == area]


def get_hourly_parkings(parkings: pd.DataFrame) -> pd.DataFrame:
    hourly_index = pd.date_range(start=parkings["time_start"].min().floor("h"),
                             end=parkings["time_end"].max().ceil("h"),
                             freq="h")

    # Initialize an empty DataFrame for hourly occupancy counts
    hourly_parking = pd.DataFrame({"ds": hourly_index})

    # Count active parkings in each hour
    def count_active_parkings(timestamp):
        return ((parkings["time_start"] <= timestamp) & (parkings["time_end"] > timestamp)).sum()
        # time stamp is between start and end time of parking

    hourly_parking["num_parkings"] = hourly_parking["ds"].apply(count_active_parkings)
    hourly_parking["ds"] = hourly_parking["ds"].dt.tz_localize(None)
    return hourly_parking


def get_weather_data(rain_threshold=RAIN_VALUE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_rain = pd.read_csv("../data/artukainen_rain.csv")
    df_rain["timestamp"] = pd.to_datetime(df_rain["timestamp"])
    df_rain_bool = df_rain.copy()   
    df_rain_bool["value"] = df_rain_bool["value"] >= rain_threshold
    df_rain_bool = df_rain_bool.rename(columns={"timestamp": "ds", "value": "rain"})
    df_rain = df_rain.rename(columns={"timestamp": "ds", "value": "rain"})
    df_temperature = pd.read_csv("../data/artukainen_temperature.csv")
    df_temperature["timestamp"] = pd.to_datetime(df_temperature["timestamp"])
    df_temperature = df_temperature.rename(columns={"timestamp": "ds", "value": "temperature"})
    return df_rain, df_rain_bool, df_temperature
       

def get_pipeline(
        regressor,
        standard_columns=STANDARD_SCALER_COLUMNS, 
        onehot_colums=ONEHOT_SCALER_COLUMNS, 
    ) -> Pipeline:
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), onehot_colums),
            ("Standard", StandardScaler(), standard_columns)
        
        ],
        force_int_remainder_cols=False
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    return pipeline


def mlp_parameter_tuning(X_train: pd.Series, y_train: pd.Series):
    param_grid = {
        "hidden_layer_sizes": [(50,), (50,25),(50, 25, 10), (50, 25,10, 5), (32, 16, 4),],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "lbfgs"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["constant", "adaptive"],
        "max_iter": [250, 500, 1000,2000]
    }
    y_train = np.array(y_train).ravel()
    mlp = MLPRegressor(random_state=42)
    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    # GridSearchCV with TimeSeriesSplit
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best parameters found: ", best_params)


def fit_model(
        pipeline: Pipeline, 
        df: pd.DataFrame, 
        feature_columns: list, 
        tune_mlp_parameters=False
    ) -> Tuple[pd.Series, pd.Series]:
    X = df[feature_columns]
    y = df[["num_parkings"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    if tune_mlp_parameters:
        mlp_parameter_tuning(X_train, y_train)
    if isinstance(pipeline["regressor"], MLPRegressor):
        y_train = np.array(y_train).ravel()
    pipeline.fit(X_train, y_train)
    return X_test, y_test


def compare_test_with_predicition(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series, model_name=None):
    predictions_test = pipeline.predict(X_test)
    df_test = pd.DataFrame({"num_parkings": y_test["num_parkings"]})
    df_pred = pd.DataFrame(predictions_test, columns=["predictions"])
    df_pred.index = df_test.index
    assert len(df_pred) == len(df_test)
    fig, ax = plt.subplots(figsize=(16, 4.5))
    print_metrics(df_test, df_pred, model_name)
    ax.set_title(f"Model {model_name}, test data(o) vs. predictions(x)")
    df_pred.plot(ax=ax, marker="x")
    df_test.plot(ax=ax, marker="o")
    ax.grid(True, which="both")
    ax.legend()
    display(pipeline)

def predict(pipeline: Pipeline, date_range: pd.DatetimeIndex, rain=None, temperature=None) -> Tuple[pd.DataFrame, int]:
    X_future = pd.DataFrame(index=date_range)
    if rain is not None:
        X_future["rain"] = rain
    if temperature is not None:
        X_future["temperature"] = temperature
    X_future = add_seasonability_columns(X_future)
    predictions = pipeline.predict(X_future)
    df_predictions = pd.DataFrame(predictions, columns=["num_parkings"])
    df_predictions.index=date_range
    return df_predictions, round(df_predictions["num_parkings"].sum())


def plot_predictions(df_predictions: pd.DataFrame, title=None):
    fig, ax = plt.subplots(figsize=(16, 4.5))
    if title is not None:
        ax.set_title(title)
    df_predictions.plot(ax=ax, marker="o")   
    ax.grid(True, which="both")
    ax.legend()


def print_metrics(test: pd.DataFrame, pred: pd.DataFrame, model_name: str):
    aligned = test.join(pred, how="inner")
    print(f"mean squared error for model {model_name}: {mean_squared_error(aligned["num_parkings"], aligned["predictions"])}")


def print_results(date_range: pd.DatetimeIndex, results: dict):
    print(f"Forecast period: {date_range.min()} - {date_range.max()}")
    base_total = results[BASE][1]
    for key, value in results.items():
        if key==BASE:
            print(f"Base forecast for parkings: {value[1]}.")

        else:    
            print(f"Forecast for parkings in {key}: {value[1]}.  Diff to base {value[1] - base_total} ({percent_difference(base_total, value[1])}%)")


def get_filtered_merged_data(parkings_file:str, zone=None, area=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if (zone is not None) and (area is not None):
        raise ValueError("Exactly one of 'zone' or 'area' must be provided.")
    
    df_parkings = get_parkings_df(parkings_file)
    if zone is not None: 
        filtered_parkings = get_parkings_in_zone(df_parkings, zone)     
    else:
        filtered_parkings = get_parkings_in_area(df_parkings, area)

    print(f"Filtered {len(filtered_parkings)} parkings.")
    hourly_parkings = get_hourly_parkings(filtered_parkings)
    # clean data
    # find the 99th percentile of the occupancy counts
    percentile99 = hourly_parkings["num_parkings"].quantile(0.99)
    # remove the outliers
    hourly_parkings = hourly_parkings[hourly_parkings["num_parkings"] < percentile99] 
    hourly_parkings["num_parkings"].ffill()
    hourly_parkings = hourly_parkings[hourly_parkings["ds"]> "2024-5-20"]
    
    df_rain, df_rain_bool, df_temperature = get_weather_data()
    hourly_parking = pd.merge(hourly_parkings, df_rain, on="ds", how="left")
    hourly_parking = pd.merge(hourly_parking, df_temperature, on="ds", how="left")

    df = hourly_parking.copy()
    df.set_index("ds", inplace=True)
    df_rain_train = pd.merge(hourly_parkings, df_rain, on="ds", how="left")
    df_rain_train.set_index("ds", inplace=True)
    df_rain_bool_train = pd.merge(hourly_parkings, df_rain_bool, on="ds", how="left")
    df_rain_bool_train.set_index("ds", inplace=True)
    df_temperature_train = pd.merge(hourly_parkings, df_temperature, on="ds", how="left")
    df_temperature_train.set_index("ds", inplace=True)
    df = add_seasonability_columns(df)
    df_rain_train = add_seasonability_columns(df_rain_train)
    df_rain_bool_train = add_seasonability_columns(df_rain_bool_train)
    df_temperature_train = add_seasonability_columns(df_temperature_train)
    return df, df_rain_train, df_rain_bool_train, df_temperature_train


def draw_results(results: dict):
    for key, value in results.items():
        plot_predictions(value[0], title=f"Predictions for {key}")


def percent_difference(old_value: float, new_value: float):
    return round(((new_value - old_value) / old_value) * 100, 2)