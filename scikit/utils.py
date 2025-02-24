import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error



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
TEMPERATURE="temperature"
ONEHOT_SCALER_COLUMNS = [HOUR, MONTH, DAY_OF_WEEK]
STANDARD_SCALER_COLUMNS = [RAIN, TEMPERATURE, YEAR]

SEASONABILITY_COLUMNS = [YEAR, MONTH, DAY_OF_WEEK, DAY_OF_YEAR, HOUR, HOUR_SIN]
SEASONABILITY_COLUMNS = [YEAR, MONTH_SIN, DAY_OF_WEEK_SIN, HOUR_SIN]
SEASONABILITY_COLUMNS = [YEAR, MONTH, DAY_OF_WEEK, HOUR]


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


def get_weather_data(rain_threshold=3.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_rain = pd.read_csv("../data/artukainen_rain.csv")
    df_rain["timestamp"] = pd.to_datetime(df_rain["timestamp"])   
    df_rain["value"] = df_rain["value"] >= rain_threshold
    df_rain = df_rain.rename(columns={"timestamp": "ds", "value": "rain"})
    df_temperature = pd.read_csv("../data/artukainen_temperature.csv")
    df_temperature["timestamp"] = pd.to_datetime(df_temperature["timestamp"])
    df_temperature = df_temperature.rename(columns={"timestamp": "ds", "value": "temperature"})
    return df_rain, df_temperature
       

def get_pipeline(standard_columns=STANDARD_SCALER_COLUMNS, onehot_colums=ONEHOT_SCALER_COLUMNS) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), onehot_colums),
            ("Standard", StandardScaler(), standard_columns)
        
        ],
        force_int_remainder_cols=False
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    return pipeline


def fit_model(pipeline: Pipeline, df: pd.DataFrame, feature_columns: list) -> Tuple[pd.Series, pd.Series]:
    X = df[feature_columns]
    y = df[["num_parkings"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    pipeline.fit(X_train, y_train)
    return X_test, y_test


def compare_test_with_predicition(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series):
    predictions_test = pipeline.predict(X_test)
    df_test = pd.DataFrame({"num_parkings": y_test["num_parkings"]})
    df_pred = pd.DataFrame(predictions_test, columns=["pred"])
    df_pred.index = df_test.index
    assert len(df_pred) == len(df_test)
    print_metrics(df_test, df_pred)
    fig, ax = plt.subplots(figsize=(16, 4.5))
    ax.set_title("Test data(o) vs. Predictions(x)")
    df_pred.plot(ax=ax, marker="x")
    df_test.plot(ax=ax, marker="o")
    ax.grid(True)
    ax.legend()
    return pipeline


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


def plot_predictions(df_predictions: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(16, 4.5))
    df_predictions.plot(ax=ax, marker="x")   
    ax.grid(True)
    ax.legend()


def print_metrics(test: pd.DataFrame, pred: pd.DataFrame):
    aligned = test.join(pred, how="inner")
    print(f"mean squared error: {mean_squared_error(aligned["num_parkings"], aligned["pred"])}")
    print(f"mean absolute percentage error {mean_absolute_percentage_error(aligned["num_parkings"], aligned["pred"])}")

