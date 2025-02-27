# Parkings and weather data correlation and forecasting

This project is for analysing and predicting parking usage rates in the City of Turku.
Weather data is used as an extra feature to analyze correlation and improve predictions.

The project has been bootstrapped with `uv` [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/).

## How to

### Data

Get necessary csv data and place it in root's `data` folder. All csv files in `data` folder are gitignored.
Following are base data files:

- `parkings_small.csv`
- `artukainen_rain.csv`
- `artukainen_temperature.csv`

File parkings_small.csv contains city of Turku paid parkings data from 2024-05-07 to 2025-02-06.
The necessary data columns are `time_start`, `time_end`, `zone_id` and `parking_area_id`.

- `time_start` and `time_end` tells when a single parking has occurred.
- `zone_id` tells which parking zone the parking was in.
- `parking_area_id` tells which parking area the parking was in.

Files `artukainen_rain` and `artukainen_temperature` contains hourly timestamped rain and temperature values.

### uv

To use `uv`, install it first ([https://docs.astral.sh/uv/getting-started/installation/#installation-methods](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)) e.g.:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Test everything works:

```bash
uv run main.py
```

Update used packages by running:

```bash
uv sync
```

Activate venv:

```bash
source .venv/bin/activate
```
