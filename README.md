# Parkings and weather data correlation and forecasting

This project is for analysing and predicting mainly Turku City's parkings usage rates.
Weather data is used as extra feature data to improve forecasting.

The project has been bootstrapped with `uv` [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/).

## How to

### Data

Get necessary csv data and place it in root's `data` folder. All csv files in `data` folder are gitignored.
Following are base data files:

- `parkings_parking.2025-02-06.csv`
- ...

### uv

To use `uv`, install it first ([https://docs.astral.sh/uv/getting-started/installation/#installation-methods](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)) e.g.:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Test everything works:

```bash
uv run hello.py
```

Update used packages by running:

```bash
uv sync
```

Activate venv:

```bash
source .venv/bin/activate
```
