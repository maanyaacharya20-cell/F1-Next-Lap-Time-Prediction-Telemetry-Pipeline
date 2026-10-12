# F1 Next-Lap Time Prediction

A Python project for analyzing Formula 1 telemetry and predicting the duration of a driver's next lap. It uses OpenF1 REST API data, Pandas/NumPy feature engineering, SQLite storage, Scikit-learn, Joblib, and Streamlit.

## Architecture

- `src/api_client.py`: OpenF1 REST API client for sessions, drivers, laps, stints, and car telemetry.
- `src/processor.py`: DataFrame conversion and existing cleaning routines.
- `src/database.py`: SQLite schema and persistence. Existing tables remain available; session metadata and lap time windows are added for ML joins.
- `src/analysis.py`: Existing lap, tyre, degradation, and telemetry analytics.
- `src/ml_features.py`: Leakage-safe per-lap feature construction.
- `src/ml_model.py`: preprocessing, RandomForestRegressor training, evaluation, persistence, and prediction.
- `main.py`: downloads a race, runs the existing analytics pipeline, stores data, and trains the model.
- `app.py`: existing analytics dashboard plus the next-lap prediction form.

## Features and target

Each training row represents one driver's current lap. Features include:

- Previous lap duration
- Average and maximum speed
- Average throttle
- Average RPM
- Tyre compound and tyre age
- Lap number
- Driver
- Session name and circuit/track

The target is the next lap duration, created per session and driver with `shift(-1)`. The current lap's target is removed when no following lap exists. Telemetry is assigned to a lap using its timestamp and the lap's `date_start`/`date_end` window when those fields are available.

Invalid numeric values, infinite values, missing required lap values, non-positive lap numbers, and implausible lap durations are removed. Numeric features are median-imputed and categorical features are most-frequent-imputed then one-hot encoded with unknown-category handling.

## ML workflow

1. `main.py` downloads OpenF1 data for the selected race session.
2. Data is stored in SQLite, including session metadata and lap time windows.
3. `build_training_dataset` sorts by session, driver, and lap number, then creates past-only features and the shifted future target.
4. Rows remain in chronological order. The first 80% is used for training and the final 20% for testing, avoiding a random split across race order.
5. A `RandomForestRegressor` is trained in a Scikit-learn pipeline.
6. MAE, RMSE, and R² are printed and the fitted pipeline is saved to `models/next_lap_model.joblib`.

## Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download, process, store, and train:

```bash
python main.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

The prediction section appears after a model has been generated. It accepts lap conditions such as driver, tyre compound, tyre age, previous lap time, lap number, and telemetry summary values.

## Notes

OpenF1 can return incomplete telemetry or lap timing fields for some sessions. The original analytics and dashboard continue to work with partial data; ML training is skipped with a clear message when there are too few valid target rows.
 