"""Feature engineering for next-lap duration prediction."""

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "previous_lap_time",
    "average_speed",
    "max_speed",
    "average_throttle",
    "average_rpm",
    "tyre_compound",
    "tyre_age",
    "lap_number",
    "driver",
    "session_name",
    "track",
]
TARGET_COLUMN = "next_lap_duration"
NUMERIC_FEATURES = [
    "previous_lap_time",
    "average_speed",
    "max_speed",
    "average_throttle",
    "average_rpm",
    "tyre_age",
    "lap_number",
]
CATEGORICAL_FEATURES = [
    "tyre_compound",
    "driver",
    "session_name",
    "track",
]


def _frame(value):
    if value is None:
        return pd.DataFrame()
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)


def _number(series):
    return pd.to_numeric(series, errors="coerce")


def _telemetry_features(laps, telemetry):
    result = pd.DataFrame(index=laps.index)
    for column in ["average_speed", "max_speed", "average_throttle", "average_rpm"]:
        result[column] = np.nan

    if telemetry.empty:
        return result

    telemetry = telemetry.copy()
    telemetry["driver_number"] = _number(telemetry["driver_number"])
    for column in ["speed", "throttle", "rpm"]:
        if column in telemetry:
            telemetry[column] = _number(telemetry[column])

    if "lap_number" in telemetry.columns:
        telemetry["lap_number"] = _number(telemetry["lap_number"])
        grouped = telemetry.groupby(["driver_number", "lap_number"])
        aggregates = grouped.agg(
            average_speed=("speed", "mean"),
            max_speed=("speed", "max"),
            average_throttle=("throttle", "mean"),
            average_rpm=("rpm", "mean"),
        ).reset_index()
        keys = laps[["driver_number", "lap_number"]].copy()
        keys["_index"] = laps.index
        merged = keys.merge(aggregates, on=["driver_number", "lap_number"], how="left")
        return merged.set_index("_index")[result.columns].reindex(laps.index)

    if not {"date_start", "date_end"}.issubset(laps.columns):
        return result

    telemetry["date"] = pd.to_datetime(telemetry["date"], errors="coerce", utc=True)
    telemetry = telemetry.dropna(subset=["date"])
    if telemetry.empty:
        return result

    rows = []
    for index, lap in laps.iterrows():
        start = pd.to_datetime(lap["date_start"], errors="coerce", utc=True)
        end = pd.to_datetime(lap["date_end"], errors="coerce", utc=True)
        if pd.isna(start) or pd.isna(end):
            continue
        samples = telemetry[
            (telemetry["driver_number"] == lap["driver_number"])
            & (telemetry["date"] >= start)
            & (telemetry["date"] <= end)
        ]
        if not samples.empty:
            rows.append({
                "_index": index,
                "average_speed": samples["speed"].mean(),
                "max_speed": samples["speed"].max(),
                "average_throttle": samples["throttle"].mean(),
                "average_rpm": samples["rpm"].mean(),
            })

    if rows:
        aggregated = pd.DataFrame(rows).set_index("_index")
        return aggregated.reindex(laps.index)[result.columns]
    return result


def build_training_dataset(laps, stints=None, telemetry=None, drivers=None, session=None):
    """Build leakage-safe, one-row-per-lap training data."""
    laps = _frame(laps)
    if laps.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS + [TARGET_COLUMN])

    required = {"driver_number", "lap_number", "lap_duration"}
    missing = required.difference(laps.columns)
    if missing:
        raise ValueError(f"Lap data is missing required columns: {sorted(missing)}")

    laps = laps.copy()
    for column in ["driver_number", "lap_number", "lap_duration"]:
        laps[column] = _number(laps[column])
    laps = laps.replace([np.inf, -np.inf], np.nan)
    laps = laps[laps["lap_duration"].between(0.1, 600)]
    laps = laps[laps["lap_number"] > 0]
    if laps.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS + [TARGET_COLUMN])

    if "session_key" not in laps:
        laps["session_key"] = "default"
    laps["session_key"] = laps["session_key"].fillna("default").astype(str)
    laps = laps.sort_values(["session_key", "driver_number", "lap_number"]).reset_index(drop=True)
    laps["previous_lap_time"] = laps.groupby(["session_key", "driver_number"])["lap_duration"].shift(1)
    laps[TARGET_COLUMN] = laps.groupby(["session_key", "driver_number"])["lap_duration"].shift(-1)

    drivers = _frame(drivers)
    if not drivers.empty and {"driver_number", "full_name"}.issubset(drivers.columns):
        driver_names = drivers[["driver_number", "full_name"]].drop_duplicates("driver_number")
        driver_names = driver_names.rename(columns={"full_name": "driver"})
        laps = laps.merge(driver_names, on="driver_number", how="left")
    elif not drivers.empty and {"driver_number", "name"}.issubset(drivers.columns):
        driver_names = drivers[["driver_number", "name"]].drop_duplicates("driver_number")
        driver_names = driver_names.rename(columns={"name": "driver"})
        laps = laps.merge(driver_names, on="driver_number", how="left")
    else:
        laps["driver"] = laps["driver_number"].astype("Int64").astype(str)
    laps["driver"] = laps["driver"].fillna(laps["driver_number"].astype(str))

    stints = _frame(stints)
    laps["tyre_compound"] = "Unknown"
    laps["tyre_age"] = np.nan
    if not stints.empty and {"driver_number", "lap_start", "lap_end"}.issubset(stints.columns):
        stint_columns = ["driver_number", "lap_start", "lap_end"]
        for column in ["compound", "tyre_age_at_start"]:
            if column in stints:
                stint_columns.append(column)
        stints = stints[stint_columns].copy()
        stints["driver_number"] = _number(stints["driver_number"])
        stints["lap_start"] = _number(stints["lap_start"])
        stints["lap_end"] = _number(stints["lap_end"])
        matches = []
        for index, lap in laps.iterrows():
            candidates = stints[
                (stints["driver_number"] == lap["driver_number"])
                & (stints["lap_start"] <= lap["lap_number"])
                & (stints["lap_end"] >= lap["lap_number"])
            ]
            if not candidates.empty:
                stint = candidates.iloc[0]
                start_age = _number(pd.Series([stint.get("tyre_age_at_start")])).iloc[0]
                matches.append((index, stint.get("compound", "Unknown"), start_age + lap["lap_number"] - stint["lap_start"]))
        for index, compound, age in matches:
            laps.loc[index, "tyre_compound"] = compound
            laps.loc[index, "tyre_age"] = age

    telemetry = _frame(telemetry)
    telemetry_summary = _telemetry_features(laps, telemetry)
    for column in telemetry_summary.columns:
        laps[column] = telemetry_summary[column].to_numpy()

    session = session or {}
    if isinstance(session, pd.DataFrame):
        session = session.iloc[0].to_dict() if not session.empty else {}
    laps["session_name"] = session.get("session_name", "Unknown")
    laps["track"] = session.get("circuit_short_name", session.get("track", "Unknown"))

    result = laps[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result[result[TARGET_COLUMN].between(0.1, 600)]
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].fillna("Unknown").astype(str)
    return result.reset_index(drop=True)


prepare_features = build_training_dataset
