import numpy as np
import pandas as pd


class F1Analyzer:

    def lap_statistics(self, df):

        result = df.groupby("driver_number").agg(
            total_laps=("lap_number", "count"),
            average_lap_time=("lap_duration", "mean"),
            fastest_lap=("lap_duration", "min")
        ).reset_index()

        result["average_lap_time"] = np.round(
            result["average_lap_time"],
            3
        )

        result["fastest_lap"] = np.round(
            result["fastest_lap"],
            3
        )

        return result.sort_values(
            "average_lap_time"
        )

    def tire_analysis(self, df):

        result = df.groupby(
            ["driver_number", "compound"]
        ).agg(
            stints=("stint_number", "count"),
            average_tyre_age=(
                "tyre_age_at_start",
                "mean"
            )
        ).reset_index()

        result["average_tyre_age"] = np.round(
            result["average_tyre_age"],
            2
        )

        return result

    def stint_analysis(self, df):

        result = df.copy()

        result["stint_laps"] = (
            result["lap_end"]
            - result["lap_start"]
            + 1
        )

        return result

    def tire_degradation(self, laps, stints):

        records = []

        for _, stint in stints.iterrows():

            driver = stint["driver_number"]
            start = stint["lap_start"]
            end = stint["lap_end"]
            compound = stint["compound"]
            stint_number = stint["stint_number"]

            stint_laps = laps[
                (laps["driver_number"] == driver)
                &
                (laps["lap_number"] >= start)
                &
                (laps["lap_number"] <= end)
            ].copy()

            stint_laps = stint_laps.dropna(
                subset=["lap_duration"]
            )

            if len(stint_laps) < 4:
                continue

            stint_laps = stint_laps.sort_values(
                "lap_number"
            )

            count = max(
                2,
                int(len(stint_laps) * 0.25)
            )

            early = stint_laps.head(count)
            late = stint_laps.tail(count)

            early_average = early[
                "lap_duration"
            ].mean()

            late_average = late[
                "lap_duration"
            ].mean()

            degradation = (
                late_average - early_average
            )

            records.append({
                "driver_number": driver,
                "compound": compound,
                "stint_number": stint_number,
                "early_average": round(
                    early_average,
                    3
                ),
                "late_average": round(
                    late_average,
                    3
                ),
                "degradation": round(
                    degradation,
                    3
                )
            })

        return pd.DataFrame(records)

    def telemetry_summary(self, df):

        if df.empty:
            return pd.DataFrame()

        summary = {
            "max_speed": df["speed"].max(),
            "average_speed": df["speed"].mean(),
            "max_rpm": df["rpm"].max(),
            "average_throttle": df["throttle"].mean(),
            "braking_samples": (
                (df["brake"] > 0).sum()
            )
        }

        return pd.DataFrame(
            [summary]
        )