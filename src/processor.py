import pandas as pd


class F1DataProcessor:

    def drivers_to_dataframe(self, drivers):

        return pd.DataFrame(drivers)

    def laps_to_dataframe(self, laps):

        return pd.DataFrame(laps)

    def stints_to_dataframe(self, stints):

        return pd.DataFrame(stints)

    def telemetry_to_dataframe(self, telemetry):

        return pd.DataFrame(telemetry)

    def clean_laps(self, df):

        df = df.copy()

        required = [
            "driver_number",
            "lap_number",
            "lap_duration"
        ]

        available = [
            column
            for column in required
            if column in df.columns
        ]

        df = df.dropna(
            subset=available
        )

        return df

    def clean_stints(self, df):

        df = df.copy()

        required = [
            "driver_number",
            "stint_number",
            "compound"
        ]

        available = [
            column
            for column in required
            if column in df.columns
        ]

        df = df.dropna(
            subset=available
        )

        return df

    def clean_telemetry(self, df):

        df = df.copy()

        if df.empty:
            return df

        required = [
            "driver_number",
            "date",
            "speed",
            "throttle",
            "brake",
            "rpm",
            "n_gear"
        ]

        available = [
            column
            for column in required
            if column in df.columns
        ]

        df = df.dropna(
            subset=available
        )

        return df