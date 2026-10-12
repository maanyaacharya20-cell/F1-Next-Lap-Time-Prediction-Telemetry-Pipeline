from src.api_client import F1APIClient
from src.processor import F1DataProcessor
from src.analysis import F1Analyzer
from src.database import F1Database
from src.ml_features import build_training_dataset
from src.ml_model import train_model


def main():

    print("=" * 60)
    print("F1 TELEMETRY & DATA ANALYTICS PIPELINE")
    print("=" * 60)

    client = F1APIClient()

    processor = F1DataProcessor()

    analyzer = F1Analyzer()

    # --------------------------------------------------
    # 1. FETCH SESSION
    # --------------------------------------------------

    print("\n[1] Fetching 2024 sessions...")

    sessions = client.get_sessions(2024)

    race_sessions = [
        session
        for session in sessions
        if session.get("session_type") == "Race"
        and session.get("date_start")
    ]

    if not race_sessions:

        print("No race session found.")

        return

    session = race_sessions[0]

    session_key = session["session_key"]

    print(
        "Race:",
        session.get("session_name")
    )

    print(
        "Circuit:",
        session.get("circuit_short_name")
    )

    print(
        "Country:",
        session.get("country_name")
    )

    print(
        "Session:",
        session_key
    )

    # --------------------------------------------------
    # 2. DOWNLOAD RACE DATA
    # --------------------------------------------------

    print("\n[2] Downloading race data...")

    drivers = client.get_drivers(
        session_key
    )

    laps = client.get_laps(
        session_key
    )

    stints = client.get_stints(
        session_key
    )

    print(
        "Drivers:",
        len(drivers)
    )

    print(
        "Laps:",
        len(laps)
    )

    print(
        "Stints:",
        len(stints)
    )

    # --------------------------------------------------
    # 3. DOWNLOAD TELEMETRY
    # --------------------------------------------------

    print(
        "\n[3] Downloading car telemetry..."
    )

    telemetry = []

    for driver in drivers:

        driver_number = driver[
            "driver_number"
        ]

        print(
            f"  Driver {driver_number}..."
        )

        try:

            driver_telemetry = (
                client.get_car_data(
                    session_key,
                    driver_number
                )
            )

            telemetry.extend(
                driver_telemetry
            )

            print(
                f"    Records: "
                f"{len(driver_telemetry)}"
            )

        except Exception as error:

            print(
                f"    Telemetry unavailable: "
                f"{error}"
            )

    print(
        "\nTotal telemetry records:",
        len(telemetry)
    )

    # --------------------------------------------------
    # 4. PROCESS DATA
    # --------------------------------------------------

    print("\n[4] Processing data...")

    drivers_df = (
        processor.drivers_to_dataframe(
            drivers
        )
    )

    laps_df = (
        processor.laps_to_dataframe(
            laps
        )
    )

    stints_df = (
        processor.stints_to_dataframe(
            stints
        )
    )

    telemetry_df = (
        processor.telemetry_to_dataframe(
            telemetry
        )
    )

    laps_df = processor.clean_laps(
        laps_df
    )

    stints_df = processor.clean_stints(
        stints_df
    )

    telemetry_df = (
        processor.clean_telemetry(
            telemetry_df
        )
    )

    print(
        "Processed lap records:",
        len(laps_df)
    )

    print(
        "Processed stint records:",
        len(stints_df)
    )

    print(
        "Processed telemetry records:",
        len(telemetry_df)
    )

    # --------------------------------------------------
    # 5. ANALYSIS
    # --------------------------------------------------

    print("\n[5] Running analytics...")

    lap_stats = analyzer.lap_statistics(
        laps_df
    )

    print(
        "\n===== LAP STATISTICS ====="
    )

    print(
        lap_stats.to_string(
            index=False
        )
    )

    tire_stats = analyzer.tire_analysis(
        stints_df
    )

    print(
        "\n===== TIRE STRATEGY ====="
    )

    print(
        tire_stats.to_string(
            index=False
        )
    )

    degradation = (
        analyzer.tire_degradation(
            laps_df,
            stints_df
        )
    )

    print(
        "\n===== TIRE DEGRADATION ====="
    )

    if not degradation.empty:

        print(
            degradation.to_string(
                index=False
            )
        )

    else:

        print(
            "No degradation data available."
        )

    if not telemetry_df.empty:

        telemetry_summary = (
            analyzer.telemetry_summary(
                telemetry_df
            )
        )

        print(
            "\n===== TELEMETRY SUMMARY ====="
        )

        print(
            telemetry_summary.to_string(
                index=False
            )
        )

    # --------------------------------------------------
    # 6. DATABASE
    # --------------------------------------------------

    print("\n[6] Creating database...")

    database = F1Database()

    database.create_tables()

    database.insert_session(
        session
    )

    database.insert_drivers(
        drivers
    )

    database.insert_laps(
        laps
    )

    database.insert_stints(
        stints
    )

    database.insert_telemetry(
        telemetry
    )

    print(
        "Data inserted into database."
    )

    # --------------------------------------------------
    # 7. NEXT-LAP MACHINE LEARNING
    # --------------------------------------------------

    print("\n[7] Training next-lap prediction model...")

    training_data = build_training_dataset(
        laps,
        stints,
        telemetry,
        drivers,
        session=session
    )

    try:
        _, metrics = train_model(
            training_data
        )
        print(
            "Model metrics:",
            f"MAE={metrics['mae']:.3f},",
            f"RMSE={metrics['rmse']:.3f},",
            f"R2={metrics['r2']:.3f}"
        )
        print(
            "Training rows:",
            metrics["train_rows"],
            "Testing rows:",
            metrics["test_rows"]
        )
    except ValueError as error:
        print(
            "Model training skipped:",
            error
        )

    # --------------------------------------------------
    # 8. SQL JOIN
    # --------------------------------------------------

    print(
        "\n[7] Running SQL JOIN analysis..."
    )

    query = """
        SELECT
            d.driver_number,
            d.name,
            d.team,
            COUNT(l.lap_number)
                AS total_laps,
            ROUND(
                AVG(l.lap_duration),
                3
            ) AS average_lap_time,
            ROUND(
                MIN(l.lap_duration),
                3
            ) AS fastest_lap
        FROM drivers d

        INNER JOIN laps l
            ON d.driver_number =
               l.driver_number

        GROUP BY
            d.driver_number,
            d.name,
            d.team

        ORDER BY
            average_lap_time ASC
    """

    columns, rows = database.run_query(
        query
    )

    print(
        "\n===== DRIVER PERFORMANCE ====="
    )

    print(
        " | ".join(columns)
    )

    for row in rows:

        print(
            " | ".join(
                str(value)
                for value in row
            )
        )

    # --------------------------------------------------
    # 9. TELEMETRY SQL QUERY
    # --------------------------------------------------

    if telemetry:

        telemetry_query = """
            SELECT
                d.name,
                d.team,
                ROUND(
                    MAX(t.speed),
                    2
                ) AS max_speed
            FROM drivers d

            INNER JOIN telemetry t
                ON d.driver_number =
                   t.driver_number

            GROUP BY
                d.driver_number,
                d.name,
                d.team

            ORDER BY
                max_speed DESC
        """

        columns, rows = (
            database.run_query(
                telemetry_query
            )
        )

        print(
            "\n===== TOP SPEED ANALYSIS ====="
        )

        print(
            " | ".join(columns)
        )

        for row in rows:

            print(
                " | ".join(
                    str(value)
                    for value in row
                )
            )

    database.close()

    print("\n" + "=" * 60)
    print(
        "PIPELINE COMPLETED SUCCESSFULLY"
    )
    print("=" * 60)


if __name__ == "__main__":

    main()