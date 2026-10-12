import os
import sqlite3

import joblib
import pandas as pd
import streamlit as st

from src.ml_model import predict_next_lap


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="F1 Telemetry Analytics",
    page_icon="🏎️",
    layout="wide"
)


# --------------------------------------------------
# DATABASE PATH
# --------------------------------------------------

DB = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "f1_telemetry.db"
)


# --------------------------------------------------
# LOAD DATABASE
# --------------------------------------------------

@st.cache_data
def load_database():

    connection = sqlite3.connect(DB)

    drivers = pd.read_sql_query(
        "SELECT * FROM drivers",
        connection
    )

    laps = pd.read_sql_query(
        "SELECT * FROM laps",
        connection
    )

    stints = pd.read_sql_query(
        "SELECT * FROM stints",
        connection
    )

    telemetry = pd.read_sql_query(
        "SELECT * FROM telemetry",
        connection
    )

    connection.close()

    return (
        drivers,
        laps,
        stints,
        telemetry
    )


# --------------------------------------------------
# CHECK DATABASE
# --------------------------------------------------

if not os.path.exists(DB):

    st.error(
        "Database not found."
    )

    st.info(
        "Run `python main.py` first."
    )

    st.stop()


try:

    (
        drivers,
        laps,
        stints,
        telemetry
    ) = load_database()

except Exception as error:

    st.error(
        f"Unable to load database: {error}"
    )

    st.stop()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title(
    "🏎️ F1 Telemetry & Data Analytics"
)

st.markdown(
    """
    **Formula 1 Race Data Analysis Pipeline**

    OpenF1 → Python OOP → Pandas → NumPy → SQLite → SQL → Streamlit
    """
)

st.divider()


# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Drivers",
        len(drivers)
    )

with col2:

    st.metric(
        "Lap Records",
        len(laps)
    )

with col3:

    st.metric(
        "Tire Stints",
        len(stints)
    )

with col4:

    st.metric(
        "Telemetry Records",
        len(telemetry)
    )


st.divider()


# --------------------------------------------------
# DRIVER PERFORMANCE
# --------------------------------------------------

st.header(
    "Driver Performance"
)

performance_query = """
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

connection = sqlite3.connect(DB)

performance = pd.read_sql_query(
    performance_query,
    connection
)

connection.close()

st.dataframe(
    performance,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# AVERAGE LAP TIME
# --------------------------------------------------

st.subheader(
    "Average Lap Time by Driver"
)

chart_data = performance.set_index(
    "name"
)[
    ["average_lap_time"]
]

st.bar_chart(
    chart_data
)


st.divider()


# --------------------------------------------------
# DRIVER SELECTOR
# --------------------------------------------------

st.header(
    "Driver Analysis"
)

driver_options = drivers[
    ["driver_number", "name"]
].drop_duplicates()

driver_options = driver_options.sort_values(
    "driver_number"
)

selected_driver = st.selectbox(
    "Select Driver",
    driver_options["name"].tolist()
)

selected_number = driver_options[
    driver_options["name"]
    == selected_driver
]["driver_number"].iloc[0]


# --------------------------------------------------
# DRIVER METRICS
# --------------------------------------------------

driver_laps = laps[
    laps["driver_number"]
    == selected_number
].copy()

driver_stints = stints[
    stints["driver_number"]
    == selected_number
].copy()

driver_telemetry = telemetry[
    telemetry["driver_number"]
    == selected_number
].copy()


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Laps",
        len(driver_laps)
    )

with col2:

    average_lap = (
        driver_laps["lap_duration"]
        .mean()
    )

    st.metric(
        "Average Lap",
        f"{average_lap:.3f} sec"
        if pd.notna(average_lap)
        else "N/A"
    )

with col3:

    fastest_lap = (
        driver_laps["lap_duration"]
        .min()
    )

    st.metric(
        "Fastest Lap",
        f"{fastest_lap:.3f} sec"
        if pd.notna(fastest_lap)
        else "N/A"
    )


# --------------------------------------------------
# LAP PROGRESSION
# --------------------------------------------------

st.subheader(
    "Lap Time Progression"
)

lap_chart = driver_laps[
    [
        "lap_number",
        "lap_duration"
    ]
].dropna()

if not lap_chart.empty:

    lap_chart = lap_chart.set_index(
        "lap_number"
    )

    st.line_chart(
        lap_chart
    )

else:

    st.info(
        "No lap data available."
    )


# --------------------------------------------------
# TIRE STRATEGY
# --------------------------------------------------

st.subheader(
    "Tire Strategy"
)

if not driver_stints.empty:

    tire_table = driver_stints[
        [
            "stint_number",
            "compound",
            "lap_start",
            "lap_end",
            "tyre_age_at_start"
        ]
    ].copy()

    tire_table["stint_laps"] = (
        tire_table["lap_end"]
        - tire_table["lap_start"]
        + 1
    )

    st.dataframe(
        tire_table,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No tire strategy data available."
    )


# --------------------------------------------------
# TIRE DEGRADATION
# --------------------------------------------------

st.subheader(
    "Tire Degradation"
)

degradation_records = []

for _, stint in driver_stints.iterrows():

    start = stint["lap_start"]

    end = stint["lap_end"]

    stint_laps = driver_laps[
        (driver_laps["lap_number"] >= start)
        &
        (driver_laps["lap_number"] <= end)
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

    early = (
        stint_laps
        .head(count)["lap_duration"]
        .mean()
    )

    late = (
        stint_laps
        .tail(count)["lap_duration"]
        .mean()
    )

    degradation_records.append({

        "stint_number":
            stint["stint_number"],

        "compound":
            stint["compound"],

        "early_average":
            round(early, 3),

        "late_average":
            round(late, 3),

        "degradation":
            round(late - early, 3)
    })


degradation_df = pd.DataFrame(
    degradation_records
)

if not degradation_df.empty:

    st.dataframe(
        degradation_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Not enough lap data for degradation analysis."
    )


st.divider()


# --------------------------------------------------
# TELEMETRY
# --------------------------------------------------

st.header(
    "Car Telemetry"
)

if not driver_telemetry.empty:

    telemetry_options = {
        "Speed": "speed",
        "Throttle": "throttle",
        "Brake": "brake",
        "RPM": "rpm",
        "Gear": "n_gear"
    }

    selected_metric = st.selectbox(
        "Telemetry Metric",
        list(telemetry_options.keys())
    )

    column = telemetry_options[
        selected_metric
    ]

    chart = driver_telemetry[
        ["date", column]
    ].copy()

    chart["date"] = pd.to_datetime(
        chart["date"],
        format="mixed",
        utc=True
    )

    chart = chart.sort_values(
        "date"
    )

    chart = chart.set_index(
        "date"
    )

    st.line_chart(
        chart
    )

else:

    st.info(
        "No telemetry data available for this driver."
    )

# --------------------------------------------------
# DRIVER COMPARISON
# --------------------------------------------------

st.divider()

st.header(
    "Driver Comparison"
)

comparison = performance[
    [
        "name",
        "team",
        "total_laps",
        "average_lap_time",
        "fastest_lap"
    ]
].copy()

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# NEXT-LAP PREDICTION
# --------------------------------------------------

st.divider()

st.header(
    "Next-Lap Time Prediction"
)

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "next_lap_model.joblib"
)

if not os.path.exists(model_path):

    st.info(
        "Train the model by running `python main.py` first."
    )

else:

    model = joblib.load(model_path)
    compound_options = sorted(
        stints["compound"].dropna().astype(str).unique().tolist()
    )
    if not compound_options:
        compound_options = ["Unknown"]

    prediction_col1, prediction_col2 = st.columns(2)

    with prediction_col1:
        prediction_driver = st.selectbox(
            "Driver",
            driver_options["name"].tolist(),
            key="prediction_driver"
        )
        prediction_compound = st.selectbox(
            "Tyre compound",
            compound_options,
            key="prediction_compound"
        )
        prediction_lap = st.number_input(
            "Current lap number",
            min_value=1,
            value=1,
            step=1,
            key="prediction_lap"
        )
        prediction_tyre_age = st.number_input(
            "Tyre age",
            min_value=0,
            value=1,
            step=1,
            key="prediction_tyre_age"
        )

    with prediction_col2:
        prediction_previous = st.number_input(
            "Previous lap time (seconds)",
            min_value=0.0,
            value=90.0,
            step=0.1,
            key="prediction_previous"
        )
        prediction_average_speed = st.number_input(
            "Average speed",
            min_value=0.0,
            value=200.0,
            step=1.0,
            key="prediction_average_speed"
        )
        prediction_max_speed = st.number_input(
            "Maximum speed",
            min_value=0.0,
            value=320.0,
            step=1.0,
            key="prediction_max_speed"
        )
        prediction_throttle = st.number_input(
            "Average throttle",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0,
            key="prediction_throttle"
        )
        prediction_rpm = st.number_input(
            "Average RPM",
            min_value=0.0,
            value=10000.0,
            step=100.0,
            key="prediction_rpm"
        )

    if st.button("Predict next lap", type="primary"):
        conditions = {
            "previous_lap_time": prediction_previous,
            "average_speed": prediction_average_speed,
            "max_speed": prediction_max_speed,
            "average_throttle": prediction_throttle,
            "average_rpm": prediction_rpm,
            "tyre_compound": prediction_compound,
            "tyre_age": prediction_tyre_age,
            "lap_number": prediction_lap,
            "driver": str(prediction_driver),
            "session_name": "Race",
            "track": "Unknown",
        }
        prediction = predict_next_lap(model, conditions)[0]
        st.success(f"Predicted next-lap time: {prediction:.3f} seconds")
