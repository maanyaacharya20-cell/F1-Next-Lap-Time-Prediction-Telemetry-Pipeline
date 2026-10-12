import sqlite3
import os


class F1Database:

    def __init__(
        self,
        db_name="f1_telemetry.db"
    ):

        project_folder = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        self.db_path = os.path.join(
            project_folder,
            db_name
        )

        self.connection = sqlite3.connect(
            self.db_path
        )

    def create_tables(self):

        cursor = self.connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drivers (
                driver_number INTEGER PRIMARY KEY,
                name TEXT,
                team TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS laps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_number INTEGER,
                lap_number INTEGER,
                lap_duration REAL,
                date_start TEXT,
                date_end TEXT,
                session_key INTEGER,
                FOREIGN KEY(driver_number)
                    REFERENCES drivers(driver_number)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_key INTEGER PRIMARY KEY,
                session_name TEXT,
                session_type TEXT,
                circuit_short_name TEXT,
                country_name TEXT,
                year INTEGER
            )
        """)

        lap_columns = {
            row[1]
            for row in cursor.execute("PRAGMA table_info(laps)")
        }
        for column in ["date_start", "date_end", "session_key"]:
            if column not in lap_columns:
                cursor.execute(
                    f"ALTER TABLE laps ADD COLUMN {column} "
                    f"{'INTEGER' if column == 'session_key' else 'TEXT'}"
                )

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_number INTEGER,
                stint_number INTEGER,
                compound TEXT,
                lap_start INTEGER,
                lap_end INTEGER,
                tyre_age_at_start INTEGER,
                FOREIGN KEY(driver_number)
                    REFERENCES drivers(driver_number)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_number INTEGER,
                date TEXT,
                speed REAL,
                throttle REAL,
                brake REAL,
                rpm REAL,
                gear INTEGER,
                drs INTEGER,
                FOREIGN KEY(driver_number)
                    REFERENCES drivers(driver_number)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_laps_driver
            ON laps(driver_number)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_stints_driver
            ON stints(driver_number)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_telemetry_driver
            ON telemetry(driver_number)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_laps_driver_lap
            ON laps(driver_number, lap_number)
        """)

        self.connection.commit()

        print("Database tables created.")

    def insert_drivers(self, drivers):

        cursor = self.connection.cursor()

        data = []

        for driver in drivers:

            data.append((
                driver.get("driver_number"),
                driver.get("full_name"),
                driver.get("team_name")
            ))

        cursor.executemany("""
            INSERT OR REPLACE INTO drivers
            (
                driver_number,
                name,
                team
            )
            VALUES (?, ?, ?)
        """, data)

        self.connection.commit()

    def insert_laps(self, laps):

        cursor = self.connection.cursor()

        data = []

        for lap in laps:

            data.append((
                lap.get("driver_number"),
                lap.get("lap_number"),
                lap.get("lap_duration"),
                lap.get("date_start"),
                lap.get("date_end"),
                lap.get("session_key")
            ))

        cursor.executemany("""
            INSERT INTO laps
            (
                driver_number,
                lap_number,
                lap_duration,
                date_start,
                date_end,
                session_key
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, data)

        self.connection.commit()

    def insert_session(self, session):

        if not session:
            return

        self.connection.execute("""
            INSERT OR REPLACE INTO sessions
            (session_key, session_name, session_type,
             circuit_short_name, country_name, year)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session.get("session_key"),
            session.get("session_name"),
            session.get("session_type"),
            session.get("circuit_short_name"),
            session.get("country_name"),
            session.get("year"),
        ))
        self.connection.commit()

    def insert_stints(self, stints):

        cursor = self.connection.cursor()

        data = []

        for stint in stints:

            data.append((
                stint.get("driver_number"),
                stint.get("stint_number"),
                stint.get("compound"),
                stint.get("lap_start"),
                stint.get("lap_end"),
                stint.get("tyre_age_at_start")
            ))

        cursor.executemany("""
            INSERT INTO stints
            (
                driver_number,
                stint_number,
                compound,
                lap_start,
                lap_end,
                tyre_age_at_start
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, data)

        self.connection.commit()

    def insert_telemetry(self, telemetry):

        if not telemetry:
            return

        cursor = self.connection.cursor()

        data = []

        for record in telemetry:

            data.append((
                record.get("driver_number"),
                record.get("date"),
                record.get("speed"),
                record.get("throttle"),
                record.get("brake"),
                record.get("rpm"),
                record.get("n_gear"),
                record.get("drs")
            ))

        cursor.executemany("""
            INSERT INTO telemetry
            (
                driver_number,
                date,
                speed,
                throttle,
                brake,
                rpm,
                gear,
                drs
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, data)

        self.connection.commit()

    def run_query(self, query):

        cursor = self.connection.cursor()

        cursor.execute(query)

        columns = [
            description[0]
            for description in cursor.description
        ]

        rows = cursor.fetchall()

        return columns, rows

    def close(self):

        self.connection.close()