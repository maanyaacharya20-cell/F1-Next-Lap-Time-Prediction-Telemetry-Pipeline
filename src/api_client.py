import requests


class F1APIClient:

    BASE_URL = "https://api.openf1.org/v1"

    def get_data(self, endpoint, params=None):

        url = f"{self.BASE_URL}/{endpoint}"

        response = requests.get(
            url,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        return response.json()

    def get_sessions(self, year):

        return self.get_data(
            "sessions",
            {
                "year": year
            }
        )

    def get_drivers(self, session_key):

        return self.get_data(
            "drivers",
            {
                "session_key": session_key
            }
        )

    def get_laps(self, session_key):

        return self.get_data(
            "laps",
            {
                "session_key": session_key
            }
        )

    def get_stints(self, session_key):

        return self.get_data(
            "stints",
            {
                "session_key": session_key
            }
        )

    def get_car_data(self, session_key, driver_number):

        return self.get_data(
            "car_data",
            {
                "session_key": session_key,
                "driver_number": driver_number
            }
        )