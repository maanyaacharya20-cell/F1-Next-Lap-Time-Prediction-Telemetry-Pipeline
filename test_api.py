import requests

url = "https://api.openf1.org/v1/stints"

params = {
    "session_key": 9165
}

response = requests.get(url, params=params)

print(response.status_code)
print(response.json()[:3])