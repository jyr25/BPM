import time
import hmac
import hashlib
import requests
from urllib.parse import quote

API_KEY = "your_api_key".strip()
SECRET = "your_secret".strip()
BASE_URL = "https://api.statiz.co.kr/baseballApi"


def normalize_query(params: dict) -> str:
    safe = "-_.!~*'()"
    return "&".join(
        f"{quote(str(k), safe=safe)}={quote(str(params[k]), safe=safe)}"
        for k in sorted(params.keys())
    )


def make_signature(secret: str, payload: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


def call_api(METHOD, PATH, QUERY=None):

    QUERY = QUERY or {}

    normalized = normalize_query(QUERY)

    timestamp = str(int(time.time()))

    payload = f"{METHOD}|{PATH}|{normalized}|{timestamp}"

    signature = make_signature(SECRET, payload)

    url = f"{BASE_URL}/{PATH}"

    if normalized:
        url += f"?{normalized}"

    headers = {
        "X-API-KEY": API_KEY,
        "X-TIMESTAMP": timestamp,
        "X-SIGNATURE": signature,
    }

    response = requests.request(METHOD, url, headers=headers, timeout=30)

    if response.status_code != 200:
        print("http_code:", response.status_code)
        print(response.text)
        return None

    return response.json()
