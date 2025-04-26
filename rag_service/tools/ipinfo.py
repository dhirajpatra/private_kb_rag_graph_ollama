import requests
import logging

logging.basicConfig(level=logging.INFO)

def get_current_location():
    """
    {
    "ip": "122.172.86.75",
    "hostname": "abts-kk-dynamic-075.86.172.122.airtelbroadband.in",
    "city": "Bengaluru",
    "region": "Karnataka",
    "country": "IN",
    "loc": "12.9719,77.5937",
    "org": "AS24560 Bharti Airtel Ltd., Telemedia Services",
    "postal": "562114",
    "timezone": "Asia/Kolkata",
    "readme": "https://ipinfo.io/missingauth"
    }
    """
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        return {
            "ip": data.get("ip"),
            "city": data.get("city"),
            "region": data.get("region"),
            "country": data.get("country"),
            "loc": data.get("loc")  # latitude,longitude
        }
    except Exception as e:
        return {"error": str(e)}

# # Example usage
# print(get_current_location())
