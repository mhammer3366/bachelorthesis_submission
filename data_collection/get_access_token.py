import base64
import requests
import json

# Replace with your SRGSSR-Audio Consumer Key and Secret
CONSUMER_KEY = "xkKmCXCUHYrQJzSj57LlR25W9iA9pUws"
CONSUMER_SECRET = "xKU1KrO3FpPGUZ5J"

# Encode credentials
auth_token = base64.b64encode(f"{CONSUMER_KEY}:{CONSUMER_SECRET}".encode()).decode()

# Get access token
auth_headers = {
    "Authorization": f"Basic {auth_token}",
    "Cache-Control": "no-cache",
    "Content-Length": "0"
}
auth_url = "https://api.srgssr.ch/oauth/v1/accesstoken?grant_type=client_credentials"
auth_response = requests.post(auth_url, headers=auth_headers)
access_data = auth_response.json()

access_token = access_data["access_token"]
print("✅ Access token received.", access_token)

# Query a podcast
podcast_name = "Echo der Zeit"
query_url = "https://api.srgssr.ch/audiometadata/v2/audios/search"
query_headers = {
    "Authorization": f"Bearer {access_token}",
    "accept": "application/json"
}
query_params = {
    "bu": "srf",
    "q": podcast_name,
    "pageSize": 5
}
response = requests.get(query_url, headers=query_headers, params=query_params)

# Print metadata
if response.status_code == 200:
    episodes = response.json().get("searchResultListMedia", [])
    for ep in episodes:
        print(f"🎧 {ep['title']} | {ep.get('date')} | Downloadable: {ep['downloadAvailable']}")
else:
    print(f"❌ Failed to fetch podcast: {response.status_code}\n{response.text}")
