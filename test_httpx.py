import os

import httpx

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
key = os.getenv("GEMINI_API_KEY")
headers = {"x-goog-api-key": key, "content-type": "application/json"}
data = {"contents":[{"parts":[{"text":"Hello"}]}]}
print("Sending request via httpx...")
try:
    with httpx.Client(timeout=10.0) as client:
        res = client.post(url, headers=headers, json=data)
    print("Status:", res.status_code)
    print("Text:", res.text)
except Exception as e:
    print("Exception:", e)
