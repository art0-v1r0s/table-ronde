import os

from google import genai

key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=key)
print("Sending request to Google GenAI...")
try:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents='Tell me a story in 300 words.'
    )
    print("Response:", response.text)
except Exception as e:
    print("Exception:", type(e), e)
