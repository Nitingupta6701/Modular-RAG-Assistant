from google import genai
from config.settings import GOOGLE_API_KEY

client = genai.Client(api_key=GOOGLE_API_KEY)

try:
    print("Checking available Gemini models...\n")

    for model in client.models.list():
        if "flash" in model.name.lower():
            print(model.name)

except Exception as e:
    print("Error:", type(e).__name__)
    print(e)