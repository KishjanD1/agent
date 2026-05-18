import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

def list_gemini_models():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        return
        
    print(f"Connecting to Gemini API using key: {api_key[:8]}...{api_key[-4:]}")
    client = genai.Client(api_key=api_key)
    
    try:
        models = client.models.list()
        print("\nAvailable models on this API Key:")
        print("-" * 50)
        for m in models:
            # Safely print model name attribute only
            print(f"- {m.name}")
        print("-" * 50)
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_gemini_models()
