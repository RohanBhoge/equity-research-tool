import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

print("--- Starting Authentication Test ---")

# 1. Load environment variables from .env file
load_dotenv()

# 2. Get the API key from the environment
api_key = os.getenv("GEMINI_API_KEY")

# 3. Check if the key was loaded
if api_key:
    # Print only the first few and last few characters for security
    print(f"API Key Loaded: {api_key[:5]}...{api_key[-4:]}")
else:
    print("API Key Loaded: None. The key was not found in the environment.")
    print("--- Test Failed ---")
    exit() # Stop the script if no key is found

# 4. Try to use the key to initialize the model
try:
    print("Attempting to initialize the Gemini model...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key
    )
    # Make a simple API call to test the connection and key
    llm.invoke("Hello")
    print("\n✅ SUCCESS: Authentication successful!")
    print("This means your API key is valid and being loaded correctly.")

except Exception as e:
    print(f"\n❌ FAILED: An error occurred.")
    print(f"Error details: {e}")

print("--- Test Finished ---")