# chat_gemini.py
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

class GeminiChat:
    """
    Wrapper around Google's Gemini API for generating text responses.
    Can be used with any custom prompt passed from the frontend.
    """
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required to initialize GeminiChat.")
        genai.configure(api_key=api_key)
        # You can switch between 'gemini-1.5-flash-latest' (fast) or 'gemini-1.5-pro-latest' (more powerful)
        self.model = genai.GenerativeModel('models/gemini-1.5-flash-latest')

    def get_response(self, message: str) -> str:
        """
        Sends a custom prompt to Gemini API and returns the response.
        
        Args:
            prompt (str): Instruction or question to send to Gemini.
            
        Returns:
            str: Generated text response from Gemini.
        """
        try:
            time.sleep(0.5)  # small delay for rate limiting safety
            response = self.model.generate_content(message)
            return response.text.strip()
        except Exception as e:
            return f"ERROR: Gemini API failed: {e}"


# Standalone test
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in environment variables")

    gemini = GeminiChat(api_key)
