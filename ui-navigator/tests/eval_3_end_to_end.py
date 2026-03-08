"""EVAL 3: End-to-end - take screenshot, send to Gemini, get element description."""
import asyncio
import base64
import os
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from google import genai
from src.sandbox.session_manager import SessionManager


async def main():
    # 1. Take screenshot
    session = SessionManager(platform="web", headless=True)
    await session.start()
    await session.goto("https://www.saucedemo.com")
    screenshot = await session.screenshot()
    await session.close()
    print(f"Screenshot captured: {len(screenshot)} bytes")

    # 2. Send to Gemini Pro with vision
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": screenshot_b64}},
                    {"text": "Identify all interactive elements on this screen. For each element, provide: type (button, input, link), label, and a suggested CSS selector."},
                ],
            }
        ],
    )

    print("\nGemini Response:")
    print(response.text)
    print("\nEVAL 3 PASSED")


if __name__ == "__main__":
    asyncio.run(main())
