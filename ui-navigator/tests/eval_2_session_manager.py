"""EVAL 2: Verify Session Manager opens browser, navigates, and takes screenshot."""
import asyncio
import sys
sys.path.insert(0, ".")

from src.sandbox.session_manager import SessionManager


async def main():
    session = SessionManager(platform="web", headless=True)

    print("Starting browser...")
    await session.start()

    print("Navigating to saucedemo.com...")
    await session.goto("https://www.saucedemo.com")

    print("Taking screenshot...")
    screenshot = await session.screenshot()
    print(f"Screenshot size: {len(screenshot)} bytes")

    # Save to verify visually
    with open("tests/eval_2_screenshot.png", "wb") as f:
        f.write(screenshot)
    print("Saved to tests/eval_2_screenshot.png")

    print("Getting page source...")
    source = await session.get_page_source()
    print(f"Page source length: {len(source)} chars")

    await session.close()
    print("EVAL 2 PASSED")


if __name__ == "__main__":
    asyncio.run(main())
