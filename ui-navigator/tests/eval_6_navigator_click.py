"""EVAL 6: Navigator generates Playwright code to click Login button and executes it."""
import asyncio
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.sandbox.session_manager import SessionManager
from src.agents.page_analyzer import analyze_screen
from src.agents.navigator import navigate_action
from src.models.action import ActionType


async def main():
    session = SessionManager(platform="web", headless=False)
    await session.start()
    await session.goto("https://www.saucedemo.com")

    # 1. Analyze the page to get real ActionDocs
    screenshot = await session.screenshot()
    page_source = await session.get_page_source()
    print("Running Page Analyzer...")
    result = await analyze_screen(screenshot, page_source)
    actions = result["actions"]

    print(f"Found {len(actions)} actions")

    # 2. Find the Login button action
    login_action = None
    for a in actions:
        if a.type == ActionType.CLICK and "login" in (
            a.element_info.text or ""
        ).lower():
            login_action = a
            break

    if not login_action:
        print("Could not find Login button in analyzed elements!")
        print("Available click actions:")
        for a in actions:
            if a.type == ActionType.CLICK:
                print(f"  - {a.element_info.text} | {a.element_info.selector}")
        await session.close()
        return

    print(f"\nTarget: {login_action.element_info.text}")
    print(f"Selector: {login_action.element_info.selector}")
    print(f"Bounds: {login_action.element_info.bounds}")

    # 3. Navigate (execute the click)
    print("\nGenerating and executing Playwright code...")
    nav_result = await navigate_action(session, login_action)

    print(f"\nSuccess: {nav_result['success']}")
    print(f"Attempts: {nav_result['attempts']}")
    print(f"Generated code:\n{nav_result['generated_code']}")
    print(f"Action status: {nav_result['action'].status.value}")

    # 4. Validate — after clicking Login with no creds, we should see an error
    screenshot_after = nav_result["screenshot_after"]
    print(f"\nScreenshot after: {len(screenshot_after)} bytes")

    await session.close()

    if nav_result["success"]:
        print("\nEVAL 6 PASSED")
    else:
        print("\nEVAL 6 FAILED")


if __name__ == "__main__":
    asyncio.run(main())
