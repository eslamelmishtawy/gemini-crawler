"""EVAL 7: Data Provider generates fill values, Navigator executes them."""
import asyncio
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.sandbox.session_manager import SessionManager
from src.agents.page_analyzer import analyze_screen
from src.agents.data_provider import provide_fill_value
from src.agents.navigator import navigate_action
from src.models.action import ActionType, ActionScenario


async def main():
    session = SessionManager(platform="web", headless=False)
    await session.start()
    await session.goto("https://www.saucedemo.com")

    # 1. Analyze page
    screenshot = await session.screenshot()
    page_source = await session.get_page_source()
    print("Running Page Analyzer...")
    result = await analyze_screen(screenshot, page_source)
    actions = result["actions"]
    screen_desc = result["screen_description"]
    print(f"Found {len(actions)} actions")
    print(f"Screen: {screen_desc}")

    # 2. Find positive fills and login click
    fill_actions = [
        a for a in actions
        if a.type == ActionType.FILL
        and a.scenario == ActionScenario.POSITIVE
    ]
    click_actions = [
        a for a in actions
        if a.type == ActionType.CLICK
        and "login" in (a.element_info.text or "").lower()
    ]

    print(f"\nPositive fill actions: {len(fill_actions)}")
    for a in fill_actions:
        print(f"  - {a.element_info.text} | {a.element_info.selector} | fill_value: {a.fill_value}")

    if len(fill_actions) < 2:
        print("Need at least 2 fill actions (username + password)")
        await session.close()
        return

    if not click_actions:
        print("No Login button found")
        await session.close()
        return

    # 3. Data Provider → Navigator for each fill action
    all_success = True
    for i, action in enumerate(fill_actions[:2]):
        action = await provide_fill_value(action, screen_desc)
        print(f"\n--- Fill {i+1}: {action.element_info.text} → \"{action.fill_value}\" ---")
        nav_result = await navigate_action(session, action)
        print(f"Success: {nav_result['success']}")
        print(f"Code:\n{nav_result['generated_code']}")
        if not nav_result["success"]:
            all_success = False

    # Click login
    print(f"\n--- Click: Login ---")
    nav_result = await navigate_action(session, click_actions[0])
    print(f"Success: {nav_result['success']}")
    print(f"Code:\n{nav_result['generated_code']}")
    if not nav_result["success"]:
        all_success = False

    # 5. Check result
    await asyncio.sleep(1)
    final_source = await session.get_page_source()
    logged_in = "inventory" in final_source.lower()
    has_error = "epic-sadface" in final_source.lower()

    print(f"\nLanded on inventory page: {logged_in}")
    print(f"Got login error: {has_error}")

    await session.close()

    # For unknown sites, we expect the fill to execute successfully
    # (even if login fails — that's expected with fake data)
    if all_success:
        print("\nEVAL 7 PASSED — all fills and click executed successfully")
        if has_error:
            print("(Login failed with generated data — expected for fake credentials)")
    else:
        print("\nEVAL 7 FAILED")


if __name__ == "__main__":
    asyncio.run(main())
