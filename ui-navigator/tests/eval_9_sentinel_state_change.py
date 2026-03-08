"""EVAL 9: Sentinel detects state changes — fill=SAME_SCREEN, login click=NEW_SCREEN."""
import asyncio
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.sandbox.session_manager import SessionManager
from src.agents.sentinel import compare_states, StateVerdict


async def main():
    session = SessionManager(platform="web", headless=False)
    await session.start()
    await session.goto("https://www.saucedemo.com")

    # --- Test 1: Fill username → should be SAME_SCREEN ---
    print("=== Test 1: Fill username field ===")
    screenshot_before = await session.screenshot()

    await session.execute_code(
        'await page.fill("#user-name", "standard_user")\n'
        'await page.wait_for_timeout(500)'
    )

    screenshot_after = await session.screenshot()

    result1 = await compare_states(screenshot_before, screenshot_after)
    print(f"Verdict: {result1['verdict']}")
    print(f"Reason: {result1['reason']}")

    test1_pass = result1["verdict"] == StateVerdict.SAME_SCREEN

    # --- Test 2: Fill password → should be SAME_SCREEN ---
    print("\n=== Test 2: Fill password field ===")
    screenshot_before = await session.screenshot()

    await session.execute_code(
        'await page.fill("#password", "secret_sauce")\n'
        'await page.wait_for_timeout(500)'
    )

    screenshot_after = await session.screenshot()

    result2 = await compare_states(screenshot_before, screenshot_after)
    print(f"Verdict: {result2['verdict']}")
    print(f"Reason: {result2['reason']}")

    test2_pass = result2["verdict"] == StateVerdict.SAME_SCREEN

    # --- Test 3: Click Login → should be NEW_SCREEN ---
    print("\n=== Test 3: Click Login button ===")
    screenshot_before = await session.screenshot()

    await session.execute_code(
        'await page.click("#login-button")\n'
        'await page.wait_for_timeout(1000)'
    )

    screenshot_after = await session.screenshot()

    result3 = await compare_states(screenshot_before, screenshot_after)
    print(f"Verdict: {result3['verdict']}")
    print(f"Reason: {result3['reason']}")

    test3_pass = result3["verdict"] == StateVerdict.NEW_SCREEN

    # --- Results ---
    await session.close()

    print("\n=== RESULTS ===")
    print(f"Test 1 (fill → SAME_SCREEN): {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (fill → SAME_SCREEN): {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (login → NEW_SCREEN): {'PASS' if test3_pass else 'FAIL'}")

    if test1_pass and test2_pass and test3_pass:
        print("\nEVAL 9 PASSED — Sentinel correctly detects state changes")
    else:
        print("\nEVAL 9 FAILED")


if __name__ == "__main__":
    asyncio.run(main())
