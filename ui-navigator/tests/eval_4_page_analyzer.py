"""EVAL 4: Page Analyzer outputs elements with selectors, coordinates, AND assertions."""
import asyncio
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.sandbox.session_manager import SessionManager
from src.agents.page_analyzer import analyze_screen


async def main():
    # 1. Get screenshot + page source
    session = SessionManager(platform="web", headless=True)
    await session.start()
    await session.goto("https://www.saucedemo.com")
    screenshot = await session.screenshot()
    page_source = await session.get_page_source()
    await session.close()
    print(f"Screenshot: {len(screenshot)} bytes, Page source: {len(page_source)} chars")

    # 2. Run Page Analyzer
    print("\nRunning Page Analyzer (DOM + Vision + Merge)...")
    result = await analyze_screen(screenshot, page_source)

    screen = result["screen"]
    actions = result["actions"]

    # 3. Print results
    print(f"\nScreen ID: {screen.screen_id}")
    print(f"Description: {result['screen_description']}")
    print(f"Resolution: {screen.resolution}")
    print(f"Elements found: {len(actions)}")

    assertions = [a for a in actions if a.expected_outcome == "assertion"]
    interactive = [a for a in actions if a.expected_outcome != "assertion"]

    print(f"\n--- Assertion Targets ({len(assertions)}) ---")
    for a in assertions:
        print(f"  [{a.priority.value}] \"{a.element_info.text}\"")
        print(f"    Selector: {a.element_info.selector}")
        print(f"    Bounds: {a.element_info.bounds}")

    print(f"\n--- Interactive Elements ({len(interactive)}) ---")
    for a in interactive:
        print(f"  [{a.type.value}] {a.element_info.text or a.action_id}")
        print(f"    Selector: {a.element_info.selector}")
        print(f"    Bounds: {a.element_info.bounds}")
        print(f"    Visible: {a.element_info.visible}")

    # 4. Validate
    has_selector = sum(1 for a in actions if a.element_info.selector)
    has_bounds = sum(1 for a in actions if a.element_info.bounds)
    has_assertions = len(assertions)

    print(f"\n--- Validation ---")
    print(f"Total elements: {len(actions)}")
    print(f"With CSS selector: {has_selector}")
    print(f"With x,y bounds: {has_bounds}")
    print(f"Assertion targets: {has_assertions}")
    print(f"Model type: {type(screen).__name__}, {type(actions[0]).__name__}")

    if has_selector > 0 and has_bounds > 0 and has_assertions > 0:
        print("\nEVAL 4 PASSED")
    else:
        print("\nEVAL 4 FAILED")


if __name__ == "__main__":
    asyncio.run(main())
