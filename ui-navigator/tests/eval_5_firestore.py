"""EVAL 5: Page Analyzer results written to Firestore and read back."""
import asyncio
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.sandbox.session_manager import SessionManager
from src.agents.page_analyzer import analyze_screen
from src.tools.firestore_tools import (
    create_screen, create_actions_batch, get_screen, get_pending_actions
)


async def main():
    # 1. Capture and analyze
    session = SessionManager(platform="web", headless=True)
    await session.start()
    await session.goto("https://www.saucedemo.com")
    screenshot = await session.screenshot()
    page_source = await session.get_page_source()
    await session.close()

    result = await analyze_screen(screenshot, page_source)
    print(f"Screen: {result['screen_id']}")

    # 2. Write screen to Firestore
    screen_data = {
        "screen_id": result["screen_id"],
        "screen_description": result["screen_description"],
        "url_or_activity": "https://www.saucedemo.com",
        "platform": "web",
        "resolution": result["resolution"],
        "status": "analyzed",
        "actions_summary": {
            "total": len(result["actions"]),
            "pending": len(result["actions"]),
            "completed": 0, "failed": 0, "deduped": 0, "skipped": 0,
        },
    }
    create_screen(screen_data)
    print(f"Screen written to Firestore")

    # 3. Write actions to Firestore
    count = create_actions_batch(result["actions"])
    print(f"{count} actions written to Firestore")

    # 4. Read back and verify
    screen_back = get_screen(result["screen_id"])
    actions_back = get_pending_actions(result["screen_id"])

    print(f"\n--- Read back from Firestore ---")
    print(f"Screen: {screen_back['screen_id']}")
    print(f"Description: {screen_back['screen_description']}")
    print(f"Pending actions: {len(actions_back)}")

    for a in actions_back:
        elem = a.get("element_info", {})
        print(f"  [{a['type']}] selector={elem.get('selector')} bounds={elem.get('bounds')}")

    if screen_back and len(actions_back) > 0:
        print("\nEVAL 5 PASSED")
    else:
        print("\nEVAL 5 FAILED")


if __name__ == "__main__":
    asyncio.run(main())
