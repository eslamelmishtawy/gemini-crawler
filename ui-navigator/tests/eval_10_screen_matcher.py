"""EVAL 10: Screen Matcher identifies known screens and detects new ones."""
import asyncio
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.agents.sentinel import match_screen


async def main():
    # Simulate a registry of known screens (as the orchestrator would maintain)
    known_screens = {
        "scr_001": "Login page with username field, password field, and Login button",
        "scr_002": "Product inventory page with grid of items, cart icon, and hamburger menu",
    }

    # --- Test 1: Description that matches scr_001 ---
    print("=== Test 1: Should match scr_001 (login page) ===")
    result1 = await match_screen(
        new_description="Sign-in page with username input, password input, and a Login button",
        known_screens=known_screens,
    )
    print(f"Match: {result1['match']}, Screen ID: {result1['screen_id']}")
    test1_pass = result1["match"] and result1["screen_id"] == "scr_001"

    # --- Test 2: Description that matches scr_002 ---
    print("\n=== Test 2: Should match scr_002 (inventory page) ===")
    result2 = await match_screen(
        new_description="Product catalog showing items in a grid layout with shopping cart and menu",
        known_screens=known_screens,
    )
    print(f"Match: {result2['match']}, Screen ID: {result2['screen_id']}")
    test2_pass = result2["match"] and result2["screen_id"] == "scr_002"

    # --- Test 3: Completely new screen — no match ---
    print("\n=== Test 3: Should NOT match (checkout page) ===")
    result3 = await match_screen(
        new_description="Checkout page with shipping address form, payment method, and Place Order button",
        known_screens=known_screens,
    )
    print(f"Match: {result3['match']}, Screen ID: {result3['screen_id']}")
    test3_pass = not result3["match"]

    # --- Test 4: Empty registry — always no match ---
    print("\n=== Test 4: Empty registry — should NOT match ===")
    result4 = await match_screen(
        new_description="Login page with username and password",
        known_screens={},
    )
    print(f"Match: {result4['match']}, Screen ID: {result4['screen_id']}")
    test4_pass = not result4["match"]

    # --- Results ---
    print("\n=== RESULTS ===")
    print(f"Test 1 (login matches scr_001):   {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (inventory matches scr_002): {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (checkout = no match):      {'PASS' if test3_pass else 'FAIL'}")
    print(f"Test 4 (empty registry = no match): {'PASS' if test4_pass else 'FAIL'}")

    if test1_pass and test2_pass and test3_pass and test4_pass:
        print("\nEVAL 10 PASSED — Screen Matcher correctly identifies known and new screens")
    else:
        print("\nEVAL 10 FAILED")


if __name__ == "__main__":
    asyncio.run(main())
