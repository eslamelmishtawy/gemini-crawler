from src.sandbox.playwright_session import PlaywrightSession


class SessionManager:
    """Unified session interface. Routes to Playwright (web) or Appium (mobile)."""

    def __init__(self, platform: str = "web", headless: bool = False):
        self.platform = platform
        if platform == "web":
            self._backend = PlaywrightSession(headless=headless)
        else:
            raise NotImplementedError(f"Platform '{platform}' not yet supported. Appium coming soon.")

    async def start(self) -> None:
        await self._backend.start()

    async def goto(self, url: str) -> None:
        await self._backend.goto(url)

    async def screenshot(self) -> bytes:
        return await self._backend.screenshot()

    async def get_page_source(self) -> str:
        return await self._backend.get_page_source()

    async def get_current_url(self) -> str:
        """Return the current page URL (web) or activity name (mobile)."""
        return await self._backend.get_current_url()

    async def go_back(self) -> None:
        """Navigate back (browser back for web, device back for mobile)."""
        await self._backend.go_back()

    async def execute_code(self, code: str) -> dict:
        return await self._backend.execute_code(code)

    async def close(self) -> None:
        await self._backend.close()
