from playwright.async_api import async_playwright, Page


class PlaywrightSession:
    """Playwright browser backend for the Session Manager."""

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._playwright = None
        self._browser = None
        self._page: Page | None = None

    async def start(self) -> None:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        context = await self._browser.new_context(
            viewport={"width": 1280, "height": 720},
        )
        self._page = await context.new_page()

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Session not started. Call start() first.")
        return self._page

    async def goto(self, url: str) -> None:
        await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)

    async def screenshot(self) -> bytes:
        return await self.page.screenshot()

    async def get_page_source(self) -> str:
        return await self.page.content()

    async def execute_code(self, code: str) -> dict:
        """Run dynamically generated Playwright code against the live page."""
        try:
            wrapped = "async def _run(page):\n"
            for line in code.strip().splitlines():
                stripped = line.strip()
                # Auto-fix: if line has a call (ends with `)`) but no `await`, add it
                if stripped and not stripped.startswith("#") and "(" in stripped and "await " not in stripped:
                    line = line.replace(stripped, f"await {stripped}", 1)
                wrapped += f"    {line}\n"

            exec_globals = {}
            exec(compile(wrapped, "<generated>", "exec"), exec_globals)
            await exec_globals["_run"](self.page)

            screenshot_after = await self.screenshot()
            return {"success": True, "screenshot": screenshot_after}
        except Exception as e:
            screenshot_after = await self.screenshot()
            return {"success": False, "error": str(e), "screenshot": screenshot_after}

    async def get_current_url(self) -> str:
        """Return the current page URL."""
        return self.page.url

    async def go_back(self) -> None:
        """Navigate back in browser history."""
        await self.page.go_back(wait_until="domcontentloaded", timeout=10000)

    async def close(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._page = None
        self._browser = None
        self._playwright = None
