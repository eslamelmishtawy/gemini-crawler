import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # GCP
    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

    # Firestore
    FIRESTORE_DATABASE = os.getenv("FIRESTORE_DATABASE", "(default)")

    # Cloud Storage
    GCS_BUCKET = os.getenv("GCS_BUCKET", "ui-navigator-artifacts")

    # Target
    TARGET_URL = os.getenv("TARGET_URL", "https://www.saucedemo.com")
    TARGET_PLATFORM = os.getenv("TARGET_PLATFORM", "web")

    # Agent
    DEDUP_REPRESENTATIVES = int(os.getenv("DEDUP_REPRESENTATIVES", "2"))
    ACTION_RETRY_LIMIT = int(os.getenv("ACTION_RETRY_LIMIT", "3"))
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "50"))

    # Models — three tiers, each node picks the tier it needs
    # PRO:   reserved for future use if needed
    # FLASH: complex tasks — vision, reasoning, code gen, exhaustive analysis
    # LITE:  simple structured tasks — parsing, matching, routing
    GEMINI_PRO_MODEL = os.getenv("GEMINI_PRO_MODEL", "gemini-3.1-pro-preview")
    GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-3-flash-preview")
    GEMINI_LITE_MODEL = os.getenv("GEMINI_LITE_MODEL", "gemini-3.1-flash-lite-preview")


config = Config()
