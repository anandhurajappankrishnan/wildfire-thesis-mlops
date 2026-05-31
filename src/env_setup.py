"""Load environment variables from .env at the project root."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_project_env() -> Path:
    """Load .env once and return the project root directory."""
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    return root
