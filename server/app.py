"""Server entry point for OpenEnv compatibility — re-exports the FastAPI app."""

import sys
import os

# Ensure the project root is on sys.path so `app` and `env` are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uvicorn

from app import app  # noqa: F401

__all__ = ["app"]


def main() -> None:
    """Run the FastAPI server via uvicorn."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
