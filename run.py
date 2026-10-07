"""Run the API server locally: `python run.py`."""

import logging

import uvicorn

from app.config import Settings

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    uvicorn.run("app.main:create_app", factory=True, host=settings.host, port=settings.port)
