"""Main application entrypoint for EXP-07 SLR Engine service."""

from __future__ import annotations

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from exp07.api.routes import router
from exp07.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("exp07")

app = FastAPI(
    title="EXP-07 Selective Layer Residency (SLR) Engine",
    description="Low-memory LLM inference engine demonstrating Selective Layer Residency",
    version="0.7.0",
)

# Narrow CORS configuration for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits Flutter desktop, web dev server, and local curl
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router)


def run():
    """Run uvicorn server bound to loopback by default."""
    logger.info(f"Starting EXP-07 SLR Engine on http://{config.host}:{config.port}")
    uvicorn.run(
        "exp07.main:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()
