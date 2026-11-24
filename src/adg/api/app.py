"""FastAPI application factory.

Per spec §10.1: FastAPI factory with CORS, lifespan, and exception handlers.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from adg.api.routes import analyze, chat, health, sessions, stats
from adg.core.errors import ADGError
from adg.core.logging import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context for FastAPI app.

    Loads models once at startup.
    """
    log.info("api_startup")
    # ponytail: manager loads on first request via dependency
    yield
    log.info("api_shutdown")


def create_app() -> FastAPI:
    """Create FastAPI application.

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Ask, Don't Guess API",
        description="Training-free ambiguity detection for small LLMs",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # ponytail: allow all for demo
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(ADGError)
    async def adg_error_handler(request: Request, exc: ADGError) -> JSONResponse:
        """Handle ADG errors.

        Args:
            request: Request
            exc: ADG error

        Returns:
            JSON response with error details
        """
        log.error(
            "adg_error",
            error_type=exc.__class__.__name__,
            error_code=exc.code,
            message=str(exc),
        )
        return JSONResponse(
            status_code=exc.http_status_code,
            content={
                "error": exc.code,
                "message": str(exc),
                "hint": exc.hint,
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle general exceptions.

        Args:
            request: Request
            exc: Exception

        Returns:
            JSON response with error
        """
        log.error("unexpected_error", error_type=exc.__class__.__name__, message=str(exc))
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
            },
        )

    # Include routers
    app.include_router(health.router, tags=["health"])
    app.include_router(chat.router, tags=["chat"])
    app.include_router(analyze.router, tags=["analyze"])
    app.include_router(sessions.router, tags=["sessions"])
    app.include_router(stats.router, tags=["stats"])

    return app
