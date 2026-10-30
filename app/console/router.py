"""Developer console: a web UI (static HTML/CSS/JS) for uploading documents, chatting with the RAG
pipeline and inspecting its per-stage timings. Enabled with CONSOLE_ENABLED=true (development only).

Opening /console sets an HttpOnly, SameSite=Strict cookie with a per-process secret; the API accepts
that cookie instead of the service token (see app/core/security.py), so the token never reaches the
browser.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse

from app.core.security import CONSOLE_COOKIE

STATIC_DIR = Path(__file__).parent / "static"

router = APIRouter(include_in_schema=False)


@router.get("/")
async def root(request: Request):
    """Send browsers to the console when it is enabled, otherwise to the API docs."""
    return RedirectResponse("/console" if request.app.state.settings.console_enabled else "/docs")


@router.get("/console", response_class=HTMLResponse)
async def console(request: Request) -> HTMLResponse:
    """Serve the console page and sign the browser in with the console cookie."""
    if not request.app.state.settings.console_enabled:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Console is disabled (set CONSOLE_ENABLED=true)")
    response = HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))
    response.set_cookie(CONSOLE_COOKIE, request.app.state.console_secret, httponly=True, samesite="strict", path="/")
    response.headers["Cache-Control"] = "no-store"
    return response
