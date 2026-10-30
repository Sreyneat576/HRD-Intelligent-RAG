"""Service-to-service auth.

Spring (data-management / chat services) is the only caller. It authenticates with a shared
service token and forwards the end user's identity in headers after it has validated the
user's Keycloak JWT itself.
"""

import secrets
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request, status


@dataclass(frozen=True)
class UserContext:
    user_id: str
    role: str | None


CONSOLE_COOKIE = "hrd_console"


def require_service_token(request: Request, authorization: str | None = Header(default=None)) -> None:
    """Reject the request (401) unless it carries `Authorization: Bearer <SERVICE_TOKEN>`, or - when the
    developer console is enabled - the console's session cookie (set by GET /console)."""
    settings = request.app.state.settings
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() == "bearer" and secrets.compare_digest(token.encode(), settings.service_token.encode()):
        return
    cookie = request.cookies.get(CONSOLE_COOKIE)
    secret = getattr(request.app.state, "console_secret", None)
    if settings.console_enabled and cookie and secret and secrets.compare_digest(cookie.encode(), secret.encode()):
        return
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing service token")


def get_user(
    _: None = Depends(require_service_token),
    x_user_id: str | None = Header(default=None),
    x_user_role: str | None = Header(default=None),
) -> UserContext:
    """Service token check plus the end user forwarded by Spring in X-User-Id / X-User-Role."""
    if not x_user_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "X-User-Id header is required")
    return UserContext(user_id=x_user_id, role=x_user_role)
