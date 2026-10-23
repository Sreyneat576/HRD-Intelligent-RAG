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


def require_service_token(request: Request, authorization: str | None = Header(default=None)) -> None:
    """Reject the request (401) unless it carries `Authorization: Bearer <SERVICE_TOKEN>`."""
    expected = request.app.state.settings.service_token
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token.encode(), expected.encode()):
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
