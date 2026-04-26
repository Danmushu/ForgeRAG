"""
Auth configuration.

ForgeRAG's auth is minimal and self-contained:

    auth:
      enabled: true           # false/absent = no auth (honour 127.0.0.1 binding)
      mode: db                # "db" = bearer tokens + password sessions stored
                              #         in Postgres (default)
                              # "forwarded" = trust an upstream OAuth proxy's
                              #         X-Forwarded-User header
      # ── mode=db knobs ──
      initial_password: forgerag   # applied at auto-bootstrap; first login
                                   # must change. Change via yaml only affects
                                   # fresh bootstraps, not existing admins.
      session_cookie_name: forgerag_session
      session_cookie_secure: true  # set false only for http://localhost dev
      password_change_revokes_other_sessions: true

      # ── mode=forwarded knobs ──
      forwarded_user_header: X-Forwarded-User

Tokens + sessions are in DB tables (``auth_users``, ``auth_tokens``,
``auth_sessions``); there's no yaml-token list. See
``docs/auth.md`` for the full operator guide.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AuthConfig(BaseModel):
    enabled: bool = False
    mode: Literal["db", "forwarded"] = "db"

    # --- mode=db ---
    initial_password: str = "forgerag"
    session_cookie_name: str = "forgerag_session"
    session_cookie_secure: bool = True
    password_change_revokes_other_sessions: bool = True

    # --- mode=forwarded ---
    forwarded_user_header: str = "X-Forwarded-User"

    # Paths that bypass auth even when enabled (health probes, static assets).
    # Matched as path prefix. /api/v1/auth/login obviously also bypasses
    # (hardcoded) so users can actually log in.
    public_paths: list[str] = Field(
        default_factory=lambda: ["/api/v1/health"],
        description="URL path prefixes that bypass auth (health probes etc.)",
    )
