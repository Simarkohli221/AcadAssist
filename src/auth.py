"""College-only account registration, verification, and cookie sessions."""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import logging
import os
import re
import secrets
import smtplib
import sqlite3
import ssl
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger("raka.auth")
router = APIRouter()

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = APP_ROOT / "data" / "auth.sqlite3"
EMAIL_PATTERN = re.compile(r"^([A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+)@([A-Za-z0-9.-]+)$")
SESSION_COOKIE = "acadassist_session"
SESSION_TTL_DAYS = 14
PASSWORD_SCRYPT_N = 1 << 14
PASSWORD_SCRYPT_R = 8
PASSWORD_SCRYPT_P = 1
PASSWORD_DKLEN = 64

_attempts: dict[str, list[float]] = {}
_attempts_lock = threading.Lock()


def normalize_college_email(value: str) -> str:
    email = value.strip()
    match = EMAIL_PATTERN.fullmatch(email)
    if not match:
        raise ValueError("Enter a valid SPIT email address ending in @spit.ac.in.")
    local, domain = match.groups()
    if (
        domain.lower() != "spit.ac.in"
        or len(local) > 64
        or len(email) > 254
        or local.startswith(".")
        or local.endswith(".")
        or ".." in local
    ):
        raise ValueError("Only an official @spit.ac.in email address is allowed.")
    return email.lower()


def _database_path() -> Path:
    return Path(os.getenv("AUTH_DATABASE_PATH", str(DEFAULT_DATABASE))).expanduser()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_auth_db() -> None:
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                is_verified INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS verification_tokens (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                expires_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON sessions(expires_at);
            """
        )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=PASSWORD_SCRYPT_N,
        r=PASSWORD_SCRYPT_R,
        p=PASSWORD_SCRYPT_P,
        dklen=PASSWORD_DKLEN,
    )
    return f"scrypt${PASSWORD_SCRYPT_N}${PASSWORD_SCRYPT_R}${PASSWORD_SCRYPT_P}${salt.hex()}${digest.hex()}"


_DUMMY_PASSWORD_HASH = _hash_password(secrets.token_urlsafe(24))


async def sanitized_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt_hex, digest_hex = encoded.split("$")
        if algorithm != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(bytes.fromhex(digest_hex)),
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def _verification_configured() -> bool:
    return bool(os.getenv("AUTH_SMTP_HOST") and os.getenv("AUTH_SMTP_FROM"))


def _send_verification_email(email: str, full_name: str, token: str) -> None:
    host = os.environ["AUTH_SMTP_HOST"]
    sender = os.environ["AUTH_SMTP_FROM"]
    port = int(os.getenv("AUTH_SMTP_PORT", "587"))
    base_url = os.getenv("AUTH_PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    verify_url = f"{base_url}/auth?verification={token}"
    message = EmailMessage()
    message["Subject"] = "Verify your AcadAssist account"
    message["From"] = sender
    message["To"] = email
    message.set_content(
        f"Hello {full_name},\n\n"
        f"Verify your SPIT email address to activate your AcadAssist account:\n{verify_url}\n\n"
        "This link expires in 30 minutes. If you did not request this account, ignore this email."
    )

    if os.getenv("AUTH_SMTP_USE_SSL", "false").lower() == "true":
        smtp_context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=smtp_context, timeout=15) as server:
            _smtp_login(server)
            server.send_message(message)
    else:
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls(context=ssl.create_default_context())
            _smtp_login(server)
            server.send_message(message)


def _smtp_login(server: smtplib.SMTP) -> None:
    username = os.getenv("AUTH_SMTP_USERNAME")
    password = os.getenv("AUTH_SMTP_PASSWORD")
    if username and password:
        server.login(username, password)


class RegistrationRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=12, max_length=128)
    confirm_password: str = Field(min_length=12, max_length=128)

    @model_validator(mode="after")
    def validate_registration(self) -> "RegistrationRequest":
        self.full_name = self.full_name.strip()
        if len(self.full_name) < 2 or any(ord(char) < 32 for char in self.full_name):
            raise ValueError("Enter a valid full name.")
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match.")
        if not (
            re.search(r"[a-z]", self.password)
            and re.search(r"[A-Z]", self.password)
            and re.search(r"\d", self.password)
            and re.search(r"[^A-Za-z0-9]", self.password)
        ):
            raise ValueError("Use at least 12 characters with uppercase, lowercase, a number, and a symbol.")
        return self


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)


def _record_failed_login(keys: list[str]) -> None:
    now = time.monotonic()
    with _attempts_lock:
        for key in keys:
            recent = [attempt for attempt in _attempts.get(key, []) if now - attempt < 900]
            recent.append(now)
            _attempts[key] = recent


def _check_login_rate(key: str) -> None:
    now = time.monotonic()
    with _attempts_lock:
        recent = [attempt for attempt in _attempts.get(key, []) if now - attempt < 900]
        _attempts[key] = recent
        if len(recent) >= 5:
            raise HTTPException(status_code=429, detail="Too many login attempts. Try again in 15 minutes.")


def _check_login_rates(keys: list[str]) -> None:
    for key in keys:
        _check_login_rate(key)


def _clear_login_attempts(keys: list[str]) -> None:
    with _attempts_lock:
        for key in keys:
            _attempts.pop(key, None)


def _issue_session(user_id: int, response: Response) -> None:
    token = secrets.token_urlsafe(32)
    now = _now()
    expiry = now + timedelta(days=SESSION_TTL_DAYS)
    with _connect() as connection:
        connection.execute(
            "INSERT INTO sessions(token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (_token_hash(token), user_id, expiry.isoformat(), now.isoformat()),
        )
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_TTL_DAYS * 24 * 60 * 60,
        httponly=True,
        secure=os.getenv("AUTH_COOKIE_SECURE", "false").lower() == "true",
        samesite="strict",
        path="/",
    )


def get_optional_user(request: Request) -> dict[str, object] | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    with _connect() as connection:
        row = connection.execute(
            """SELECT users.id, users.full_name, users.email
               FROM sessions JOIN users ON users.id = sessions.user_id
               WHERE sessions.token_hash = ? AND sessions.expires_at > ? AND users.is_verified = 1""",
            (_token_hash(token), _now().isoformat()),
        ).fetchone()
    if row is None:
        return None
    return {"id": row["id"], "full_name": row["full_name"], "email": row["email"]}


def get_current_user(request: Request) -> dict[str, object]:
    user = get_optional_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Please sign in to continue.")
    return user


@router.post("/api/auth/register", status_code=202)
def register(payload: RegistrationRequest, request: Request) -> dict[str, str]:
    try:
        email = normalize_college_email(payload.email)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    local_mode = os.getenv("AUTH_LOCAL_MODE", "false").lower() == "true"
    client_host = request.client.host if request.client else ""
    try:
        is_loopback = ipaddress.ip_address(client_host).is_loopback
    except ValueError:
        is_loopback = False
    if local_mode and not is_loopback:
        raise HTTPException(status_code=403, detail="Local account mode is only available from this machine.")
    if not local_mode and not _verification_configured():
        raise HTTPException(status_code=503, detail="Email verification is not configured on this server.")

    now = _now()
    verification_token = None if local_mode else secrets.token_urlsafe(32)
    try:
        with _connect() as connection:
            cursor = connection.execute(
                """INSERT INTO users(full_name, email, password_hash, is_verified, created_at, updated_at)
                         VALUES (?, ?, ?, ?, ?, ?)""",
                (payload.full_name, email, _hash_password(payload.password), int(local_mode), now.isoformat(), now.isoformat()),
            )
            user_id = cursor.lastrowid
            if verification_token is not None:
                connection.execute(
                    "INSERT INTO verification_tokens(token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                    (_token_hash(verification_token), user_id, (now + timedelta(minutes=30)).isoformat()),
                )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="An account with this email already exists.") from exc

    if verification_token is not None:
        try:
            _send_verification_email(email, payload.full_name, verification_token)
        except Exception as exc:
            logger.warning("Could not send account verification email (%s)", type(exc).__name__)
            with _connect() as connection:
                connection.execute("DELETE FROM users WHERE id = ? AND is_verified = 0", (user_id,))
            raise HTTPException(status_code=503, detail="We could not send the verification email. Please try again later.") from exc
        return {"message": "Check your SPIT inbox for a link to verify your account."}
    return {"message": "Local account created. Email ownership was not verified; sign in to continue."}


@router.post("/api/auth/verify")
def verify_email(payload: dict[str, str]) -> dict[str, str]:
    token = payload.get("token", "")
    if not token or len(token) > 200:
        raise HTTPException(status_code=400, detail="This verification link is invalid or expired.")
    now = _now().isoformat()
    with _connect() as connection:
        row = connection.execute(
            "SELECT user_id FROM verification_tokens WHERE token_hash = ? AND expires_at > ?",
            (_token_hash(token), now),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=400, detail="This verification link is invalid or expired.")
        connection.execute("UPDATE users SET is_verified = 1, updated_at = ? WHERE id = ?", (now, row["user_id"]))
        connection.execute("DELETE FROM verification_tokens WHERE user_id = ?", (row["user_id"],))
    return {"message": "Your email is verified. You can now sign in."}


@router.post("/api/auth/login")
def login(payload: LoginRequest, request: Request, response: Response) -> dict[str, object]:
    try:
        email = normalize_college_email(payload.email)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    client_host = request.client.host if request.client else "unknown"
    rate_keys = [f"ip:{client_host}", f"email:{email}"]
    _check_login_rates(rate_keys)

    with _connect() as connection:
        row = connection.execute(
            "SELECT id, full_name, password_hash, is_verified FROM users WHERE email = ?", (email,)
        ).fetchone()
    password_hash = row["password_hash"] if row is not None else _DUMMY_PASSWORD_HASH
    if not _verify_password(payload.password, password_hash):
        _record_failed_login(rate_keys)
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    if not row["is_verified"]:
        raise HTTPException(status_code=403, detail="Verify your SPIT email before signing in.")

    _clear_login_attempts(rate_keys)
    _issue_session(row["id"], response)
    return {"user": {"full_name": row["full_name"], "email": email}}


@router.get("/api/auth/me")
def current_user(user: dict[str, object] = Depends(get_current_user)) -> dict[str, object]:
    return {"user": user}


@router.post("/api/auth/logout")
def logout(request: Request, response: Response) -> dict[str, str]:
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        with _connect() as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
    response.delete_cookie(
        SESSION_COOKIE,
        path="/",
        httponly=True,
        secure=os.getenv("AUTH_COOKIE_SECURE", "false").lower() == "true",
        samesite="strict",
    )
    return {"message": "Signed out."}