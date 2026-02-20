import base64
import hashlib
import hmac
import json
import os
import secrets
import time

SECRET_KEY = os.getenv("GAIA_SECRET_KEY", "dev-secret-change-me")
TOKEN_TTL_SECONDS = int(os.getenv("GAIA_TOKEN_TTL_SECONDS", "604800"))
PBKDF2_ROUNDS = int(os.getenv("GAIA_PBKDF2_ROUNDS", "200000"))
RESET_CODE_TTL_SECONDS = int(os.getenv("GAIA_RESET_CODE_TTL_SECONDS", "900"))


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ROUNDS,
    )
    return f"{base64.b64encode(salt).decode('utf-8')}${base64.b64encode(derived).decode('utf-8')}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_b64, derived_b64 = stored.split("$", 1)
        salt = base64.b64decode(salt_b64.encode("utf-8"))
        expected = base64.b64decode(derived_b64.encode("utf-8"))
    except Exception:
        return False

    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ROUNDS,
    )
    return hmac.compare_digest(expected, derived)


def generate_reset_code() -> str:
    return f"{secrets.randbelow(1000000):06d}"


def hash_reset_code(code: str) -> str:
    return hmac.new(SECRET_KEY.encode("utf-8"), code.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_reset_code(code: str, stored_hash: str) -> bool:
    if not stored_hash:
        return False
    computed = hash_reset_code(code)
    return hmac.compare_digest(computed, stored_hash)


def create_token(user_id: int) -> str:
    header = {"alg": "HS256", "typ": "GAIA"}
    payload = {"sub": user_id, "exp": int(time.time()) + TOKEN_TTL_SECONDS}
    header_b64 = _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    message = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(SECRET_KEY.encode("utf-8"), message, hashlib.sha256).digest()
    signature_b64 = _b64url(signature)
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode_token(token: str):
    parts = token.split(".")
    if len(parts) != 3:
        return None

    header_b64, payload_b64, signature_b64 = parts
    message = f"{header_b64}.{payload_b64}".encode("utf-8")
    expected_sig = hmac.new(SECRET_KEY.encode("utf-8"), message, hashlib.sha256).digest()
    try:
        actual_sig = _b64url_decode(signature_b64)
    except Exception:
        return None

    if not hmac.compare_digest(expected_sig, actual_sig):
        return None

    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception:
        return None

    if payload.get("exp", 0) < int(time.time()):
        return None

    return payload
