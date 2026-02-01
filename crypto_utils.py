import os
from typing import Optional

def _get_fernet():
    key = os.getenv("APP_FERNET_KEY", "").strip()
    if not key:
        return None
    from cryptography.fernet import Fernet
    return Fernet(key.encode("utf-8"))

def get_secret(env_name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Reads an environment variable.
    - If it starts with 'enc:' -> decrypt using APP_FERNET_KEY
    - Otherwise -> return as-is
    """
    val = os.getenv(env_name)
    if val is None:
        if required:
            raise RuntimeError(f"Missing required environment variable: {env_name}")
        return default

    val = val.strip()
    if val.lower().startswith("enc:"):
        f = _get_fernet()
        if f is None:
            raise RuntimeError(
                f"{env_name} is encrypted but APP_FERNET_KEY is missing. "
                "Set APP_FERNET_KEY in environment variables."
            )
        token = val[4:].encode("utf-8")
        return f.decrypt(token).decode("utf-8")

    return val or default



