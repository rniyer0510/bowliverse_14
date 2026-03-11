#!/usr/bin/env python3
import argparse
import secrets
import string
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common.auth import hash_password
from app.persistence.models import User
from app.persistence.session import SessionLocal


def generate_password(length: int = 12) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def reset_password(username: str, new_password: str) -> None:
    db = SessionLocal()
    try:
        normalized = username.strip().lower()
        user = db.query(User).filter(User.username == normalized).first()
        if not user:
            raise SystemExit(f"User not found: {normalized}")

        user.password_hash = hash_password(new_password)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset an ActionLab user's password by username.",
    )
    parser.add_argument("username", help="Username to reset")
    parser.add_argument(
        "--password",
        help="Explicit new password. If omitted, a temporary password is generated.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=12,
        help="Generated password length when --password is omitted (default: 12).",
    )
    args = parser.parse_args()

    if args.password:
        new_password = args.password
    else:
        if args.length < 8:
            raise SystemExit("Password length must be at least 8.")
        new_password = generate_password(args.length)

    reset_password(args.username, new_password)

    print(f"Username: {args.username.strip().lower()}")
    print(f"New password: {new_password}")


if __name__ == "__main__":
    main()
