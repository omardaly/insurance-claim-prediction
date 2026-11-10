"""User accounts: registration, login, roles, approval. Python standard library only.

Passwords are stored as salted scrypt hashes, never in clear text. Every storage call lives in this module, so
moving the users table to PostgreSQL later only changes the connection code here, not the app.
"""
import hashlib
import hmac
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "auth" / "users.db"
USERNAME_RE = re.compile(r"^[a-z0-9._-]{3,32}$")
MIN_PASSWORD = 10
MAX_ATTEMPTS = 5
LOCK_MINUTES = 15
SCRYPT = dict(n=2 ** 14, r=8, p=1, dklen=32)
GENERIC_FAIL = "Wrong username or password."
ROLES = ("admin", "expert")
STATUSES = ("pending", "active", "disabled")


# ------------------------------------------------------------------ storage
@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with _conn() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            full_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('admin', 'expert')),
            status TEXT NOT NULL CHECK (status IN ('pending', 'active', 'disabled')),
            failed_attempts INTEGER NOT NULL DEFAULT 0,
            locked_until TEXT,
            created_at TEXT NOT NULL,
            last_login TEXT)""")


def _now():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------ passwords
def hash_password(password):
    salt = os.urandom(16)
    h = hashlib.scrypt(password.encode("utf-8"), salt=salt, **SCRYPT)
    return f"scrypt${SCRYPT['n']}${SCRYPT['r']}${SCRYPT['p']}${salt.hex()}${h.hex()}"


def verify_password(password, stored):
    try:
        _, n, r, p, salt, h = stored.split("$")
        got = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt),
                             n=int(n), r=int(r), p=int(p), dklen=len(h) // 2)
        return hmac.compare_digest(got.hex(), h)
    except Exception:
        return False


_DUMMY_HASH = hash_password("dummy-password-for-timing")      # unknown users cost the same time as known ones


def password_problems(password, username=""):
    out = []
    if len(password) < MIN_PASSWORD:
        out.append(f"at least {MIN_PASSWORD} characters")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        out.append("at least one letter and one digit")
    if username and username.lower() in password.lower():
        out.append("must not contain the username")
    return out


# ------------------------------------------------------------------ accounts
def has_admin():
    with _conn() as con:
        return con.execute("SELECT 1 FROM users WHERE role='admin' AND status='active'").fetchone() is not None


def register(username, full_name, password):
    """Self-registration: always an 'expert' in 'pending' status. An admin has to approve it."""
    username, full_name = (username or "").strip().lower(), (full_name or "").strip()
    if not USERNAME_RE.match(username):
        return False, "Username: 3 to 32 characters, lowercase letters, digits, dot, dash or underscore."
    if not 2 <= len(full_name) <= 80:
        return False, "Please enter your full name."
    bad = password_problems(password, username)
    if bad:
        return False, "Password needs: " + "; ".join(bad) + "."
    try:
        with _conn() as con:
            con.execute("INSERT INTO users (username, full_name, password_hash, role, status, created_at) "
                        "VALUES (?, ?, ?, 'expert', 'pending', ?)",
                        (username, full_name, hash_password(password), _iso(_now())))
    except sqlite3.IntegrityError:
        return False, "This username is already taken."
    return True, "Account created. An administrator has to approve it before you can sign in."


def create_admin(username, full_name, password):
    """Used by the command-line tool only, never by the registration form."""
    username = username.strip().lower()
    if not USERNAME_RE.match(username):
        raise ValueError("invalid username")
    bad = password_problems(password, username)
    if bad:
        raise ValueError("Password needs: " + "; ".join(bad))
    with _conn() as con:
        con.execute("INSERT INTO users (username, full_name, password_hash, role, status, created_at) "
                    "VALUES (?, ?, ?, 'admin', 'active', ?)",
                    (username, full_name.strip(), hash_password(password), _iso(_now())))


def reset_password(username, new_password):
    username = username.strip().lower()
    bad = password_problems(new_password, username)
    if bad:
        raise ValueError("Password needs: " + "; ".join(bad))
    with _conn() as con:
        n = con.execute("UPDATE users SET password_hash=?, failed_attempts=0, locked_until=NULL WHERE username=?",
                        (hash_password(new_password), username)).rowcount
    if n == 0:
        raise ValueError("no such user")


def login(username, password):
    """Returns (user_dict, '') on success, or (None, message). The password is checked before the account
    status is revealed, so status is only shown to someone who knows the password."""
    username = (username or "").strip().lower()
    with _conn() as con:
        row = con.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if row is None:
            verify_password(password or "", _DUMMY_HASH)
            return None, GENERIC_FAIL
        if row["locked_until"] and row["locked_until"] > _iso(_now()):
            return None, "Too many failed attempts. Try again later."
        if not verify_password(password or "", row["password_hash"]):
            fails = row["failed_attempts"] + 1
            lock = _iso(_now() + timedelta(minutes=LOCK_MINUTES)) if fails >= MAX_ATTEMPTS else None
            con.execute("UPDATE users SET failed_attempts=?, locked_until=? WHERE id=?",
                        (0 if lock else fails, lock, row["id"]))
            return None, GENERIC_FAIL
        if row["status"] == "pending":
            return None, "Your account is waiting for approval by an administrator."
        if row["status"] == "disabled":
            return None, "This account is disabled. Contact an administrator."
        con.execute("UPDATE users SET failed_attempts=0, locked_until=NULL, last_login=? WHERE id=?",
                    (_iso(_now()), row["id"]))
        return {"username": row["username"], "full_name": row["full_name"], "role": row["role"]}, ""


def current_state(username):
    """Fresh role and status from the database (used to re-check a session)."""
    with _conn() as con:
        r = con.execute("SELECT role, status FROM users WHERE username=?", (username,)).fetchone()
    return (r["role"], r["status"]) if r else (None, None)


# ------------------------------------------------------------------ administration
def list_users():
    with _conn() as con:
        rows = con.execute("SELECT username, full_name, role, status, created_at, last_login "
                           "FROM users ORDER BY (status='pending') DESC, username").fetchall()
    return [dict(r) for r in rows]


def _active_admins(con):
    return con.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND status='active'").fetchone()[0]


def set_status(username, status, acting):
    if status not in STATUSES:
        return False, "Unknown status."
    with _conn() as con:
        row = con.execute("SELECT role, status FROM users WHERE username=?", (username,)).fetchone()
        if row is None:
            return False, "No such user."
        if username == acting and status != "active":
            return False, "You cannot disable your own account."
        if row["role"] == "admin" and row["status"] == "active" and status != "active" and _active_admins(con) <= 1:
            return False, "This is the last active administrator."
        con.execute("UPDATE users SET status=? WHERE username=?", (status, username))
    return True, f"{username}: status is now '{status}'."


def set_role(username, role, acting):
    if role not in ROLES:
        return False, "Unknown role."
    with _conn() as con:
        row = con.execute("SELECT role, status FROM users WHERE username=?", (username,)).fetchone()
        if row is None:
            return False, "No such user."
        if row["role"] == "admin" and role != "admin" and row["status"] == "active" and _active_admins(con) <= 1:
            return False, "This is the last active administrator."
        if username == acting and role != "admin":
            return False, "You cannot remove your own administrator role."
        con.execute("UPDATE users SET role=? WHERE username=?", (role, username))
    return True, f"{username}: role is now '{role}'."