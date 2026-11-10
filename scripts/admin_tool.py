"""Command-line tool: create the first administrator, or reset a password.
    python scripts/admin_tool.py create
    python scripts/admin_tool.py reset USERNAME
"""
import getpass
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import auth


def ask_password():
    p1 = getpass.getpass("Password: ")
    if p1 != getpass.getpass("Repeat password: "):
        sys.exit("The passwords do not match.")
    return p1


auth.init_db()
if len(sys.argv) == 2 and sys.argv[1] == "create":
    username = input("Username: ")
    full_name = input("Full name: ")
    try:
        auth.create_admin(username, full_name, ask_password())
    except Exception as e:
        sys.exit(f"Could not create the administrator: {e}")
    print("Administrator created.")
elif len(sys.argv) == 3 and sys.argv[1] == "reset":
    try:
        auth.reset_password(sys.argv[2], ask_password())
    except Exception as e:
        sys.exit(f"Could not reset the password: {e}")
    print("Password reset.")
else:
    sys.exit(__doc__)