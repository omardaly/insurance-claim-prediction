"""Sign-in and registration screen, and the sign-out control."""
import streamlit as st
from src import auth
import ui


def require_login():
    """Returns the signed-in user, or shows the login screen and stops the page."""
    auth.init_db()
    user = st.session_state.get("user")
    if user:
        role, status = auth.current_state(user["username"])       # re-check: could have been disabled meanwhile
        if status == "active":
            user["role"] = role
            return user
        st.session_state.pop("user", None)
        st.warning("Your session ended because the account is no longer active.")

    ui.show(ui.hero("Claim frequency model", "Sign in to continue"))
    if not auth.has_admin():
        st.error("No administrator exists yet. Create one from the command line: "
                 "`python scripts/admin_tool.py create`")
        st.stop()

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        tab_in, tab_reg = st.tabs(["Sign in", "Create account"])
        with tab_in:
            with st.form("login"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                go = st.form_submit_button("Sign in")
            if go:
                found, msg = auth.login(username, password)
                if found:
                    st.session_state["user"] = found
                    st.rerun()
                st.error(msg)
        with tab_reg:
            with st.form("register"):
                full_name = st.text_input("Full name")
                new_user = st.text_input("Username", help="3 to 32 characters: lowercase letters, digits, . - _")
                p1 = st.text_input("Password", type="password",
                                   help="At least 10 characters, with a letter and a digit")
                p2 = st.text_input("Repeat password", type="password")
                go2 = st.form_submit_button("Create account")
            if go2:
                if p1 != p2:
                    st.error("The passwords do not match.")
                else:
                    ok, msg = auth.register(new_user, full_name, p1)
                    (st.success if ok else st.error)(msg)
    st.stop()


def sidebar_user(user):
    role = "Administrator" if user["role"] == "admin" else "Expert"
    ui.show(f'<div class="userbox"><div class="uname">{user["full_name"]}</div>'
            f'<div class="urole">{role} · {user["username"]}</div></div>', sidebar=True)
    if st.sidebar.button("Sign out"):
        st.session_state.pop("user", None)
        st.rerun()