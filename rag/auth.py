import os

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


class Auth:

    def __init__(self):

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")

        if not url:
            raise ValueError(
                "SUPABASE_URL is missing."
            )

        if not key:
            raise ValueError(
                "SUPABASE_ANON_KEY is missing."
            )

        self.client = create_client(
            url,
            key
        )

    # =========================================
    # RESTORE SESSION
    # =========================================

    def restore_session(
        self,
        access_token,
        refresh_token
    ):

        self.client.auth.set_session(
            access_token,
            refresh_token
        )

    # =========================================
    # SIGN UP
    # =========================================

    def sign_up(
        self,
        email,
        password
    ):

        response = self.client.auth.sign_up(
            {
                "email": email,
                "password": password
            }
        )

        return response

    # =========================================
    # LOGIN
    # =========================================

    def sign_in(
        self,
        email,
        password
    ):

        response = (
            self.client
            .auth
            .sign_in_with_password(
                {
                    "email": email,
                    "password": password
                }
            )
        )

        return response

    # =========================================
    # LOGOUT
    # =========================================

    def sign_out(self):

        self.client.auth.sign_out()

    # =========================================
    # CURRENT USER
    # =========================================

    def get_user(self):

        response = self.client.auth.get_user()

        return response.user