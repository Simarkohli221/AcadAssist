import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import Depends, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import auth  # noqa: E402
from web_server import app as production_app  # noqa: E402


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["AUTH_DATABASE_PATH"] = str(Path(self.temp_dir.name) / "auth.sqlite3")
        os.environ["AUTH_SMTP_HOST"] = "smtp.test.invalid"
        os.environ["AUTH_SMTP_FROM"] = "accounts@spit.ac.in"
        os.environ["AUTH_LOCAL_MODE"] = "false"
        auth.initialize_auth_db()
        with auth._attempts_lock:
            auth._attempts.clear()

        self.app = FastAPI()
        self.app.add_exception_handler(RequestValidationError, auth.sanitized_validation_error)
        self.app.include_router(auth.router)

        @self.app.get("/api/protected")
        def protected(user=Depends(auth.get_current_user)):
            return {"email": user["email"]}

        self.client = TestClient(self.app, client=("127.0.0.1", 50000))
        self.production_client = TestClient(production_app)

    def tearDown(self):
        self.client.close()
        self.production_client.close()
        self.temp_dir.cleanup()
        os.environ.pop("AUTH_DATABASE_PATH", None)
        os.environ.pop("AUTH_SMTP_HOST", None)
        os.environ.pop("AUTH_SMTP_FROM", None)
        os.environ.pop("AUTH_LOCAL_MODE", None)

    def _register_and_verify(self, email="student123@spit.ac.in"):
        payload = {
            "full_name": "Test Student",
            "email": email,
            "password": "SecurePass123!",
            "confirm_password": "SecurePass123!",
        }
        with patch("auth._send_verification_email") as send_email:
            response = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(response.status_code, 202)
        token = send_email.call_args.args[2]
        verification = self.client.post("/api/auth/verify", json={"token": token})
        self.assertEqual(verification.status_code, 200)
        return payload

    def test_exact_spit_email_domain_and_case_normalization(self):
        self.assertEqual(auth.normalize_college_email("Nishant@SPIT.AC.IN"), "nishant@spit.ac.in")
        for email in (
            "nishant@gmail.com",
            "student@yahoo.com",
            "user@fake-spit.ac.in",
            "user@spit.ac.in.example.com",
            "user@spit.ac.in.evil.com",
        ):
            with self.subTest(email=email), self.assertRaises(ValueError):
                auth.normalize_college_email(email)

    def test_registration_rejects_non_spit_email_and_weak_password(self):
        payload = {
            "full_name": "Test Student",
            "email": "student@gmail.com",
            "password": "SecurePass123!",
            "confirm_password": "SecurePass123!",
        }
        response = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("@spit.ac.in", response.json()["detail"])

        payload["email"] = "student@spit.ac.in"
        payload["password"] = "short"
        payload["confirm_password"] = "short"
        response = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertNotIn('"input"', response.text)

    def test_duplicate_registration_is_rejected_case_insensitively(self):
        self._register_and_verify("Student123@SPIT.AC.IN")
        payload = {
            "full_name": "Another Student",
            "email": "student123@spit.ac.in",
            "password": "SecurePass123!",
            "confirm_password": "SecurePass123!",
        }
        with patch("auth._send_verification_email"):
            response = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(response.status_code, 409)

    def test_unverified_user_cannot_login(self):
        payload = {
            "full_name": "Test Student",
            "email": "student@spit.ac.in",
            "password": "SecurePass123!",
            "confirm_password": "SecurePass123!",
        }
        with patch("auth._send_verification_email"):
            self.client.post("/api/auth/register", json=payload)
        response = self.client.post("/api/auth/login", json={"email": payload["email"], "password": payload["password"]})
        self.assertEqual(response.status_code, 403)

    def test_local_mode_creates_login_ready_account_without_smtp(self):
        os.environ["AUTH_LOCAL_MODE"] = "true"
        os.environ.pop("AUTH_SMTP_HOST", None)
        os.environ.pop("AUTH_SMTP_FROM", None)
        payload = {
            "full_name": "Local Student",
            "email": "localstudent@spit.ac.in",
            "password": "SecurePass123!",
            "confirm_password": "SecurePass123!",
        }
        with patch("auth._send_verification_email") as send_email:
            response = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(response.status_code, 202)
        self.assertIn("not verified", response.json()["message"])
        send_email.assert_not_called()

        login = self.client.post("/api/auth/login", json={"email": payload["email"], "password": payload["password"]})
        self.assertEqual(login.status_code, 200)
        self.assertIn("acadassist_session", self.client.cookies)

    def test_local_mode_rejects_non_loopback_registration(self):
        os.environ["AUTH_LOCAL_MODE"] = "true"
        os.environ.pop("AUTH_SMTP_HOST", None)
        os.environ.pop("AUTH_SMTP_FROM", None)
        remote_client = TestClient(self.app, client=("203.0.113.10", 50000))
        response = remote_client.post(
            "/api/auth/register",
            json={
                "full_name": "Remote Student",
                "email": "remote@spit.ac.in",
                "password": "SecurePass123!",
                "confirm_password": "SecurePass123!",
            },
        )
        self.assertEqual(response.status_code, 403)
        remote_client.close()

    def test_wrong_password_is_generic_and_correct_login_sets_session(self):
        payload = self._register_and_verify()
        wrong = self.client.post("/api/auth/login", json={"email": payload["email"], "password": "WrongPass123!"})
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(wrong.json()["detail"], "Email or password is incorrect.")

        login = self.client.post("/api/auth/login", json={"email": payload["email"], "password": payload["password"]})
        self.assertEqual(login.status_code, 200)
        self.assertIn("acadassist_session", self.client.cookies)
        self.assertEqual(self.client.get("/api/auth/me").json()["user"]["email"], payload["email"])

        logout = self.client.post("/api/auth/logout")
        self.assertEqual(logout.status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_unauthenticated_protected_route_is_rejected(self):
        response = self.client.get("/api/protected")
        self.assertEqual(response.status_code, 401)

    def test_unauthenticated_real_chat_api_is_rejected(self):
        response = self.production_client.post("/api/chat", json={"question": "What is the attendance rule?"})
        self.assertEqual(response.status_code, 401)

    def test_chat_page_is_not_publicly_available_as_a_static_file(self):
        response = self.production_client.get("/static/index.html")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()