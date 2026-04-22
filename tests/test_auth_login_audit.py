import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from app.auth_routes import LoginAudit, login


def _query_chain(first_return=None):
    query = MagicMock()
    query.filter.return_value = query
    query.first.return_value = first_return
    return query


class LoginAuditTests(unittest.TestCase):
    def test_successful_login_writes_audit_row(self):
        user = SimpleNamespace(
            user_id="user-1",
            account_id="account-1",
            username="khan",
            password_hash="hashed",
        )
        db = MagicMock()
        db.query.return_value = _query_chain(first_return=user)
        db.new = set()
        db.dirty = set()
        db.deleted = set()
        request = SimpleNamespace(
            client=SimpleNamespace(host="49.207.151.27"),
            headers={"user-agent": "ActionLab/1.0 Android"},
        )

        with patch("app.auth_routes.verify_password",
            return_value=True,
        ), patch(
            "app.auth_routes.create_access_token", return_value="token-123"
        ):
            result = login({"username": "khan", "password": "secret"}, request=request, db=db)

        self.assertEqual(result, {"access_token": "token-123"})
        audit = db.add.call_args.args[0]
        self.assertIsInstance(audit, LoginAudit)
        self.assertEqual(audit.username, "khan")
        self.assertEqual(audit.user_id, "user-1")
        self.assertEqual(audit.account_id, "account-1")
        self.assertEqual(audit.ip_address, "49.207.151.27")
        self.assertEqual(audit.device, "ActionLab/1.0 Android")
        self.assertTrue(audit.success)
        self.assertIsNone(audit.failure_reason)
        db.commit.assert_called_once()
        db.close.assert_not_called()

    def test_invalid_login_writes_failed_audit_row(self):
        user = SimpleNamespace(
            user_id="user-1",
            account_id="account-1",
            username="khan",
            password_hash="hashed",
        )
        db = MagicMock()
        db.query.return_value = _query_chain(first_return=user)
        db.new = set()
        db.dirty = set()
        db.deleted = set()
        request = SimpleNamespace(
            client=SimpleNamespace(host="49.207.151.27"),
            headers={"user-agent": "ActionLab/1.0 Android"},
        )

        with patch("app.auth_routes.verify_password",
            return_value=False,
        ):
            with self.assertRaises(HTTPException) as context:
                login({"username": "khan", "password": "wrong"}, request=request, db=db)

        self.assertEqual(context.exception.status_code, 401)
        audit = db.add.call_args.args[0]
        self.assertIsInstance(audit, LoginAudit)
        self.assertEqual(audit.username, "khan")
        self.assertFalse(audit.success)
        self.assertEqual(audit.failure_reason, "invalid_credentials")
        db.commit.assert_called_once()
        db.close.assert_not_called()

    def test_audit_failure_does_not_rollback_request_session(self):
        user = SimpleNamespace(
            user_id="user-1",
            account_id="account-1",
            username="khan",
            password_hash="hashed",
        )
        db = MagicMock()
        db.query.return_value = _query_chain(first_return=user)
        db.new = set()
        db.dirty = set()
        db.deleted = set()
        request = SimpleNamespace(
            client=SimpleNamespace(host="49.207.151.27"),
            headers={"user-agent": "ActionLab/1.0 Android"},
        )
        db.commit.side_effect = RuntimeError("audit unavailable")

        with patch("app.auth_routes.verify_password",
            return_value=True,
        ), patch(
            "app.auth_routes.create_access_token",
            return_value="token-123",
        ):
            result = login({"username": "khan", "password": "secret"}, request=request, db=db)

        self.assertEqual(result, {"access_token": "token-123"})
        self.assertEqual(db.commit.call_count, 2)
        self.assertEqual(db.rollback.call_count, 2)
        db.close.assert_not_called()

    def test_audit_uses_isolated_session_when_request_session_is_dirty(self):
        user = SimpleNamespace(
            user_id="user-1",
            account_id="account-1",
            username="khan",
            password_hash="hashed",
        )
        db = MagicMock()
        db.query.return_value = _query_chain(first_return=user)
        db.new = {object()}
        db.dirty = set()
        db.deleted = set()
        request = SimpleNamespace(
            client=SimpleNamespace(host="49.207.151.27"),
            headers={"user-agent": "ActionLab/1.0 Android"},
        )
        audit_db = MagicMock()

        with patch("app.auth_routes.SessionLocal", return_value=audit_db), patch(
            "app.auth_routes.verify_password",
            return_value=True,
        ), patch(
            "app.auth_routes.create_access_token", return_value="token-123"
        ):
            result = login({"username": "khan", "password": "secret"}, request=request, db=db)

        self.assertEqual(result, {"access_token": "token-123"})
        db.commit.assert_not_called()
        audit_db.commit.assert_called_once()
        audit_db.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
