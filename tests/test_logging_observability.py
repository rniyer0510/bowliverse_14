import importlib
import logging
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.common.logger import get_logger


class LoggerConfigTests(unittest.TestCase):
    def test_log_level_respects_env(self):
        with patch.dict(os.environ, {"ACTIONLAB_LOG_LEVEL": "DEBUG"}, clear=False):
            logger = get_logger("actionlab.tests.logger")
            self.assertEqual(logger.level, logging.DEBUG)


class RequestMiddlewareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep imports side-effect-safe for tests.
        os.environ.setdefault("ACTIONLAB_SECRET", "test-secret")
        os.environ["ACTIONLAB_AUTO_CREATE_SCHEMA"] = "false"

        module = importlib.import_module("app.orchestrator.orchestrator")
        cls.client = TestClient(module.app)

    def test_request_id_is_set_on_response(self):
        response = self.client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        request_id = response.headers.get("x-request-id")
        self.assertIsNotNone(request_id)
        self.assertTrue(len(request_id.strip()) > 0)

    def test_request_id_echoes_incoming_header(self):
        expected_request_id = "test-request-id-123"
        response = self.client.get(
            "/openapi.json",
            headers={"X-Request-ID": expected_request_id},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("x-request-id"), expected_request_id)


if __name__ == "__main__":
    unittest.main()
