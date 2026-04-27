import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.persistence.notifications import (
    create_analysis_completed_notification,
    register_device,
)


class NotificationFoundationTests(unittest.TestCase):
    def test_create_analysis_completed_notification_skips_when_no_active_devices(self):
        db = MagicMock()
        player_query = MagicMock()
        player_query.filter.return_value.first.return_value = SimpleNamespace(
            player_name="Raj"
        )
        device_query = MagicMock()
        device_query.filter.return_value.count.return_value = 0
        db.query.side_effect = [player_query, device_query]

        event = create_analysis_completed_notification(
            account_id="11111111-1111-1111-1111-111111111111",
            result={
                "run_id": "run-1",
                "input": {"player_id": "22222222-2222-2222-2222-222222222222"},
            },
            db=db,
        )

        self.assertEqual(event.status, "SKIPPED_NO_DEVICE")
        self.assertEqual(event.active_device_count, 0)
        self.assertEqual(
            event.payload_json["player_name"],
            "Raj",
        )

    def test_create_analysis_completed_notification_queues_when_devices_are_active(self):
        db = MagicMock()
        player_query = MagicMock()
        player_query.filter.return_value.first.return_value = None
        device_query = MagicMock()
        device_query.filter.return_value.count.return_value = 2
        db.query.side_effect = [player_query, device_query]

        event = create_analysis_completed_notification(
            account_id="11111111-1111-1111-1111-111111111111",
            result={
                "run_id": "run-2",
                "input": {"player_id": "22222222-2222-2222-2222-222222222222"},
            },
            db=db,
        )

        self.assertEqual(event.status, "PENDING")
        self.assertEqual(event.active_device_count, 2)
        self.assertIsNone(event.payload_json["player_name"])

    def test_register_device_reactivates_existing_token(self):
        existing = SimpleNamespace(
            account_id=None,
            platform="unknown",
            push_provider="unknown",
            push_token="device-token",
            device_label=None,
            app_version=None,
            locale=None,
            timezone=None,
            is_active=False,
            last_seen_at=None,
            updated_at=None,
        )
        db = MagicMock()
        query = MagicMock()
        query.filter.return_value.first.return_value = existing
        db.query.return_value = query

        row = register_device(
            account_id="11111111-1111-1111-1111-111111111111",
            platform="iOS",
            push_provider="APNS",
            push_token="device-token",
            device_label="Coach iPhone",
            app_version="1.2.3",
            locale="en-IN",
            timezone="Asia/Kolkata",
            db=db,
        )

        self.assertIs(row, existing)
        self.assertTrue(row.is_active)
        self.assertEqual(row.platform, "ios")
        self.assertEqual(row.push_provider, "apns")
        self.assertEqual(row.device_label, "Coach iPhone")
        db.add.assert_not_called()
        db.flush.assert_called_once()


if __name__ == "__main__":
    unittest.main()
