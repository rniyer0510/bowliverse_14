import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import HTTPException

from app.auth_routes import delete_account


def _chain_query(*, all_return=None):
    query = MagicMock()
    query.filter.return_value = query
    query.all.return_value = all_return if all_return is not None else []
    query.delete.return_value = 1
    return query


class DeleteAccountTests(unittest.TestCase):
    def test_delete_account_success_cleans_data_and_commits(self):
        db = MagicMock()
        current_user = SimpleNamespace(account_id="acc-1")

        owned_players = [
            SimpleNamespace(player_id="player-1"),
            SimpleNamespace(player_id="player-2"),
        ]
        run_ids = [SimpleNamespace(run_id="run-1")]

        db.query.side_effect = [
            _chain_query(all_return=owned_players),  # owned players
            _chain_query(all_return=run_ids),  # runs by owned players
            _chain_query(),  # event_anchor delete
            _chain_query(),  # biomech_signal delete
            _chain_query(),  # risk_measurement delete
            _chain_query(),  # visual_evidence delete
            _chain_query(),  # analysis_result_raw delete
            _chain_query(),  # analysis_run delete
            _chain_query(),  # account_player_link by player delete
            _chain_query(),  # player delete
            _chain_query(),  # account_player_link by account delete
            _chain_query(),  # user delete
            _chain_query(),  # account delete
        ]

        result = delete_account(current_user=current_user, db=db)

        self.assertEqual(result, {"status": "deleted"})
        db.commit.assert_called_once()
        db.rollback.assert_not_called()

    def test_delete_account_rolls_back_on_error(self):
        db = MagicMock()
        current_user = SimpleNamespace(account_id="acc-1")
        db.query.side_effect = RuntimeError("db error")

        with self.assertRaises(HTTPException) as context:
            delete_account(current_user=current_user, db=db)

        self.assertEqual(context.exception.status_code, 500)
        self.assertEqual(context.exception.detail, "Failed to delete account")
        db.rollback.assert_called_once()
        db.commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
