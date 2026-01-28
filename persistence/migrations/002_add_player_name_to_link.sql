-- ActionLab Persistence Migration 002
-- Add player_name to account_player_link (relationship label)
ALTER TABLE account_player_link
ADD COLUMN IF NOT EXISTS player_name VARCHAR;

CREATE INDEX IF NOT EXISTS idx_apl_account_playername
ON account_player_link (account_id, player_name);
