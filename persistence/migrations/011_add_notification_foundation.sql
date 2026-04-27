CREATE TABLE IF NOT EXISTS device_registration (
    device_registration_id UUID PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES account(account_id),
    platform TEXT NOT NULL
        CHECK (platform IN ('ios','android','web','unknown')),
    push_provider TEXT NOT NULL
        CHECK (push_provider IN ('fcm','apns','unknown')),
    push_token TEXT NOT NULL UNIQUE,
    device_label TEXT NULL,
    app_version TEXT NULL,
    locale TEXT NULL,
    timezone TEXT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_device_registration_account_id
    ON device_registration (account_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS notification_event (
    notification_event_id UUID PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES account(account_id),
    event_type TEXT NOT NULL
        CHECK (event_type IN ('analysis_completed','analysis_failed','profile_country_required')),
    status TEXT NOT NULL
        CHECK (status IN ('PENDING','SKIPPED_NO_DEVICE','SENT','FAILED')),
    active_device_count INTEGER NOT NULL DEFAULT 0,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_detail TEXT NULL,
    sent_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_notification_event_account_id
    ON notification_event (account_id, created_at DESC);
