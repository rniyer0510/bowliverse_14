CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS players (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT,
    age INT,
    handedness TEXT,
    bowler_type TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id UUID REFERENCES players(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    fps FLOAT,
    duration_sec FLOAT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id UUID REFERENCES players(id) ON DELETE CASCADE,
    video_id UUID REFERENCES videos(id),

    schema_version TEXT NOT NULL,
    biomechanics_json JSONB NOT NULL,
    clinician_yaml JSONB NOT NULL,

    verdict TEXT,
    risk_summary JSONB,
    positives JSONB,
    strength_focus JSONB,

    created_at TIMESTAMP DEFAULT now()
);
