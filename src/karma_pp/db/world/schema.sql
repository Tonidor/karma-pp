CREATE TABLE IF NOT EXISTS world (
  world_hash  TEXT PRIMARY KEY,           -- SHA-256
  json        TEXT NOT NULL,              -- canonical JSON
  created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
