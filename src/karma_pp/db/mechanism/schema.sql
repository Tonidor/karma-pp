CREATE TABLE IF NOT EXISTS mechanism (
  mechanism_hash  TEXT PRIMARY KEY,           -- SHA-256 over canonical JSON
  json            TEXT NOT NULL,              -- canonical JSON
  created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

