CREATE SEQUENCE IF NOT EXISTS artifact_id_seq;

CREATE TABLE IF NOT EXISTS artifact (
  artifact_id INTEGER PRIMARY KEY DEFAULT nextval('artifact_id_seq'),
  exp_id      INTEGER REFERENCES experiment,
  name        TEXT,
  path        TEXT,
  sha256      TEXT,
  created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
