CREATE TABLE IF NOT EXISTS population (
  population_hash  TEXT PRIMARY KEY,           -- SHA-256 over canonical members JSON
  created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE SEQUENCE IF NOT EXISTS population_member_id_seq;

CREATE TABLE IF NOT EXISTS population_member (
  id              INTEGER PRIMARY KEY DEFAULT nextval('population_member_id_seq'),
  population_hash TEXT NOT NULL REFERENCES population(population_hash),
  model_id        TEXT NOT NULL,
  agent_hash      TEXT NOT NULL REFERENCES agent(agent_hash),
  n_agents        INTEGER NOT NULL,
  weight          REAL NOT NULL,
  UNIQUE (population_hash, model_id)
);

CREATE INDEX IF NOT EXISTS idx_population_member_population_hash ON population_member(population_hash);
CREATE INDEX IF NOT EXISTS idx_population_member_agent_hash ON population_member(agent_hash);

