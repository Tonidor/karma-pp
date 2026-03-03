CREATE SEQUENCE IF NOT EXISTS experiment_id_seq;

CREATE TABLE IF NOT EXISTS experiment (
    exp_id          INTEGER PRIMARY KEY DEFAULT nextval('experiment_id_seq'),
    world_hash      TEXT NOT NULL REFERENCES world(world_hash),
    mechanism_hash  TEXT NOT NULL REFERENCES mechanism(mechanism_hash),
    population_hash TEXT NOT NULL REFERENCES population(population_hash),
    seed            INTEGER NOT NULL,
    n_steps         INTEGER NOT NULL,
    git_commit      TEXT NOT NULL,
    name            TEXT NOT NULL,
    comment         TEXT,
    runtime_s       REAL,
    started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at        TIMESTAMP,
    status          TEXT CHECK (status IN ('pending','running','finished','failed'))
);
