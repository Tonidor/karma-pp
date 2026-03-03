CREATE SEQUENCE IF NOT EXISTS exp_group_id_seq;
CREATE SEQUENCE IF NOT EXISTS exp_group_member_id_seq;

CREATE TABLE IF NOT EXISTS exp_group (
  group_id    INTEGER PRIMARY KEY DEFAULT nextval('exp_group_id_seq'),
  label       TEXT NOT NULL,
  created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS exp_group_member (
  id         INTEGER PRIMARY KEY DEFAULT nextval('exp_group_member_id_seq'),
  group_id   INTEGER,
  exp_id     INTEGER REFERENCES experiment,
  UNIQUE (group_id, exp_id)
);
