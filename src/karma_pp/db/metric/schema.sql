CREATE TABLE IF NOT EXISTS metric (
  exp_id       INTEGER NOT NULL REFERENCES experiment,
  step         INTEGER,
  metric_name  TEXT    NOT NULL,
  metric_value TEXT    NOT NULL,
  recorded_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (exp_id, step, metric_name)
);
CREATE INDEX IF NOT EXISTS idx_metric_metric_name ON metric(metric_name);
