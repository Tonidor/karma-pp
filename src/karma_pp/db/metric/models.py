from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Metric:
    __table__ = "metric"
    __pk__ = ("exp_id", "step", "metric_name")  # keep if you extend BaseRepo

    exp_id: int
    step: int  # use 0 for “final” metric
    metric_name: str
    metric_value: float
    recorded_at: datetime
