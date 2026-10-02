"""Order retention and exchange-side cancellation heartbeat settings."""
from decimal import Decimal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DeadmanSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = True
    timeout_seconds: int = Field(default=30, ge=10, le=120)
    renew_seconds: float = Field(default=10, ge=1)

    @model_validator(mode="after")
    def timing(self):
        if self.renew_seconds * 3 > self.timeout_seconds:
            raise ValueError("Heartbeat renewal must be at most one third of timeout")
        return self


class ExecutionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    retain_quotes: bool = True
    max_quote_age_seconds: float = Field(default=60, ge=15, le=300)
    reprice_bps: Decimal = Field(default=Decimal(2), ge=0, le=20)
    size_tolerance: Decimal = Field(default=Decimal("0.1"), ge=0, le=Decimal("0.25"))
