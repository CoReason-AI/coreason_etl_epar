from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, field_validator


class EPARSourceRow(BaseModel):
    category: Literal["Human"]  # Strict Filter
    product_number: str  # Primary Key (e.g., EMEA/H/C/001234)
    medicine_name: str
    marketing_authorisation_holder: str
    active_substance: str

    # Optional fields (May be missing in Refusals/Withdrawals)
    therapeutic_area: Optional[str] = None
    atc_code: Optional[str] = None

    # Business Flags (Source often uses "Yes"/"No" or Boolean)
    generic: Optional[bool] = False
    biosimilar: Optional[bool] = False
    orphan: Optional[bool] = False
    conditional_approval: Optional[bool] = False
    exceptional_circumstances: Optional[bool] = False

    authorisation_status: str
    revision_date: Optional[datetime] = None
    url: str

    @field_validator("product_number")
    @classmethod
    def validate_ema_format(cls, v: str) -> str:
        if not v.startswith("EMEA/"):
            raise ValueError("Invalid EMA Product Number format")
        return v
