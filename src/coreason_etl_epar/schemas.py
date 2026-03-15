# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_etl_epar

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


class EPARSourceRow(BaseModel):
    category: Literal["Human"] = Field(description="Strict Filter")
    product_number: str = Field(description="Primary Key (e.g., EMEA/H/C/001234)")
    medicine_name: str
    marketing_authorisation_holder: str
    active_substance: str

    therapeutic_area: str | None = Field(
        default=None, description="Optional fields (May be missing in Refusals/Withdrawals)"
    )
    atc_code: str | None = Field(default=None, description="Optional fields (May be missing in Refusals/Withdrawals)")

    generic: bool | None = Field(default=False, description="Business Flags (Source often uses 'Yes'/'No' or Boolean)")
    biosimilar: bool | None = Field(
        default=False, description="Business Flags (Source often uses 'Yes'/'No' or Boolean)"
    )
    orphan: bool | None = Field(default=False, description="Business Flags (Source often uses 'Yes'/'No' or Boolean)")
    conditional_approval: bool | None = Field(
        default=False, description="Business Flags (Source often uses 'Yes'/'No' or Boolean)"
    )
    exceptional_circumstances: bool | None = Field(
        default=False, description="Business Flags (Source often uses 'Yes'/'No' or Boolean)"
    )

    authorisation_status: str
    revision_date: datetime | None = None
    url: HttpUrl

    @field_validator("product_number")
    @classmethod
    def validate_ema_format(cls, v: str) -> str:
        if not v.startswith("EMEA/"):
            raise ValueError("Invalid EMA Product Number format")
        return v


class SPOROrganisationRow(BaseModel):
    org_id: str = Field(min_length=1, description="SPOR Organization ID")
    org_name: str = Field(min_length=1, description="SPOR Organization Name")
