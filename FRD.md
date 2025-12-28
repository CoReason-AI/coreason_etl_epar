# Functional Requirements Document (FRD): coreason_etl_epar

* **Repository:** coreason-source-ema-epar
* **Protocol:** P-IP-001 (Clean Room)
* **Framework:** dlt (Ingestion) + polars (Transform) + pydantic (Validation)

## 1. Phase 1: Ingestion Strategy (The Source)

### 1.1 Source Definitions
We adhere to the "Offline First" principle to ensure robustness.

* **Primary Source:** EMA Medicines for Human Use (Excel Index).
    * **Endpoint:** https://www.ema.europa.eu/sites/default/files/Medicines_output_european_public_assessment_reports.xlsx (Dynamic verification required).
    * **Refresh Strategy:** Daily Snapshot.
* **Secondary Source (Enrichment):** EMA SPOR OMS (Bulk Export).
    * **Endpoint:** https://spor-net.ema.europa.eu/oms-api/v1/organisations/export (Zip/XML).
    * **Strategy:** Ingest the full organization list as a separate dlt resource. Do not call the API per row.

### 1.2 Ingestion Logic (dlt)
* **Resource A: epar_index**
    * **Loader:** Pandas/Polars Excel Reader $\to$ dlt.
    * **Filter:** Explicitly FILTER rows where Category == 'Human'. Drop 'Veterinary'.
    * **Validation:** Rows failing schema validation must be routed to _quarantine.
* **Resource B: spor_organisations_master**
    * **Loader:** XML Streamer $\to$ dlt.
    * **Filter:** If possible, limit to roles "Marketing Authorisation Holder" to reduce volume.

### 1.3 Validation Schema (Pydantic)
Fields specific to "Refused" drugs must be Optional to prevents false negatives.

```python
from pydantic import BaseModel, HttpUrl, validator
from typing import Optional, Literal
from datetime import datetime

class EPARSourceRow(BaseModel):
    category: Literal["Human"] # Strict Filter
    product_number: str # Primary Key (e.g., EMEA/H/C/001234)
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
    revision_date: Optional[datetime]
    url: str

    @validator('product_number')
    def validate_ema_format(cls, v):
        if not v.startswith('EMEA/'):
            raise ValueError('Invalid EMA Product Number format')
        return v
```

## 2. Phase 2: The Medallion Pipeline

### Layer 1: Bronze (The Lake)
* **Objective:** Lossless storage of the raw source snapshots.
* **Storage:** PostgreSQL JSONB or S3 Parquet.
* **Schema:** source_id (Product Number), ingestion_ts, _dlt_load_id, raw_payload, source_file_hash.
* **Write Disposition:** Replace (We overwrite Bronze daily; history is built in Silver).

### Layer 2: Silver (The Refinery)
* **Engine:** polars (LazyFrames).
* **The "Dual ID" Mandate:**
    * **source_id:** Product Number (e.g., EMEA/H/C/001234).
    * **coreason_id:** UUID5(NAMESPACE_EMA, source_id).

#### Transformation Logic 1: The Time Machine (SCD Type 2)
Since the source is a daily snapshot, Silver must construct the history.
* **Input:** Today's Snapshot (Bronze).
* **Comparison:** Join with Silver_Current on source_id.
* **Detect Changes:** Calculate row_hash = MD5(columns_of_interest).
    * If source_id exists AND row_hash differs $\to$ Update.
    * If source_id is new $\to$ Insert.
    * If source_id is missing in source but present in Silver $\to$ Close/Delete (or mark as "Vanished").
* **Execution:**
    * Close old records: Update valid_to = ingestion_ts and is_current = False.
    * Insert new records: valid_from = ingestion_ts, valid_to = NULL, is_current = True.

#### Transformation Logic 2: Entity Resolution & Cleaning
* **Procedure Grouping (The "Family" Link):**
    * **Constraint:** Extract the "Base ID" to group variants.
    * **Logic:** Regex extract EMEA/H/C/(\d+) from product_number. Store as base_procedure_id.
* **Substance Normalization:**
    * **Logic:** active_substance.str.split(by=["/", "+"]). Trim whitespace. Store as Array.
* **ATC Code Explosion:**
    * **Logic:** atc_code.str.split(by=[";", ","]). Validate format (L7 standard). Store as Array.
* **Organization Enrichment (Offline Join):**
    * **Logic:** Silver_EPAR LEFT JOIN Silver_SPOR_Orgs ON fuzzy_match(mah_name, org_name).
    * **Threshold:** Jaro-Winkler distance > 0.90.
* **Status Standardization (Granular Enum):**
    * Authorised $\to$ APPROVED
    * Conditional $\to$ CONDITIONAL_APPROVAL
    * Exceptional Circumstances $\to$ EXCEPTIONAL_CIRCUMSTANCES
    * Refused $\to$ REJECTED
    * Withdrawn $\to$ WITHDRAWN
    * Suspended $\to$ SUSPENDED

### Layer 3: Gold (The Product Schema)
* **Objective:** Queryable History & Analytics.

**Table: dim_medicine (Immutable Entity Attributes)**
* coreason_id (PK)
* medicine_name
* base_procedure_id (Grouping Key)
* brand_name
* is_biosimilar (Boolean)
* is_generic (Boolean)
* is_orphan (Boolean)
* ema_product_url

**Table: fact_regulatory_history (SCD Type 2 Timeline)**
* history_id (PK)
* coreason_id (FK)
* status (Enum)
* valid_from (Timestamp)
* valid_to (Timestamp)
* is_current (Boolean)
* spor_mah_id (Organization ID from SPOR)

**Table: bridge_medicine_features (Search Index)**
* coreason_id (FK)
* feature_type (Enum: 'ATC_CODE', 'SUBSTANCE', 'THERAPEUTIC_AREA')
* feature_value (Normalized String)

## 3. Non-Functional Requirements (NFRs)

### 3.1 Performance & Reliability
* **Memory Safety:** polars streaming mode must be enabled for the Offline Join with SPOR data.
* **Resilience:** The pipeline must handle the scenario where the SPOR bulk export is unavailable (use previous day's cached SPOR master).
* **Idempotency:** Re-running the pipeline on the same day's file must result in zero state changes to the Gold layer.

### 3.2 Observability
* **Logs:** JSON structured logs including ingestion_batch_id.
* **Metrics:**
    * scd_updates_count: Number of records that changed status/data today.
    * spor_match_rate: % of MAHs successfully mapped to a SPOR ID (Alert if < 90%).
    * veterinary_drop_count: Count of rows filtered out.
