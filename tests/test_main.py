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
from unittest.mock import Mock, patch

import polars as pl

from coreason_etl_epar.main import hello_world, run_pipeline
from coreason_etl_epar.schemas import RegulatoryStatusEnum


def test_hello_world() -> None:
    assert hello_world() == "Hello World!"


@patch("coreason_etl_epar.main.dlt.pipeline")
@patch("coreason_etl_epar.main.get_spor_organisations_resource")
@patch("coreason_etl_epar.main.get_epar_index_resource")
def test_run_pipeline_initial_load(mock_epar_res: Mock, mock_spor_res: Mock, mock_dlt_pipeline: Mock) -> None:
    mock_dlt_pipeline.return_value = Mock()
    # Synthetic Bronze Data from Generators
    mock_epar_res.return_value = iter(
        [
            {
                "category": "Human",
                "product_number": "EMEA/H/C/001234",
                "medicine_name": "SuperDrug",
                "marketing_authorisation_holder": "PharmaCorp",
                "active_substance": "Substance X",
                "therapeutic_area": "Area 1",
                "atc_code": "A10BA02",
                "generic": False,
                "biosimilar": True,
                "orphan": False,
                "conditional_approval": False,
                "exceptional_circumstances": False,
                "additional_monitoring": True,
                "authorisation_status": "Authorised",
                "revision_date": "2023-01-01T00:00:00",
                "url": "https://www.ema.europa.eu/en/medicines/human/EPAR/superdrug",
            }
        ]
    )

    mock_spor_res.return_value = iter([{"org_id": "ORG1000", "org_name": "pharmacorp"}])

    ingestion_ts = datetime(2023, 10, 1)

    dim, fact, bridge = run_pipeline(
        epar_url="http://fake-epar",
        spor_url="http://fake-spor",
        ingestion_ts=ingestion_ts,
        current_history=None,
        destination="dummy",
    )

    assert isinstance(dim, pl.DataFrame)
    assert isinstance(fact, pl.DataFrame)
    assert isinstance(bridge, pl.DataFrame)

    assert len(dim) == 1
    assert dim["medicine_name"][0] == "SuperDrug"
    assert dim["is_biosimilar"][0] is True
    assert dim["additional_monitoring"][0] is True

    assert len(fact) == 1
    assert fact["status"][0] == RegulatoryStatusEnum.APPROVED.value
    assert fact["spor_mah_id"][0] == "ORG1000"
    assert fact["valid_from"][0] == ingestion_ts
    assert fact["valid_to"][0] is None
    assert fact["is_current"][0] is True

    # 1 Substance, 1 ATC Code, 1 Therapeutic Area
    assert len(bridge) == 3


@patch("coreason_etl_epar.main.dlt.pipeline")
@patch("coreason_etl_epar.main.get_spor_organisations_resource")
@patch("coreason_etl_epar.main.get_epar_index_resource")
def test_run_pipeline_quarantine_records_ignored(
    mock_epar_res: Mock, mock_spor_res: Mock, mock_dlt_pipeline: Mock
) -> None:
    mock_dlt_pipeline.return_value = Mock()
    import dlt

    mock_epar_res.return_value = iter(
        [
            dlt.mark.with_table_name({"error": "test_error"}, "epar_index_quarantine"),
            {
                "category": "Human",
                "product_number": "EMEA/H/C/001235",
                "medicine_name": "GoodDrug",
                "marketing_authorisation_holder": "GoodCorp",
                "active_substance": "Substance G",
                "therapeutic_area": "Area G",
                "atc_code": "B10BA02",
                "generic": False,
                "biosimilar": False,
                "orphan": False,
                "conditional_approval": False,
                "exceptional_circumstances": False,
                "additional_monitoring": False,
                "authorisation_status": "Authorised",
                "revision_date": "2023-01-01T00:00:00",
                "url": "https://www.ema.europa.eu/en/medicines/human/EPAR/gooddrug",
            },
        ]
    )
    mock_spor_res.return_value = iter([])

    dim, fact, _bridge = run_pipeline(
        epar_url="http://fake-epar",
        spor_url="http://fake-spor",
        ingestion_ts=datetime(2023, 10, 1),
        destination="dummy",
    )

    # We expect only the valid record to be processed
    assert len(dim) == 1
    assert dim["medicine_name"][0] == "GoodDrug"
    assert len(fact) == 1


@patch("coreason_etl_epar.main.dlt.pipeline")
@patch("coreason_etl_epar.main.get_spor_organisations_resource")
@patch("coreason_etl_epar.main.get_epar_index_resource")
def test_run_pipeline_empty_load(mock_epar_res: Mock, mock_spor_res: Mock, mock_dlt_pipeline: Mock) -> None:
    mock_dlt_pipeline.return_value = Mock()
    mock_epar_res.return_value = iter([])
    mock_spor_res.return_value = iter([])

    dim, fact, bridge = run_pipeline(
        epar_url="http://fake", spor_url="http://fake", ingestion_ts=datetime(2023, 1, 1), destination="dummy"
    )

    assert isinstance(dim, pl.DataFrame)
    assert len(dim) == 0
    assert len(fact) == 0
    assert len(bridge) == 0


@patch("coreason_etl_epar.main.dlt.pipeline")
@patch("coreason_etl_epar.main.get_spor_organisations_resource")
@patch("coreason_etl_epar.main.get_epar_index_resource")
def test_run_pipeline_incremental_load(mock_epar_res: Mock, mock_spor_res: Mock, mock_dlt_pipeline: Mock) -> None:
    mock_dlt_pipeline.return_value = Mock()
    import uuid

    from coreason_etl_epar.transform import NAMESPACE_EMA

    # Provide existing history
    coreason_id = str(uuid.uuid5(NAMESPACE_EMA, "EMEA/H/C/001234"))

    current_history = pl.DataFrame(
        {
            "history_id": ["old-hist-1"],
            "coreason_id": [coreason_id],
            "status": [RegulatoryStatusEnum.APPROVED.value],
            "valid_from": [datetime(2023, 9, 1)],
            "valid_to": [None],
            "is_current": [True],
            "spor_mah_id": ["ORG1000"],
            "is_orphan": [False],
        }
    )

    # New snapshot has status change to WITHDRAWN
    mock_epar_res.return_value = iter(
        [
            {
                "category": "Human",
                "product_number": "EMEA/H/C/001234",
                "medicine_name": "SuperDrug",
                "marketing_authorisation_holder": "PharmaCorp",
                "active_substance": "Substance X",
                "atc_code": "A10BA02",
                "therapeutic_area": "Area 1",
                "authorisation_status": "Withdrawn",
                "url": "https://www.ema.europa.eu/en/medicines/human/EPAR/superdrug",
                "generic": False,
                "biosimilar": False,
                "orphan": False,
                "conditional_approval": False,
                "exceptional_circumstances": False,
                "additional_monitoring": False,
            }
        ]
    )

    mock_spor_res.return_value = iter([{"org_id": "ORG1000", "org_name": "pharmacorp"}])

    ingestion_ts = datetime(2023, 10, 1)

    _dim, fact, _bridge = run_pipeline(
        epar_url="http://fake-epar",
        spor_url="http://fake-spor",
        ingestion_ts=ingestion_ts,
        current_history=current_history,
        destination="dummy",
    )

    # Fact should now have 2 rows for this coreason_id:
    # 1. Closed old record (valid_to = ingestion_ts)
    # 2. Open new record (valid_from = ingestion_ts, status = WITHDRAWN)
    assert len(fact) == 2

    # Sort by valid_from to check history
    fact_sorted = fact.sort("valid_from")

    # Check old record
    assert fact_sorted["status"][0] == RegulatoryStatusEnum.APPROVED.value
    assert fact_sorted["is_current"][0] is False
    assert fact_sorted["valid_to"][0] == ingestion_ts
    assert fact_sorted["valid_from"][0] == datetime(2023, 9, 1)

    # Check new record
    assert fact_sorted["status"][1] == RegulatoryStatusEnum.WITHDRAWN.value
    assert fact_sorted["is_current"][1] is True
    assert fact_sorted["valid_to"][1] is None
    assert fact_sorted["valid_from"][1] == ingestion_ts


@patch("coreason_etl_epar.main.dlt.pipeline")
@patch("coreason_etl_epar.main.get_spor_organisations_resource")
@patch("coreason_etl_epar.main.get_epar_index_resource")
def test_run_pipeline_idempotency(mock_epar_res: Mock, mock_spor_res: Mock, mock_dlt_pipeline: Mock) -> None:
    mock_dlt_pipeline.return_value = Mock()
    # 1st run data
    epar_data = [
        {
            "category": "Human",
            "product_number": "EMEA/H/C/001234",
            "medicine_name": "IdempotentDrug",
            "marketing_authorisation_holder": "IdempotentCorp",
            "active_substance": "Substance I",
            "therapeutic_area": "Area I",
            "atc_code": "I10BA02",
            "generic": False,
            "biosimilar": False,
            "orphan": False,
            "conditional_approval": False,
            "exceptional_circumstances": False,
            "additional_monitoring": False,
            "authorisation_status": "Authorised",
            "revision_date": "2023-01-01T00:00:00",
            "url": "https://www.ema.europa.eu/en/medicines/human/EPAR/idempotentdrug",
        }
    ]
    spor_data = [{"org_id": "ORG2000", "org_name": "idempotentcorp"}]

    mock_epar_res.return_value = iter(epar_data)
    mock_spor_res.return_value = iter(spor_data)

    ingestion_ts_1 = datetime(2023, 10, 1)

    _dim_1, fact_1, _bridge_1 = run_pipeline(
        epar_url="http://fake-epar",
        spor_url="http://fake-spor",
        ingestion_ts=ingestion_ts_1,
        current_history=None,
        destination="dummy",
    )

    # Reset iterators for the second run
    mock_epar_res.return_value = iter(epar_data)
    mock_spor_res.return_value = iter(spor_data)

    # 2nd run on the same data but later ingestion_ts
    ingestion_ts_2 = datetime(2023, 10, 2)

    _dim_2, fact_2, _bridge_2 = run_pipeline(
        epar_url="http://fake-epar",
        spor_url="http://fake-spor",
        ingestion_ts=ingestion_ts_2,
        current_history=fact_1,
        destination="dummy",
    )

    # Asserts for strict idempotency
    # 1. Row counts should be exactly the same
    assert len(fact_1) == len(fact_2)

    # 2. No new facts created
    assert len(fact_2) == 1

    # 3. History remains completely unchanged from the first run
    # valid_from and valid_to timestamps should still reflect ingestion_ts_1
    assert fact_2["valid_from"][0] == ingestion_ts_1
    assert fact_2["valid_to"][0] is None
    assert fact_2["is_current"][0] is True


@patch("coreason_etl_epar.main.dlt.pipeline")
@patch("coreason_etl_epar.main.get_spor_organisations_resource")
@patch("coreason_etl_epar.main.get_epar_index_resource")
def test_run_pipeline_complex_idempotency_with_changes(
    mock_epar_res: Mock, mock_spor_res: Mock, mock_dlt_pipeline: Mock
) -> None:
    mock_dlt_pipeline.return_value = Mock()
    # This test verifies that after an update occurs, subsequent identical payloads
    # do not create new facts or alter the history timeline established during the update.
    # It tests: Day 1 (Insert) -> Day 2 (Update) -> Day 3 (Idempotent replay of Day 2).

    # Day 1 Data
    epar_data_day1 = [
        {
            "category": "Human",
            "product_number": "EMEA/H/C/009999",
            "medicine_name": "ComplexDrug",
            "marketing_authorisation_holder": "ComplexCorp",
            "active_substance": "Substance C",
            "therapeutic_area": "Area C",
            "atc_code": "C10BA02",
            "generic": False,
            "biosimilar": False,
            "orphan": False,
            "additional_monitoring": False,
            "conditional_approval": False,
            "exceptional_circumstances": False,
            "authorisation_status": "Authorised",
            "revision_date": "2023-01-01T00:00:00",
            "url": "https://www.ema.europa.eu/en/medicines/human/EPAR/complexdrug",
        }
    ]
    spor_data = [{"org_id": "ORG9999", "org_name": "complexcorp"}]

    mock_epar_res.return_value = iter(epar_data_day1)
    mock_spor_res.return_value = iter(spor_data)

    # Run 1: Initial Insert (Day 1)
    ingestion_ts_day1 = datetime(2023, 10, 1)
    _dim_1, fact_1, _bridge_1 = run_pipeline(
        epar_url="http://fake-epar",
        spor_url="http://fake-spor",
        ingestion_ts=ingestion_ts_day1,
        current_history=None,
        destination="dummy",
    )

    # Asserts for Day 1
    assert len(fact_1) == 1
    assert fact_1["status"][0] == RegulatoryStatusEnum.APPROVED.value
    assert fact_1["valid_from"][0] == ingestion_ts_day1
    assert fact_1["valid_to"][0] is None
    assert fact_1["is_current"][0] is True

    # Day 2 Data (Status changes to Withdrawn)
    epar_data_day2 = [
        {
            "category": "Human",
            "product_number": "EMEA/H/C/009999",
            "medicine_name": "ComplexDrug",
            "marketing_authorisation_holder": "ComplexCorp",
            "active_substance": "Substance C",
            "therapeutic_area": "Area C",
            "atc_code": "C10BA02",
            "generic": False,
            "biosimilar": False,
            "orphan": False,
            "conditional_approval": False,
            "exceptional_circumstances": False,
            "additional_monitoring": False,
            "authorisation_status": "Withdrawn",
            "revision_date": "2023-10-02T00:00:00",
            "url": "https://www.ema.europa.eu/en/medicines/human/EPAR/complexdrug",
        }
    ]

    mock_epar_res.return_value = iter(epar_data_day2)
    mock_spor_res.return_value = iter(spor_data)

    # Run 2: Status Update (Day 2)
    ingestion_ts_day2 = datetime(2023, 10, 2)
    _dim_2, fact_2, _bridge_2 = run_pipeline(
        epar_url="http://fake-epar",
        spor_url="http://fake-spor",
        ingestion_ts=ingestion_ts_day2,
        current_history=fact_1,
        destination="dummy",
    )

    # Asserts for Day 2: Should now have 2 rows (old closed, new open)
    assert len(fact_2) == 2
    fact_2_sorted = fact_2.sort("valid_from")

    # Verify Old Row Closed
    assert fact_2_sorted["status"][0] == RegulatoryStatusEnum.APPROVED.value
    assert fact_2_sorted["is_current"][0] is False
    assert fact_2_sorted["valid_to"][0] == ingestion_ts_day2

    # Verify New Row Opened
    assert fact_2_sorted["status"][1] == RegulatoryStatusEnum.WITHDRAWN.value
    assert fact_2_sorted["is_current"][1] is True
    assert fact_2_sorted["valid_from"][1] == ingestion_ts_day2
    assert fact_2_sorted["valid_to"][1] is None

    # Run 3: Idempotent replay of Day 2 payload on Day 3
    mock_epar_res.return_value = iter(epar_data_day2)
    mock_spor_res.return_value = iter(spor_data)

    ingestion_ts_day3 = datetime(2023, 10, 3)
    _dim_3, fact_3, _bridge_3 = run_pipeline(
        epar_url="http://fake-epar",
        spor_url="http://fake-spor",
        ingestion_ts=ingestion_ts_day3,
        current_history=fact_2,
        destination="dummy",
    )

    # Asserts for Day 3: Strict Idempotency check
    # 1. Total history length must still be exactly 2
    assert len(fact_3) == 2

    fact_3_sorted = fact_3.sort("valid_from")

    # 2. Old row remains unchanged from Day 2 state
    assert fact_3_sorted["status"][0] == RegulatoryStatusEnum.APPROVED.value
    assert fact_3_sorted["is_current"][0] is False
    assert fact_3_sorted["valid_to"][0] == ingestion_ts_day2

    # 3. New row remains perfectly active from Day 2 ingestion_ts
    assert fact_3_sorted["status"][1] == RegulatoryStatusEnum.WITHDRAWN.value
    assert fact_3_sorted["is_current"][1] is True
    assert fact_3_sorted["valid_from"][1] == ingestion_ts_day2  # Lock to Day 2!
    assert fact_3_sorted["valid_to"][1] is None


@patch("coreason_etl_epar.main.dlt.pipeline")
@patch("coreason_etl_epar.main.get_spor_organisations_resource")
@patch("coreason_etl_epar.main.get_epar_index_resource")
def test_run_pipeline_idempotency_vanished_reappears(
    mock_epar_res: Mock, mock_spor_res: Mock, mock_dlt_pipeline: Mock
) -> None:
    mock_dlt_pipeline.return_value = Mock()

    # Day 1: Drug is present
    epar_data_day1 = [
        {
            "category": "Human",
            "product_number": "EMEA/H/C/000001",
            "medicine_name": "GhostDrug",
            "marketing_authorisation_holder": "GhostCorp",
            "active_substance": "Substance G",
            "therapeutic_area": "Area G",
            "atc_code": "G10BA01",
            "generic": False,
            "biosimilar": False,
            "orphan": False,
            "additional_monitoring": False,
            "conditional_approval": False,
            "exceptional_circumstances": False,
            "authorisation_status": "Authorised",
            "revision_date": "2023-01-01T00:00:00",
            "url": "https://www.ema.europa.eu/en/medicines/human/EPAR/ghostdrug",
        }
    ]
    spor_data = [{"org_id": "ORG0001", "org_name": "ghostcorp"}]

    mock_epar_res.return_value = iter(epar_data_day1)
    mock_spor_res.return_value = iter(spor_data)

    # Run 1: Initial Insert (Day 1)
    ingestion_ts_day1 = datetime(2023, 10, 1)
    _dim_1, fact_1, _bridge_1 = run_pipeline(
        epar_url="http://fake",
        spor_url="http://fake",
        ingestion_ts=ingestion_ts_day1,
        current_history=None,
    )

    # Day 2: Drug vanishes from snapshot (deleted/closed)
    # Return an empty list, but wrapped in iter() to simulate the generator
    # However, to avoid schema inference issues on empty dataframe in test,
    # let's return a different dummy drug so schema is preserved.
    epar_data_day2_dummy = [
        {
            "category": "Human",
            "product_number": "EMEA/H/C/000999",
            "medicine_name": "Dummy",
            "marketing_authorisation_holder": "GhostCorp",
            "active_substance": "Substance G",
            "therapeutic_area": "Area G",
            "atc_code": "G10BA01",
            "generic": False,
            "biosimilar": False,
            "orphan": False,
            "additional_monitoring": False,
            "conditional_approval": False,
            "exceptional_circumstances": False,
            "authorisation_status": "Authorised",
            "revision_date": "2023-01-01T00:00:00",
            "url": "https://www.ema.europa.eu/en/medicines/human/EPAR/ghostdrug",
        }
    ]
    mock_epar_res.return_value = iter(epar_data_day2_dummy)
    mock_spor_res.return_value = iter(spor_data)

    ingestion_ts_day2 = datetime(2023, 10, 2)
    _dim_2, fact_2, _bridge_2 = run_pipeline(
        epar_url="http://fake",
        spor_url="http://fake",
        ingestion_ts=ingestion_ts_day2,
        current_history=fact_1,
    )

    # Assert Day 2: Record should be closed (is_current=False)
    # Plus the dummy record is inserted
    assert len(fact_2) == 2
    fact_2_ghost = fact_2.filter(pl.col("coreason_id") == fact_1["coreason_id"][0])
    assert fact_2_ghost["is_current"][0] is False
    assert fact_2_ghost["valid_to"][0] == ingestion_ts_day2

    # Day 3: Drug reappears (identical data)
    mock_epar_res.return_value = iter(epar_data_day1)
    mock_spor_res.return_value = iter(spor_data)

    ingestion_ts_day3 = datetime(2023, 10, 3)
    _dim_3, fact_3, _bridge_3 = run_pipeline(
        epar_url="http://fake",
        spor_url="http://fake",
        ingestion_ts=ingestion_ts_day3,
        current_history=fact_2,
    )

    # Assert Day 3: Should have 2 records (1 closed from Day 2, 1 new active from Day 3)
    # We ignore the dummy drug since it's not present in Day 3 anymore
    # Actually just check ghost
    fact_3_ghost = fact_3.filter(pl.col("coreason_id") == fact_1["coreason_id"][0]).sort("valid_from")
    assert len(fact_3_ghost) == 2
    assert fact_3_ghost["is_current"][0] is False
    assert fact_3_ghost["valid_to"][0] == ingestion_ts_day2
    assert fact_3_ghost["is_current"][1] is True
    assert fact_3_ghost["valid_from"][1] == ingestion_ts_day3

    # Day 4: Idempotent replay of Day 3
    mock_epar_res.return_value = iter(epar_data_day1)
    mock_spor_res.return_value = iter(spor_data)

    ingestion_ts_day4 = datetime(2023, 10, 4)
    _dim_4, fact_4, _bridge_4 = run_pipeline(
        epar_url="http://fake",
        spor_url="http://fake",
        ingestion_ts=ingestion_ts_day4,
        current_history=fact_3,
    )

    # Assert Day 4: Strict idempotency (history matches Day 3 exactly)
    assert len(fact_4) == len(fact_3)
    fact_4_ghost = fact_4.filter(pl.col("coreason_id") == fact_1["coreason_id"][0]).sort("valid_from")
    assert len(fact_4_ghost) == 2
    assert fact_4_ghost["is_current"][1] is True
    assert fact_4_ghost["valid_from"][1] == ingestion_ts_day3


@patch("coreason_etl_epar.main.dlt.pipeline")
@patch("coreason_etl_epar.main.get_spor_organisations_resource")
@patch("coreason_etl_epar.main.get_epar_index_resource")
def test_run_pipeline_idempotency_status_reverted(
    mock_epar_res: Mock, mock_spor_res: Mock, mock_dlt_pipeline: Mock
) -> None:
    mock_dlt_pipeline.return_value = Mock()

    # Day 1: Approved
    epar_data_day1 = [
        {
            "category": "Human",
            "product_number": "EMEA/H/C/000002",
            "medicine_name": "FlapDrug",
            "marketing_authorisation_holder": "FlapCorp",
            "active_substance": "Substance F",
            "therapeutic_area": "Area F",
            "atc_code": "F10BA02",
            "authorisation_status": "Authorised",
            "url": "https://www.ema.europa.eu",
            "generic": False,
            "biosimilar": False,
            "orphan": False,
            "additional_monitoring": False,
            "conditional_approval": False,
            "exceptional_circumstances": False,
        }
    ]
    spor_data = [{"org_id": "ORG0002", "org_name": "flapcorp"}]

    mock_epar_res.return_value = iter(epar_data_day1)
    mock_spor_res.return_value = iter(spor_data)

    ingestion_ts_day1 = datetime(2023, 10, 1)
    _dim_1, fact_1, _bridge_1 = run_pipeline(
        epar_url="http://fake",
        spor_url="http://fake",
        ingestion_ts=ingestion_ts_day1,
        current_history=None,
    )

    # Day 2: Suspended
    epar_data_day2 = [dict(epar_data_day1[0])]
    epar_data_day2[0]["authorisation_status"] = "Suspended"

    mock_epar_res.return_value = iter(epar_data_day2)
    mock_spor_res.return_value = iter(spor_data)

    ingestion_ts_day2 = datetime(2023, 10, 2)
    _dim_2, fact_2, _bridge_2 = run_pipeline(
        epar_url="http://fake",
        spor_url="http://fake",
        ingestion_ts=ingestion_ts_day2,
        current_history=fact_1,
    )

    # Day 3: Reverted back to Approved
    mock_epar_res.return_value = iter(epar_data_day1)
    mock_spor_res.return_value = iter(spor_data)

    ingestion_ts_day3 = datetime(2023, 10, 3)
    _dim_3, fact_3, _bridge_3 = run_pipeline(
        epar_url="http://fake",
        spor_url="http://fake",
        ingestion_ts=ingestion_ts_day3,
        current_history=fact_2,
    )

    # Assert Day 3: Should have 3 records (Day 1 closed, Day 2 closed, Day 3 active)
    assert len(fact_3) == 3
    fact_3_sorted = fact_3.sort("valid_from")

    assert fact_3_sorted["status"][0] == RegulatoryStatusEnum.APPROVED.value
    assert fact_3_sorted["is_current"][0] is False

    assert fact_3_sorted["status"][1] == RegulatoryStatusEnum.SUSPENDED.value
    assert fact_3_sorted["is_current"][1] is False
    assert fact_3_sorted["valid_to"][1] == ingestion_ts_day3

    assert fact_3_sorted["status"][2] == RegulatoryStatusEnum.APPROVED.value
    assert fact_3_sorted["is_current"][2] is True
    assert fact_3_sorted["valid_from"][2] == ingestion_ts_day3

    # Day 4: Idempotent replay
    mock_epar_res.return_value = iter(epar_data_day1)
    mock_spor_res.return_value = iter(spor_data)

    ingestion_ts_day4 = datetime(2023, 10, 4)
    _dim_4, fact_4, _bridge_4 = run_pipeline(
        epar_url="http://fake",
        spor_url="http://fake",
        ingestion_ts=ingestion_ts_day4,
        current_history=fact_3,
    )

    # Assert Day 4: Strict idempotency
    assert len(fact_4) == 3
    fact_4_sorted = fact_4.sort("valid_from")
    assert fact_4_sorted["is_current"][2] is True
    assert fact_4_sorted["valid_from"][2] == ingestion_ts_day3


@patch("coreason_etl_epar.main.dlt.pipeline")
@patch("coreason_etl_epar.main.get_spor_organisations_resource")
@patch("coreason_etl_epar.main.get_epar_index_resource")
def test_run_pipeline_schema_and_table_naming_conventions(
    mock_epar_res: Mock, mock_spor_res: Mock, mock_dlt_pipeline: Mock
) -> None:
    # We want to trace all calls to dlt.pipeline and the returned pipeline's run() method.
    mock_pipeline_instance = Mock()
    mock_dlt_pipeline.return_value = mock_pipeline_instance

    mock_epar_res.return_value = iter(
        [
            {
                "category": "Human",
                "product_number": "EMEA/H/C/000003",
                "medicine_name": "TestDrug",
                "marketing_authorisation_holder": "TestCorp",
                "active_substance": "Substance T",
                "therapeutic_area": "Area T",
                "atc_code": "T10BA02",
                "generic": False,
                "biosimilar": False,
                "orphan": False,
                "conditional_approval": False,
                "exceptional_circumstances": False,
                "additional_monitoring": False,
                "authorisation_status": "Authorised",
                "url": "https://www.ema.europa.eu",
            }
        ]
    )
    mock_spor_res.return_value = iter([{"org_id": "ORG0003", "org_name": "testcorp"}])

    run_pipeline(
        epar_url="http://fake",
        spor_url="http://fake",
        ingestion_ts=datetime(2023, 10, 1),
    )

    # Check all calls to dlt.pipeline
    pipeline_calls = mock_dlt_pipeline.call_args_list
    assert len(pipeline_calls) == 3

    # 1. Bronze Pipeline
    assert pipeline_calls[0].kwargs["dataset_name"] == "bronze"
    assert pipeline_calls[0].kwargs["pipeline_name"] == "coreason_etl_epar_bronze"

    # 2. Silver Pipeline
    assert pipeline_calls[1].kwargs["dataset_name"] == "silver"
    assert pipeline_calls[1].kwargs["pipeline_name"] == "coreason_etl_epar_silver"

    # 3. Gold Pipeline
    assert pipeline_calls[2].kwargs["dataset_name"] == "gold"
    assert pipeline_calls[2].kwargs["pipeline_name"] == "coreason_etl_epar_gold"

    # Now check the resources passed to the pipeline.run() calls
    run_calls = mock_pipeline_instance.run.call_args_list
    assert len(run_calls) == 3

    # We can inspect the resources (which are DltResource objects) by their .name attribute
    bronze_resources = run_calls[0].args[0]
    assert len(bronze_resources) == 2
    assert bronze_resources[0].name == "coreason_etl_epar_bronze_epar_index"
    assert bronze_resources[1].name == "coreason_etl_epar_bronze_spor_organisations"

    silver_resources = run_calls[1].args[0]
    assert len(silver_resources) == 1
    assert silver_resources[0].name == "coreason_etl_epar_silver_epar_normalized"

    gold_resources = run_calls[2].args[0]
    assert len(gold_resources) == 3
    assert gold_resources[0].name == "coreason_etl_epar_gold_dim_medicine"
    assert gold_resources[1].name == "coreason_etl_epar_gold_fact_regulatory_history"
    assert gold_resources[2].name == "coreason_etl_epar_gold_bridge_medicine_features"
