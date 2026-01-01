import polars as pl

from coreason_etl_epar.transform_silver import get_status_normalization_expr


def normalize(status_list: list[str]) -> list[str]:
    """Helper to run normalization on a list of strings."""
    df = pl.DataFrame({"status": status_list})
    return df.select(get_status_normalization_expr("status").alias("norm"))["norm"].to_list()


def test_false_positive_authorisation() -> None:
    """
    Test ensuring words containing 'AUTHORISED' but meaning the opposite are not mapped to APPROVED.
    """
    inputs = [
        "Unauthorised",
        "De-authorised",
        "Pre-authorisation",  # Maybe UNKNOWN? Definitely not fully APPROVED yet?
        "Non-authorised",
    ]
    # Current logic might map these to APPROVED because they contain "AUTHORISED" and not "NOT ".
    # We expect UNKNOWN (or specific mapping), but definitely NOT APPROVED.
    results = normalize(inputs)
    for res, inp in zip(results, inputs, strict=False):
        assert res != "APPROVED", f"Failed: '{inp}' mapped to APPROVED"


def test_withdrawal_nuances() -> None:
    """
    Test nuances of withdrawal.
    """
    # "Not Renewed" implies it was authorized but is no longer.
    # Logic: If it contains "NOT", it currently goes to UNKNOWN.
    # Ideally, "Not Renewed" should be WITHDRAWN.

    # "Withdrawn by Applicant" -> WITHDRAWN
    assert normalize(["Withdrawn by Applicant"]) == ["WITHDRAWN"]
    assert normalize(["Marketing Authorisation Not Renewed"]) == ["WITHDRAWN"]


def test_refusal_nuances() -> None:
    """
    Test nuances of refusal.
    """
    # Partially Refused -> REJECTED (Terminal state precedence)
    assert normalize(["Partially Refused"]) == ["REJECTED"]

    # Refusé: If we normalize to uppercase, it becomes "REFUSÉ".
    # Regex "REFUSED" won't match "REFUSÉ".
    # Should we handle this? EPARs are usually English, but safety is good.
    # If we strip accents, "REFUSE" -> Matches "REFUSED"? No, missing D.

    # For now, let's see current behavior.


def test_mixed_garbage() -> None:
    """
    Test robustness against garbage.
    """
    inputs = [
        "AUTHORISED;",
        "AUTHORISED (Safety)",
        "AUTHORISED/Suspended",  # Should be SUSPENDED (Precedence)
    ]

    expected = ["APPROVED", "APPROVED", "SUSPENDED"]
    assert normalize(inputs) == expected
