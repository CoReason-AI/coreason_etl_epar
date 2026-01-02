import polars as pl

from coreason_etl_epar.transform_silver import get_status_normalization_expr


def normalize(status_list: list[str]) -> list[str]:
    """Helper to run normalization on a list of strings."""
    df = pl.DataFrame({"status": status_list})
    return df.select(get_status_normalization_expr("status").alias("norm"))["norm"].to_list()


def test_status_precedence_lifted() -> None:
    """
    Test 'LIFTED' precedence.
    'LIFTED' implies 'Suspension Lifted', which effectively means APPROVED.
    It MUST take precedence over 'SUSPENDED' or 'SUSPENSION' keywords appearing in the same string.
    """
    inputs = ["Suspension Lifted", "Lifted Suspension", "Suspension of Marketing Authorisation Lifted", "Lifted"]
    expected = ["APPROVED", "APPROVED", "APPROVED", "APPROVED"]
    assert normalize(inputs) == expected


def test_status_precedence_suspension() -> None:
    """
    Test 'SUSPENSION' and 'SUSPENDED' mapping.
    Should map to SUSPENDED.
    """
    inputs = ["Suspended", "Suspension of Authorisation", "Marketing Authorisation Suspended", "Suspension"]
    expected = ["SUSPENDED", "SUSPENDED", "SUSPENDED", "SUSPENDED"]
    assert normalize(inputs) == expected


def test_status_precedence_authorisation() -> None:
    """
    Test 'AUTHORISATION' vs 'AUTHORISED'.
    Both should map to APPROVED.
    Must handle exclusion of 'NOT'.
    """
    inputs = [
        "Authorised",
        "Marketing Authorisation",
        "Authorisation",
        "Not Authorised",
        "Authorisation Refused",  # Should be REJECTED (precedence check)
        "Authorisation Suspended",  # Should be SUSPENDED (precedence check)
    ]
    expected = [
        "APPROVED",
        "APPROVED",
        "APPROVED",
        "UNKNOWN",  # 'Not Authorised' -> UNKNOWN
        "REJECTED",  # REFUSED > AUTHORISATION
        "SUSPENDED",  # SUSPENDED > AUTHORISATION (Wait, check implementation precedence!)
    ]
    assert normalize(inputs) == expected


def test_status_precedence_expired() -> None:
    """
    Test 'EXPIRED' mapping.
    Should map to WITHDRAWN.
    """
    inputs = ["Expired", "Marketing Authorisation Expired", "Expired Authorisation"]
    expected = ["WITHDRAWN", "WITHDRAWN", "WITHDRAWN"]
    assert normalize(inputs) == expected


def test_status_complex_combinations() -> None:
    """
    Test complex combinations to verify strict precedence order.
    Order: REFUSED > WITHDRAWN/EXPIRED > SUSPENDED (LIFTED>SUSP) > CONDITIONAL > EXCEPTIONAL > APPROVED
    """
    # Case: "Refused Authorisation"
    # Contains: REFUSED, AUTHORISATION
    # Expected: REJECTED (REFUSED is #1)
    assert normalize(["Refused Authorisation"]) == ["REJECTED"]

    # Case: "Expired Marketing Authorisation"
    # Contains: EXPIRED, AUTHORISATION
    # Expected: WITHDRAWN (EXPIRED is #2)
    assert normalize(["Expired Marketing Authorisation"]) == ["WITHDRAWN"]

    # Case: "Suspension Lifted"
    # Contains: SUSPENSION, LIFTED
    # Expected: APPROVED (LIFTED > SUSPENDED)
    assert normalize(["Suspension Lifted"]) == ["APPROVED"]

    # Case: "Suspended Authorisation"
    # Contains: SUSPENDED, AUTHORISATION
    # Expected: SUSPENDED (SUSPENDED > APPROVED)
    assert normalize(["Suspended Authorisation"]) == ["SUSPENDED"]

    # Case: "Conditional Marketing Authorisation"
    # Contains: CONDITIONAL, AUTHORISATION
    # Expected: CONDITIONAL_APPROVAL (CONDITIONAL > APPROVED)
    assert normalize(["Conditional Marketing Authorisation"]) == ["CONDITIONAL_APPROVAL"]

    # Case: "Withdrawn (prior Conditional approval)"
    # Contains: WITHDRAWN, CONDITIONAL
    # Expected: WITHDRAWN (WITHDRAWN > CONDITIONAL)
    assert normalize(["Withdrawn (prior Conditional approval)"]) == ["WITHDRAWN"]
