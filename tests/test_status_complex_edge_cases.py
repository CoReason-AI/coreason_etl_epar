import polars as pl

from coreason_etl_epar.transform_silver import get_status_normalization_expr


def run_status_norm(status: str) -> str:
    """Helper to run the expression on a single string."""
    df = pl.DataFrame({"status": [status]})
    return str(df.select(get_status_normalization_expr("status").alias("res"))["res"].item())


def test_status_terminal_priority() -> None:
    """
    Terminal states (Withdrawn, Refused, Suspended) should take precedence
    over qualifiers like Conditional or Exceptional.
    """
    assert run_status_norm("Withdrawn (prior Conditional approval)") == "WITHDRAWN"
    assert run_status_norm("Refused (under Exceptional Circumstances)") == "REJECTED"
    assert run_status_norm("Suspended Marketing Authorisation") == "SUSPENDED"


def test_status_formatting_edge_cases() -> None:
    """
    Test odd formatting, casing, and spacing.
    """
    assert run_status_norm("  WITHDRAWN  ") == "WITHDRAWN"
    assert run_status_norm("conditional marketing authorisation") == "CONDITIONAL_APPROVAL"
    assert run_status_norm("Authorised\t") == "APPROVED"
    assert run_status_norm("") == "UNKNOWN"
    assert run_status_norm("---") == "UNKNOWN"


def test_status_multiple_qualifiers() -> None:
    """
    Test combinations of qualifiers.
    """
    res = run_status_norm("Conditional approval under exceptional circumstances")
    assert res in ["CONDITIONAL_APPROVAL", "EXCEPTIONAL_CIRCUMSTANCES"]
    assert res != "APPROVED"


def test_status_not_authorised_defect() -> None:
    """
    Defect Fix Verification: 'Not Authorised' should NOT map to 'APPROVED'.
    It should fall through to UNKNOWN (or REJECTED if we changed logic, but plan said UNKNOWN).
    """
    assert run_status_norm("Not Authorised") == "UNKNOWN"
    # Ensure standard Authorised still works
    assert run_status_norm("Marketing Authorised") == "APPROVED"


def test_status_nuances() -> None:
    """
    Test new nuanced status logic: LIFTED, SUSPENSION, EXPIRED, AUTHORISATION.
    """
    # LIFTED
    assert run_status_norm("Suspension of the marketing authorisation lifted") == "APPROVED"
    assert run_status_norm("Suspension lifted") == "APPROVED"

    # SUSPENSION (without LIFTED)
    assert run_status_norm("Suspension of the marketing authorisation") == "SUSPENDED"

    # EXPIRED
    assert run_status_norm("Marketing Authorisation Expired") == "WITHDRAWN"

    # AUTHORISATION vs AUTHORISED
    # "Marketing Authorisation" -> APPROVED (was UNKNOWN previously if checks were strictly AUTHORISED)
    assert run_status_norm("Marketing Authorisation") == "APPROVED"

    # Negative check still works for Authorisation
    assert run_status_norm("Not Marketing Authorisation") == "UNKNOWN"
