from coreason_etl_epar.transform_silver import normalize_status


def test_status_terminal_priority() -> None:
    """
    Terminal states (Withdrawn, Refused, Suspended) should take precedence
    over qualifiers like Conditional or Exceptional.
    """
    # "Withdrawn" AND "Conditional" -> Should be WITHDRAWN
    # Current logic (Conditional first) will likely fail here
    assert normalize_status("Withdrawn (prior Conditional approval)") == "WITHDRAWN"

    # "Refused" AND "Exceptional" -> Should be REJECTED
    assert normalize_status("Refused (under Exceptional Circumstances)") == "REJECTED"

    # "Suspended" AND "Authorised" -> Should be SUSPENDED
    assert normalize_status("Suspended Marketing Authorisation") == "SUSPENDED"


def test_status_formatting_edge_cases() -> None:
    """
    Test odd formatting, casing, and spacing.
    """
    assert normalize_status("  WITHDRAWN  ") == "WITHDRAWN"
    assert normalize_status("conditional marketing authorisation") == "CONDITIONAL_APPROVAL"
    # Test internal whitespace/tabs if cleaned first? normalization strips \u200b but here we test the util directly
    assert normalize_status("Authorised\t") == "APPROVED"
    assert normalize_status("") == "UNKNOWN"
    assert normalize_status("---") == "UNKNOWN"


def test_status_multiple_qualifiers() -> None:
    """
    Test combinations of qualifiers.
    """
    # "Conditional" AND "Exceptional" -> Hard to say which wins, but usually Exceptional is a sub-condition
    # of how it's authorised.
    # However, for this exercise, we just ensure it doesn't default to APPROVED or UNKNOWN.
    # If both are present, either is better than APPROVED.
    # Let's assume Conditional is the 'stronger' regulatory status label for the graph if both appear,
    # but strictly speaking they are distinct attributes.
    # We will just assert it is NOT "APPROVED".
    res = normalize_status("Conditional approval under exceptional circumstances")
    assert res in ["CONDITIONAL_APPROVAL", "EXCEPTIONAL_CIRCUMSTANCES"]
    assert res != "APPROVED"
