import polars as pl
from coreason_etl_epar.transform_enrich import enrich_epar, jaro_winkler, normalize_status


def test_jaro_winkler() -> None:
    # Exact match
    assert jaro_winkler("test", "test") == 1.0
    # No match
    assert jaro_winkler("abc", "xyz") == 0.0
    # Empty strings
    assert jaro_winkler("", "abc") == 0.0
    assert jaro_winkler("abc", "") == 0.0
    assert jaro_winkler("", "") == 1.0  # Handled by first check
    # Branch check for len == 0 handled.
    # Check transposition
    # "dixon" vs "dicksonx"
    # Matches: d, i, o, n (4).
    # Transposition logic triggered if s1[i] != s2[k] but matched.
    # 'martha' vs 'marhta' -> t and h are swapped.
    assert jaro_winkler("martha", "marhta") > 0.9
    # Check no match branch inside inner loop (lines 48-51) is implicitly covered by low scores.

    # Check break in prefix loop
    assert jaro_winkler("abc", "abd") < 1.0  # prefix = 2
    assert jaro_winkler("dixon", "dicksonx") > 0.7

    # Check duplicate char handling (covers 'continue' when s2_matches[j] is True)
    # s1="aa", s2="a"
    # i=0 matches s2[0]. s2_matches[0]=True.
    # i=1 sees s2[0] used -> continue.
    assert jaro_winkler("aa", "a") > 0.0

    # Specific case for hitting 'continue' in inner loop
    # s1="aba", s2="bab"
    # i=0 ('a') matches s2[1]. s2_matches[1]=True
    # i=1 ('b') matches s2[0]. s2_matches[0]=True
    # i=2 ('a') looks at s2[0] (used->continue? No s2[0] is 'b'!=s1[2])
    # Let's construct a case: s1="aa", s2="aa".
    # i=0 ('a') matches s2[0]. s2_matches[0]=True.
    # i=1 ('a') checks s2[0]. s2_matches[0] is True -> continue (Hits line 15).
    # Then checks s2[1]. Match.
    assert jaro_winkler("aa", "aa") == 1.0

    # Extra cases to force coverage
    assert jaro_winkler("CRATE", "TRACE") > 0.0  # Transposition
    assert jaro_winkler("DwAyNE", "DuANE") > 0.0
    assert jaro_winkler("ABC", "XBC") > 0.0  # Match logic


def test_normalize_status() -> None:
    assert normalize_status("Authorised") == "APPROVED"
    assert normalize_status(" Conditional ") == "CONDITIONAL_APPROVAL"
    assert normalize_status("Exceptional Circumstances") == "EXCEPTIONAL_CIRCUMSTANCES"
    assert normalize_status("Refused") == "REJECTED"
    assert normalize_status("Withdrawn") == "WITHDRAWN"
    assert normalize_status("Suspended") == "SUSPENDED"
    assert normalize_status("Unknown Status") == "UNKNOWN"


def test_enrich_epar_logic() -> None:
    # Setup Data
    df = pl.DataFrame(
        {
            "product_number": ["EMEA/H/C/001234", "EMEA/H/C/999"],
            "active_substance": ["Sub A / Sub B", "Sub C + Sub D"],
            "atc_code": ["A01;B02", "C03,D04"],
            "authorisation_status": ["Authorised", "Refused"],
            "marketing_authorisation_holder": ["Pharma Corp", "BioTech Inc"],
        }
    )

    # Use names that are very close to satisfy > 0.90 threshold
    spor_df = pl.DataFrame(
        {"name": ["Pharma Corp.", "BioTech Inc.", "Other Co"], "org_id": ["ORG-100", "ORG-200", "ORG-300"]}
    )

    # Run Enrichment
    result = enrich_epar(df, spor_df)

    # Check Base Procedure ID
    assert result.filter(pl.col("product_number") == "EMEA/H/C/001234")["base_procedure_id"].item() == "001234"

    # Check Substance List
    row1 = result.row(0, named=True)
    assert row1["active_substance_list"] == ["Sub A", "Sub B"]
    row2 = result.row(1, named=True)
    assert row2["active_substance_list"] == ["Sub C", "Sub D"]  # Handled +

    # Check ATC List
    assert row1["atc_code_list"] == ["A01", "B02"]
    assert row2["atc_code_list"] == ["C03", "D04"]  # Handled ,

    # Check Status
    assert row1["status_normalized"] == "APPROVED"

    # Check Fuzzy Match
    # Pharma Corp vs Pharma Corp. -> Should be > 0.90

    assert row1["spor_mah_id"] == "ORG-100"
    assert row2["spor_mah_id"] == "ORG-200"


def test_enrich_epar_no_match() -> None:
    df = pl.DataFrame(
        {
            "product_number": ["EMEA/H/C/001234"],
            "active_substance": ["A"],
            "atc_code": ["A"],
            "authorisation_status": ["Authorised"],
            "marketing_authorisation_holder": ["Unique Name"],
        }
    )

    spor_df = pl.DataFrame({"name": ["Totally Different"], "org_id": ["ORG-999"]})

    result = enrich_epar(df, spor_df)
    assert result["spor_mah_id"].item() is None


def test_enrich_tie_breaker_determinism() -> None:
    # Setup: 1 EPAR MAH, 2 SPOR MAHs with IDENTICAL names (score 1.0) but different IDs.
    # We want deterministic choice (lowest ID).

    df = pl.DataFrame(
        {
            "product_number": ["P1"],
            "medicine_name": ["M1"],
            "marketing_authorisation_holder": ["Pharma Corp"],
            "active_substance": ["S1"],
            "atc_code": ["A1"],
            "authorisation_status": ["Authorised"],
            "url": ["u"],
        }
    )

    # SPOR data with duplicates or very similar
    spor_df = pl.DataFrame(
        {
            "name": ["Pharma Corp", "Pharma Corp"],
            "org_id": ["ORG-002", "ORG-001"],  # ORG-001 should be picked if sorting by ID ascending
        }
    )

    result = enrich_epar(df, spor_df)

    # Assert
    assert result["spor_mah_id"].item() == "ORG-001"


def test_unicode_handling() -> None:
    # Enrichment (Jaro-Winkler with unicode)
    # "Société" vs "Societe"
    # Matches: S,o,c,i,t (5). é vs e is mismatch.
    # Should calculate a score.
    # Just ensure it doesn't crash.
    score = jaro_winkler("Société", "Societe")
    assert score > 0.0


def test_enrich_epar_empty_spor() -> None:
    df = pl.DataFrame(
        {
            "product_number": ["EMEA/H/C/001234"],
            "active_substance": ["A"],
            "atc_code": ["A"],
            "authorisation_status": ["Authorised"],
            "marketing_authorisation_holder": ["Unique Name"],
        }
    )

    spor_df = pl.DataFrame(schema={"name": pl.String, "org_id": pl.String})

    result = enrich_epar(df, spor_df)
    assert result["spor_mah_id"].item() is None
