import polars as pl

from coreason_etl_epar.transform import normalize_atc_code

df = pl.DataFrame({"atc_code": ["A01B; C02D", "D03E,F04G", "A10BA02; INVALID", "X99XX99", "A12BC34, b99cd99"]})
print(normalize_atc_code(df))
