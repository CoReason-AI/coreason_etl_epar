import pandas as pd

# 1. Revert our previous patch in ingest.py back to skipping 8 rows
with open("src/coreason_etl_epar/ingest.py", "r") as f:
    code = f.read().replace("skiprows=0", "skiprows=8")
with open("src/coreason_etl_epar/ingest.py", "w") as f:
    f.write(code)

# 2. Download the file and extract just the headers
url = "https://www.ema.europa.eu/en/documents/report/medicines-output-medicines-report_en.xlsx"
print("Downloading EMA Excel file to inspect new column headers...\n")
df = pd.read_excel(url, skiprows=8, nrows=0)

print("🚨 THE NEW EMA COLUMN HEADERS ARE:")
for col in df.columns:
    print(f" - {col}")
