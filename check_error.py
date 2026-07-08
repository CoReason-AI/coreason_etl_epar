from coreason_etl_epar.ingest import get_epar_index_resource

EPAR_EXCEL_URL = "https://www.ema.europa.eu/en/documents/report/medicines-output-medicines-report_en.xlsx"

print("Fetching EPAR data to check validation errors...\n")
generator = get_epar_index_resource(EPAR_EXCEL_URL)

for item in generator:
    # If the item is a quarantined error dictionary
    if isinstance(item, dict) and "error" in item:
        print("🚨 PYDANTIC VALIDATION ERROR:")
        print(item["error"])
        print("\n📦 RAW ROW DATA RECEIVED:")
        for key, value in item["raw_data"].items():
            print(f"  {key}: {value}")
        break
