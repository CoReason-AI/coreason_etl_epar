# 1. Update ingest.py to read the new Excel format starting at row 0
with open("src/coreason_etl_epar/ingest.py", "r") as f:
    ingest_code = f.read()

ingest_code = ingest_code.replace("skiprows=8", "skiprows=0")

with open("src/coreason_etl_epar/ingest.py", "w") as f:
    f.write(ingest_code)

# 2. Update main.py to safely ignore quarantined error records
with open("src/coreason_etl_epar/main.py", "r") as f:
    main_code = f.read()

main_code = main_code.replace(
    "epar_dicts = [item for item in epar_generator if isinstance(item, dict)]",
    "epar_dicts = [item for item in epar_generator if isinstance(item, dict) and \"error\" not in item]"
)

with open("src/coreason_etl_epar/main.py", "w") as f:
    f.write(main_code)

print("Successfully patched ingest.py and main.py!")
