with open("src/coreason_etl_epar/ingest.py", "r") as f:
    code = f.read()

# Tell Pydantic to output standard strings instead of custom HttpUrl objects
code = code.replace("valid_row.model_dump()", "valid_row.model_dump(mode='json')")

with open("src/coreason_etl_epar/ingest.py", "w") as f:
    f.write(code)

print("Successfully updated Pydantic serialization in ingest.py!")
