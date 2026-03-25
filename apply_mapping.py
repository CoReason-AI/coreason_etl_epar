with open("src/coreason_etl_epar/ingest.py", "r") as f:
    code = f.read()

old_mapping = """    column_mapping = {
        "international_non-proprietary_name_inn___common_name": "active_substance",
        "active_substance": "active_substance",
        "orphan_medicine": "orphan",
        "marketing_authorisation_holder_company_name": "marketing_authorisation_holder",
    }"""

new_mapping = """    column_mapping = {
        "name_of_medicine": "medicine_name",
        "ema_product_number": "product_number",
        "medicine_status": "authorisation_status",
        "marketing_authorisation_developer___applicant___holder": "marketing_authorisation_holder",
        "international_non-proprietary_name_inn___common_name": "active_substance",
        "active_substance": "active_substance",
        "therapeutic_area_mesh": "therapeutic_area",
        "atc_code_human": "atc_code",
        "orphan_medicine": "orphan",
        "medicine_url": "url"
    }"""

code = code.replace(old_mapping, new_mapping)

with open("src/coreason_etl_epar/ingest.py", "w") as f:
    f.write(code)

print("Successfully updated column mapping in ingest.py!")
