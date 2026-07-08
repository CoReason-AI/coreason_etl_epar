import threading
import http.server
import socketserver
from datetime import datetime
from coreason_etl_epar.main import run_pipeline

# Start a background local web server to host your REAL downloaded zip file
class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress server logs to keep terminal clean

PORT = 8080
socketserver.TCPServer.allow_reuse_address = True
httpd = socketserver.TCPServer(("", PORT), QuietHandler)
thread = threading.Thread(target=httpd.serve_forever)
thread.daemon = True
thread.start()

if __name__ == "__main__":
    # Live EMA EPAR URL (100% Real Data)
    EPAR_EXCEL_URL = "https://www.ema.europa.eu/en/documents/report/medicines-output-medicines-report_en.xlsx"
    
    # Feeds your REAL downloaded SPOR database into the pipeline
    SPOR_XML_ZIP_URL = f"http://localhost:{PORT}/OMS_Export.zip"

    print("Starting EPAR ETL Pipeline with 100% REAL data...")
    
    # Execute the pipeline
    dim, fact, bridge = run_pipeline(
        epar_url=EPAR_EXCEL_URL,
        spor_url=SPOR_XML_ZIP_URL,
        ingestion_ts=datetime.now(),
        destination="postgres"
    )
    
    print("\nPipeline execution complete!")
    print(f"Loaded {len(dim)} records into dim_medicine.")
