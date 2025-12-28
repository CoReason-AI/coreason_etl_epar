# Business Requirements Document (BRD): coreason_etl_epar

## 1. Executive Summary
This initiative establishes a Type A (Commodity) data package to ingest European Public Assessment Reports (EPAR). EPARs represent the official scientific evaluation of medicines authorized, refused, or withdrawn in the European Union by the EMA.
This pipeline constitutes the "European Leg" of the CoReason Regulatory Knowledge Graph. By harmonizing EMA data to ISO/IDMP standards, we enable the Reasoning Engine to detect conflicts, opportunities, and safety signals across US (FDA) and EU (EMA) markets.

## 2. Stakeholders & Ownership
* **Business Owner:** Head of Regulatory Intelligence.
* **Technical Owner:** Lead Data Engineer (Core Platform).
* **Domain Expert:** Senior Regulatory Affairs Manager (EU Specialist).
* **Consumers:** Regulatory Strategy Teams, Market Access Modeling, Downstream Graph Pipelines.

## 3. Business Goals
* **Regulatory Divergence:** Detect "Gap Signals"—drugs approved in the US but Refused or Conditional in the EU.
* **Biosimilar Intelligence:** Authoritative mapping of Reference Products vs. Biosimilars to power market access models.
* **Safety Surveillance:** Tracking "Black Triangle" (Additional Monitoring) status and Class-Based Safety signals (using ATC codes).
* **Orphan Drug Tracking:** Monitor "Orphan Maintenance" status—identifying when high-value orphan designations are withdrawn.

## 4. Success Criteria
* **History Tracking (SCD Type 2):** The system must answer: "On what date did this drug's status change from Conditional to Full Approval?"
* **Zero "API Dependency" Failures:** Enrichment must rely on cached/bulk datasets (SPOR Exports), not live API calls.
* **Completeness:** 100% ingestion of "Human" category records. "Veterinary" records must be strictly excluded.
* **Granularity:** Successful extraction of individual ATC codes and Therapeutic Areas into searchable arrays.
