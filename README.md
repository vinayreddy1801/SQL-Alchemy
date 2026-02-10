# Real-Time Career Intelligence Engine

A production-grade portfolio project that monitors job market data, calculates an 'Opportunity Score' using advanced SQL, and forecasts skill demand.

## Features

- **Job Market Monitoring**: Ingests job data into a local database.
- **Opportunity Score**: Calculates high-pay/high-demand skills using SQL CTEs.
- **Sentinel Reliability**: Data integrity checks (null handling, schema rules) before ingestion.
- **Interactive Dashboard**: Streamlit-based UI for exploring job market trends.
- **Zero Config**: Uses SQLite for instant local deployment.

## Tech Stack

- **Ingestion**: Python, Pandas
- **Database**: SQLite (Local)
- **Analytics**: SQL (CTEs)
- **UI**: Streamlit, Plotly

## Quick Start

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd SQL_endtoend
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Ingestion (First Time Setup)**:
    This will create the database `career_intelligence.db` and populate it with sample data (or your CSV if provided).
    ```bash
    python ingest_data.py
    ```

4.  **Run Dashboard**:
    ```bash
    streamlit run app.py
    ```

## Sentinel Logic

The ingestion script includes a 'Sentinel' module that validates:
-   Null salary checks (alerts if > 50% missing).
-   Duplicate Job ID checks.

## Analytics

The 'Opportunity Score' is calculated by joining job demand with average salary to identify the most valuable skills (Demand * Salary / 1000).
