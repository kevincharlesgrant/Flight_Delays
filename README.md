# U.S. Flight Delay Risk

Flight-delay risk modeling analysis using BTS Reporting Carrier On-Time Performance records, archived NOAA/NWS forecast hazard products, the U.S. federal holiday calendar, and FAA airport hub classifications.

## Question

Can information available 24 hours before scheduled departure identify flights at elevated risk of arriving at least 15 minutes late?

The label is arrival delay of at least 15 minutes among completed, non-diverted flights. Cancellations and diversions remain in the processed store for sensitivity analysis but are not silently treated as ordinary missing arrival delays.

## Experimental design

- Development years: 2022–2023
- Feature selection and hyperparameter validation: 2024
- Untouched final temporal test: 2025
- Candidate models: L2-regularized logistic regression, Explainable Boosting Machine (EBM), and XGBoost
- Prediction cutoff: 24 hours before scheduled departure
- Leakage guardrail: actual departure time, departure delay, taxi time, reported delay causes, and post-cutoff observed weather are excluded

Features use presentation-friendly names. For example, the raw BTS field `CRSElapsedTime` is exposed as `scheduled_duration_minutes`. The redundant day-of-year sine/cosine fields were removed; month and weekday are modeled directly.

## Additional features

- NWS products issued by the cutoff and valid at the origin or destination at scheduled departure time
- Federal holiday-window flag
- FAA 2023 origin and destination hub class
- Prior-month schedule-network PageRank and betweenness (tested, then excluded when the block did not improve 2024 validation)
- Recent origin, destination, and route delay rates ending two days before the flight (retained as leakage-safe propagation features)
- Secondary EBM severity endpoints at 30, 60, and 120 minutes, with monotone probability reconciliation
- Mixed-type multicollinearity screening, including within-airport conditional permutation tests for congestion features
- Production review of EBM response functions with smoothing and monotonic-constraint recommendations
- Explicit feature-completeness audit: unknown FAA hub classes become a categorical level; XGBoost and EBM retain native numeric missing-value handling; logistic regression uses missing-history indicators and time-safe hierarchical backoffs for sparse propagation-rate cold starts

Candidate feature blocks must add at least 0.0010 absolute average precision (10 basis points) on 2024 validation to enter the final models.

## Setup and pipeline

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_pipeline.ps1
```

Run these commands from the project root. Activation is optional because the pipeline calls the environment's interpreter directly.

The explicit sequence is:

```powershell
.\.venv\Scripts\python.exe src\download_bts.py
.\.venv\Scripts\python.exe src\download_nws_hazards.py
.\.venv\Scripts\python.exe src\download_faa.py
.\.venv\Scripts\python.exe src\build_dataset.py
.\.venv\Scripts\python.exe src\sample_model_data.py
.\.venv\Scripts\python.exe src\eda.py
.\.venv\Scripts\python.exe src\enrich_features.py
.\.venv\Scripts\python.exe src\train_models.py
.\.venv\Scripts\python.exe src\missingness_audit.py
.\.venv\Scripts\python.exe src\train_delay_thresholds.py
.\.venv\Scripts\python.exe src\multicollinearity_and_shapes.py
.\.venv\Scripts\python.exe src\make_report.py
```

Downloads and large intermediate files are resumable and ignored by Git.

The notebook is the presentation artifact. Modeling and feature engineering live in `src/`; `run_pipeline.ps1` is the single reproducibility entry point. The GraphSAGE prototype is research-only and is not part of the main pipeline.

## Data provenance

- [BTS Reporting Carrier On-Time Performance monthly archives](https://www.transtats.bts.gov/)
- [NOAA/NWS VTEC products archived by Iowa State IEM](https://mesonet.agron.iastate.edu/request/gis/watchwarn.phtml)
- [FAA CY2023 all-airport enplanements and hub classifications](https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/cy23_all_enplanements)
- `holidays` Python package for the U.S. federal holiday calendar
- `airportsdata` package for airport coordinates and time zones

The IEM archive is used because it exposes historical NWS issuance and expiration times plus warning geometries in a reproducible bulk format.
