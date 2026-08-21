# Model results

All final metrics use the untouched 2025 temporal holdout.

|                     |   roc_auc |   average_precision |   log_loss |   brier_score |   base_rate |   top_decile_delay_rate |   top_decile_lift |   share_delays_captured_top_decile |   fit_minutes |
|:--------------------|----------:|--------------------:|-----------:|--------------:|------------:|------------------------:|------------------:|-----------------------------------:|--------------:|
| xgboost             |    0.6781 |              0.3714 |     0.4962 |        0.1613 |      0.2235 |                  0.4515 |            2.0205 |                             0.202  |        1.6574 |
| ebm                 |    0.6774 |              0.3712 |     0.4965 |        0.1613 |      0.2235 |                  0.451  |            2.0185 |                             0.2018 |       12.1366 |
| logistic_regression |    0.6636 |              0.3567 |     0.5087 |        0.1647 |      0.2235 |                  0.4324 |            1.9349 |                             0.1935 |        0.6245 |

## Feature-block screening

| feature_group             |   validation_ap_without_group |   validation_ap_gain |   validation_auc_gain |   paired_log_loss_gain |   loss_gain_ci_low |   loss_gain_ci_high | included   | decision                              |
|:--------------------------|------------------------------:|---------------------:|----------------------:|-----------------------:|-------------------:|--------------------:|:-----------|:--------------------------------------|
| Flight identity           |                    nan        |           nan        |            nan        |             nan        |         nan        |          nan        | True       | Required task definition              |
| Schedule timing           |                    nan        |           nan        |            nan        |             nan        |         nan        |          nan        | True       | Required task definition              |
| Scheduled congestion      |                      0.357711 |             0.001602 |              0.001245 |               0.000463 |           0.000363 |            0.000563 | True       | Included                              |
| Federal holiday calendar  |                      0.357963 |             0.001349 |              0.000728 |               0.000314 |           0.000219 |            0.000409 | True       | Included                              |
| FAA airport class         |                      0.358301 |             0.001011 |              0.000759 |               0.000337 |           0.000248 |            0.000425 | True       | Included                              |
| NWS forecast hazards      |                      0.353954 |             0.005358 |              0.003939 |               0.001541 |           0.001386 |            0.001696 | True       | Included                              |
| Lagged network centrality |                      0.359435 |            -0.000123 |              0.000174 |               2.1e-05  |          -8.1e-05  |            0.000124 | False      | Excluded: no material validation lift |
| Recent delay propagation  |                      0.338387 |             0.020925 |              0.015852 |               0.006263 |           0.005915 |            0.006611 | True       | Included                              |

## Dataset summary

|   year |     flights |   eligible_rate |   cancellation_rate |   diversion_rate |   delay_rate |
|-------:|------------:|----------------:|--------------------:|-----------------:|-------------:|
|   2022 | 6.72912e+06 |          0.9707 |              0.0269 |           0.0024 |       0.2108 |
|   2023 | 6.8479e+06  |          0.9847 |              0.0128 |           0.0024 |       0.2056 |
|   2024 | 7.07906e+06 |          0.9839 |              0.0136 |           0.0025 |       0.2082 |
|   2025 | 7.00162e+06 |          0.9826 |              0.0147 |           0.0028 |       0.2231 |

## Delay-severity thresholds

|   delay_threshold_minutes |   base_rate |   roc_auc |   average_precision |   log_loss |   brier_score |   fit_minutes |
|--------------------------:|------------:|----------:|--------------------:|-----------:|--------------:|--------------:|
|                        30 |      0.1487 |    0.6845 |              0.2733 |     0.393  |        0.1193 |        4.8932 |
|                        60 |      0.0821 |    0.6878 |              0.1671 |     0.2671 |        0.0725 |        4.3213 |
|                       120 |      0.0338 |    0.6825 |              0.0731 |     0.1412 |        0.0322 |        2.7635 |

## Multicollinearity audit

Maximum numeric VIF: 1.71; maximum absolute numeric Spearman correlation: 0.565.

| feature                             | conditioned_within   |   baseline_average_precision |   conditional_permutation_average_precision |   average_precision_drop_basis_points |   log_loss_increase | interpretation                            |
|:------------------------------------|:---------------------|-----------------------------:|--------------------------------------------:|--------------------------------------:|--------------------:|:------------------------------------------|
| origin_hourly_scheduled_flights     | origin_airport       |                     0.368854 |                                    0.367297 |                               15.5721 |            0.00056  | Retains within-airport incremental signal |
| destination_daily_scheduled_flights | destination_airport  |                     0.368854 |                                    0.367105 |                               17.4895 |            0.000637 | Retains within-airport incremental signal |

## EBM response-function review

| feature                                   | display_name                                        |   bins_or_levels |   roughness_ratio |   direction_reversals |   spearman_x_to_effect | production_recommendation                                                                       |
|:------------------------------------------|:----------------------------------------------------|-----------------:|------------------:|----------------------:|-----------------------:|:------------------------------------------------------------------------------------------------|
| destination_daily_scheduled_flights       | Destination flights scheduled that day              |              254 |            1.784  |                     2 |                 0.9978 | Smooth/coarsen; avoid a hard monotone constraint because hub identity creates plateaus          |
| destination_recent_delay_rate             | Recent destination delay rate (days -8 to -2)       |              254 |            1.4079 |                    20 |                 0.9855 | Apply a non-decreasing monotonic constraint                                                     |
| origin_recent_delay_rate                  | Recent origin delay rate (days -8 to -2)            |              254 |            1.3964 |                    19 |                 0.9953 | Apply a non-decreasing monotonic constraint                                                     |
| scheduled_duration_minutes                | Scheduled gate-to-gate duration (minutes)           |              249 |            1.1592 |                    11 |                 0.2882 | Smooth with ~15-minute bins or max_bins~=64; do not impose global monotonicity                  |
| origin_hourly_scheduled_flights           | Origin flights scheduled in the same hour           |               88 |            1.0952 |                     1 |                 0.9934 | Consider weak non-decreasing constraint after checking within-airport calibration               |
| route_recent_delay_rate                   | Recent route delay rate (days -29 to -2)            |              254 |            1.0198 |                     3 |                 0.9995 | Apply a non-decreasing monotonic constraint                                                     |
| scheduled_departure_hour                  | Scheduled departure hour                            |               22 |            0.8992 |                     2 |                 0.8001 | Keep unconstrained; adjacent-hour smoothing is acceptable but the daily pattern is non-monotone |
| within_three_days_of_federal_holiday      | Within three days of a federal holiday              |                2 |            0      |                     0 |               nan      | Binary indicator: no smoothing or monotonic constraint needed                                   |
| origin_nws_hazard_valid_at_departure      | Origin NWS hazard valid at departure                |                2 |            0      |                     0 |               nan      | Binary indicator: no smoothing or monotonic constraint needed                                   |
| destination_nws_hazard_valid_at_departure | Destination NWS hazard valid at scheduled departure |                2 |            0      |                     0 |               nan      | Binary indicator: no smoothing or monotonic constraint needed                                   |
| origin_nws_hazard_active_at_cutoff        | Origin NWS hazard active at prediction time         |                2 |            0      |                     0 |               nan      | Binary indicator: no smoothing or monotonic constraint needed                                   |
| destination_nws_hazard_active_at_cutoff   | Destination NWS hazard active at prediction time    |                2 |            0      |                     0 |               nan      | Binary indicator: no smoothing or monotonic constraint needed                                   |
| origin_nws_winter_hazard                  | Origin winter hazard forecast                       |                2 |            0      |                     0 |               nan      | Binary indicator: no smoothing or monotonic constraint needed                                   |
| destination_nws_winter_hazard             | Destination winter hazard forecast                  |                2 |            0      |                     0 |               nan      | Binary indicator: no smoothing or monotonic constraint needed                                   |
