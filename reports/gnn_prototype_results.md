# GNN prototype results

Research-only edge-classification experiment; not included in the case-study notebook.

| model               |   roc_auc |   average_precision |   log_loss |   brier_score |   top_decile_delay_rate |   top_decile_lift |   fit_minutes |
|:--------------------|----------:|--------------------:|-----------:|--------------:|------------------------:|------------------:|--------------:|
| embedding_mlp       |  0.681372 |            0.376015 |   0.495527 |      0.161038 |                 0.45588 |           2.03525 |       2.00865 |
| graphsage           |  0.680741 |            0.37601  |   0.49578  |      0.161104 |                 0.45886 |           2.04855 |       2.23209 |
| xgboost             |  0.678933 |            0.372119 |   0.496664 |      0.161459 |                 0.45226 |           2.01909 |     nan       |
| ebm                 |  0.677755 |            0.371466 |   0.49709  |      0.161597 |                 0.45104 |           2.01364 |     nan       |
| logistic_regression |  0.663225 |            0.357014 |   0.50889  |      0.164893 |                 0.43162 |           1.92694 |     nan       |

## Paired log-loss comparisons

| reference   | alternative   |   graphsage_log_loss_gain |     ci_low |    ci_high |
|:------------|:--------------|--------------------------:|-----------:|-----------:|
| graphsage   | embedding_mlp |                -0.0002527 | -0.0003799 | -0.0001256 |
| graphsage   | xgboost       |                 0.0008843 |  0.0006844 |  0.0010841 |
| graphsage   | ebm           |                 0.0013102 |  0.0011066 |  0.0015138 |
