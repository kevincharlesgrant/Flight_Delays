from __future__ import annotations

"""Research-only airport GraphSAGE prototype; intentionally not used by the notebook."""

import math
import os
import sys
import time
from pathlib import Path

# A short optional target path avoids the Windows MAX_PATH issue in large wheels.
local_torch = Path(os.environ.get("FLIGHT_DELAYS_TORCH_PATH", r"C:\Users\kevin\Documents\Codex\torch_cpu"))
if local_torch.exists():
    sys.path.insert(0, str(local_torch))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from common import INTERIM, MODELS, REPORTS, ensure_dirs


TARGET = "arrival_delay_15min"
NUMERIC = [
    "scheduled_duration_minutes", "origin_hourly_scheduled_flights",
    "destination_daily_scheduled_flights", "within_three_days_of_federal_holiday",
    "origin_nws_hazard_valid_at_departure", "destination_nws_hazard_valid_at_departure",
    "origin_nws_hazard_active_at_cutoff", "destination_nws_hazard_active_at_cutoff",
    "origin_nws_convective_hazard", "destination_nws_convective_hazard",
    "origin_nws_winter_hazard", "destination_nws_winter_hazard",
    "origin_recent_delay_rate", "destination_recent_delay_rate", "route_recent_delay_rate",
]


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    cutoff = np.quantile(p, 0.9)
    top = p >= cutoff
    base = y.mean()
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "top_decile_delay_rate": float(y[top].mean()),
        "top_decile_lift": float(y[top].mean() / base),
    }


def paired_loss_gain(y: np.ndarray, reference: np.ndarray, alternative: np.ndarray):
    eps = 1e-7
    reference, alternative = np.clip(reference, eps, 1 - eps), np.clip(alternative, eps, 1 - eps)
    ref_loss = -(y * np.log(reference) + (1 - y) * np.log(1 - reference))
    alt_loss = -(y * np.log(alternative) + (1 - y) * np.log(1 - alternative))
    delta = alt_loss - ref_loss
    mean = float(delta.mean())
    half = 1.96 * float(delta.std(ddof=1) / math.sqrt(len(delta)))
    return mean, mean - half, mean + half


def mappings(train: pd.DataFrame, test: pd.DataFrame):
    airports = sorted(set(train.origin_airport) | set(train.destination_airport)
                      | set(test.origin_airport) | set(test.destination_airport))
    airport_map = {value: index for index, value in enumerate(airports)}
    airline_values = sorted(set(train.airline.astype(str)))
    airline_map = {value: index + 1 for index, value in enumerate(airline_values)}
    return airport_map, airline_map


def graph_data(frame: pd.DataFrame, airport_map: dict[str, int]):
    n = len(airport_map)
    edge_counts = frame.groupby(["origin_airport", "destination_airport"]).size().reset_index(name="weight")
    src = edge_counts.origin_airport.map(airport_map).to_numpy()
    dst = edge_counts.destination_airport.map(airport_map).to_numpy()
    weight = edge_counts.weight.to_numpy(dtype=np.float32)

    def normalized_sparse(row, col, values):
        totals = np.bincount(row, weights=values, minlength=n).astype(np.float32)
        normalized = values / np.maximum(totals[row], 1.0)
        indices = torch.tensor(np.vstack([row, col]), dtype=torch.long)
        return torch.sparse_coo_tensor(indices, torch.tensor(normalized), (n, n)).coalesce()

    out_adj = normalized_sparse(src, dst, weight)
    in_adj = normalized_sparse(dst, src, weight)

    records = []
    for side, airport_col, other_col, prefix in [
        ("origin", "origin_airport", "destination_airport", "out"),
        ("destination", "destination_airport", "origin_airport", "in"),
    ]:
        agg = frame.groupby(airport_col).agg(
            flights=(TARGET, "size"), delay_rate=(TARGET, "mean"),
            unique_neighbors=(other_col, "nunique"),
            mean_duration=("scheduled_duration_minutes", "mean"),
        )
        agg.columns = [f"{prefix}_{c}" for c in agg.columns]
        records.append(agg)
    node = records[0].join(records[1], how="outer").reindex(airport_map.keys())
    for c in ["out_flights", "in_flights"]:
        node[c] = np.log1p(node[c])
    node = node.fillna(node.median(numeric_only=True)).fillna(0)
    values = node.to_numpy(dtype=np.float32)
    values = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-6)
    return torch.tensor(values, dtype=torch.float32), out_adj, in_adj


def numeric_scaler(frame: pd.DataFrame):
    values = frame[NUMERIC].apply(pd.to_numeric, errors="coerce")
    median = values.median()
    values = values.fillna(median)
    mean, std = values.mean(), values.std().replace(0, 1)
    return median, mean, std


def encode_frame(frame: pd.DataFrame, airport_map, airline_map, scaler):
    median, mean, std = scaler
    numeric = frame[NUMERIC].apply(pd.to_numeric, errors="coerce").fillna(median)
    numeric = ((numeric - mean) / std).to_numpy(dtype=np.float32)
    return {
        "origin": torch.tensor(frame.origin_airport.map(airport_map).to_numpy(), dtype=torch.long),
        "destination": torch.tensor(frame.destination_airport.map(airport_map).to_numpy(), dtype=torch.long),
        "airline": torch.tensor(frame.airline.astype(str).map(airline_map).fillna(0).to_numpy(), dtype=torch.long),
        "month": torch.tensor(pd.to_numeric(frame.departure_month).astype(int).to_numpy() - 1, dtype=torch.long),
        "weekday": torch.tensor(pd.to_numeric(frame.departure_weekday).astype(int).to_numpy(), dtype=torch.long),
        "hour": torch.tensor(pd.to_numeric(frame.scheduled_departure_hour).astype(int).to_numpy(), dtype=torch.long),
        "numeric": torch.tensor(numeric, dtype=torch.float32),
        "target": torch.tensor(frame[TARGET].to_numpy(dtype=np.float32)),
    }


class EdgeModel(torch.nn.Module):
    def __init__(self, n_nodes, node_features, n_airlines, use_message_passing=True, hidden=32):
        super().__init__()
        self.use_message_passing = use_message_passing
        self.node_identity = torch.nn.Embedding(n_nodes, 16)
        input_dim = 16 + node_features
        if use_message_passing:
            self.self_1, self.out_1, self.in_1 = [torch.nn.Linear(input_dim, hidden) for _ in range(3)]
            self.self_2, self.out_2, self.in_2 = [torch.nn.Linear(hidden, hidden) for _ in range(3)]
        else:
            self.node_project = torch.nn.Linear(input_dim, hidden)
        self.airline = torch.nn.Embedding(n_airlines + 1, 8)
        self.month = torch.nn.Embedding(12, 4)
        self.weekday = torch.nn.Embedding(7, 3)
        self.hour = torch.nn.Embedding(24, 5)
        decoder_dim = hidden * 4 + 8 + 4 + 3 + 5 + len(NUMERIC)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(decoder_dim, 96), torch.nn.ReLU(), torch.nn.Dropout(0.12),
            torch.nn.Linear(96, 48), torch.nn.ReLU(), torch.nn.Dropout(0.08),
            torch.nn.Linear(48, 1),
        )

    def nodes(self, node_x, out_adj, in_adj):
        base = torch.cat([self.node_identity.weight, node_x], dim=1)
        if not self.use_message_passing:
            return torch.relu(self.node_project(base))
        out_message, in_message = torch.sparse.mm(out_adj, base), torch.sparse.mm(in_adj, base)
        hidden = torch.relu(self.self_1(base) + self.out_1(out_message) + self.in_1(in_message))
        out_message, in_message = torch.sparse.mm(out_adj, hidden), torch.sparse.mm(in_adj, hidden)
        return torch.relu(self.self_2(hidden) + self.out_2(out_message) + self.in_2(in_message))

    def forward(self, batch, node_x, out_adj, in_adj):
        nodes = self.nodes(node_x, out_adj, in_adj)
        origin, destination = nodes[batch["origin"]], nodes[batch["destination"]]
        parts = [origin, destination, origin * destination, torch.abs(origin - destination),
                 self.airline(batch["airline"]), self.month(batch["month"]),
                 self.weekday(batch["weekday"]), self.hour(batch["hour"]), batch["numeric"]]
        return self.decoder(torch.cat(parts, dim=1)).squeeze(1)


def subset_tensors(data, index):
    return {key: value[index] for key, value in data.items()}


@torch.no_grad()
def predict(model, data, node_x, out_adj, in_adj, batch_size=32768):
    model.eval()
    probabilities = []
    for start in range(0, len(data["target"]), batch_size):
        index = slice(start, min(start + batch_size, len(data["target"])))
        logits = model(subset_tensors(data, index), node_x, out_adj, in_adj)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def fit_model(model, train, valid, node_x, out_adj, in_adj, seed, epochs=8):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=2e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_state, best_ap, best_epoch, patience = None, -np.inf, 0, 0
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train["target"]), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), 8192):
            index = order[start:start + 8192]
            optimizer.zero_grad(set_to_none=True)
            logits = model(subset_tensors(train, index), node_x, out_adj, in_adj)
            loss = criterion(logits, train["target"][index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss) * len(index)
        valid_probability = predict(model, valid, node_x, out_adj, in_adj)
        ap = average_precision_score(valid["target"].numpy(), valid_probability)
        print(f"epoch {epoch + 1}: loss={total_loss / len(order):.5f}, validation AP={ap:.6f}", flush=True)
        if ap > best_ap + 1e-5:
            best_ap = ap
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    model.load_state_dict(best_state)
    return model, best_ap, best_epoch


def fit_fixed(model, train, node_x, out_adj, in_adj, seed, epochs):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=2e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train["target"]), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), 8192):
            index = order[start:start + 8192]
            optimizer.zero_grad(set_to_none=True)
            logits = model(subset_tensors(train, index), node_x, out_adj, in_adj)
            loss = criterion(logits, train["target"][index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss) * len(index)
        print(f"fixed epoch {epoch + 1}/{epochs}: loss={total_loss / len(order):.5f}", flush=True)
    return model


def main():
    ensure_dirs()
    torch.set_num_threads(max(1, min(12, os.cpu_count() or 4)))
    full_train = pd.read_parquet(INTERIM / "model_train_enriched.parquet")
    full_test = pd.read_parquet(INTERIM / "model_test_enriched.parquet")
    primary_predictions = pd.read_parquet(REPORTS / "test_predictions.parquet")
    for frame in (full_train, full_test):
        for column in ["airline", "origin_airport", "destination_airport"]:
            frame[column] = frame[column].fillna("Missing").astype(str)

    development = full_train[full_train.year.le(2023)].sample(n=800_000, random_state=71)
    validation = full_train[full_train.year.eq(2024)].sample(n=300_000, random_state=72)
    airport_map, airline_map = mappings(development, validation)
    node_x, out_adj, in_adj = graph_data(development, airport_map)
    scaler = numeric_scaler(development)
    dev_tensors = encode_frame(development, airport_map, airline_map, scaler)
    val_tensors = encode_frame(validation, airport_map, airline_map, scaler)
    validation_results, epoch_choices = [], {}
    for name, message_passing in [("embedding_mlp", False), ("graphsage", True)]:
        model = EdgeModel(len(airport_map), node_x.shape[1], len(airline_map), message_passing)
        started = time.time()
        model, best_ap, best_epoch = fit_model(model, dev_tensors, val_tensors, node_x, out_adj, in_adj, 42, epochs=8)
        probability = predict(model, val_tensors, node_x, out_adj, in_adj)
        validation_results.append({"model": name, **evaluate(val_tensors["target"].numpy(), probability),
                                   "fit_minutes": (time.time() - started) / 60, "best_validation_ap": best_ap,
                                   "selected_epochs": best_epoch})
        epoch_choices[name] = best_epoch
    pd.DataFrame(validation_results).to_csv(REPORTS / "gnn_validation_results.csv", index=False)

    train = full_train.sample(n=1_500_000, random_state=73)
    test_index = full_test.sample(n=500_000, random_state=74).index
    test = full_test.loc[test_index]
    airport_map, airline_map = mappings(train, test)
    node_x, out_adj, in_adj = graph_data(train, airport_map)
    scaler = numeric_scaler(train)
    train_tensors = encode_frame(train, airport_map, airline_map, scaler)
    test_tensors = encode_frame(test, airport_map, airline_map, scaler)
    y_test = test_tensors["target"].numpy()
    predictions = pd.DataFrame({"actual": y_test}, index=test_index)
    result_rows = []
    fitted = {}
    for name, message_passing in [("embedding_mlp", False), ("graphsage", True)]:
        seed_predictions = []
        fit_seconds = 0.0
        for seed in [42, 43, 44]:
            model = EdgeModel(len(airport_map), node_x.shape[1], len(airline_map), message_passing)
            started = time.time()
            model = fit_fixed(model, train_tensors, node_x, out_adj, in_adj, seed, epoch_choices[name])
            fit_seconds += time.time() - started
            seed_predictions.append(predict(model, test_tensors, node_x, out_adj, in_adj))
            fitted[(name, seed)] = model.state_dict()
        probability = np.mean(seed_predictions, axis=0)
        predictions[name] = probability
        result_rows.append({"model": name, **evaluate(y_test, probability), "fit_minutes": fit_seconds / 60})

    for name in ["logistic_regression", "xgboost", "ebm"]:
        probability = primary_predictions.loc[test_index, name].to_numpy()
        predictions[name] = probability
        result_rows.append({"model": name, **evaluate(y_test, probability), "fit_minutes": np.nan})
    results = pd.DataFrame(result_rows).sort_values("average_precision", ascending=False)
    results.to_csv(REPORTS / "gnn_test_results.csv", index=False)
    predictions.to_parquet(REPORTS / "gnn_test_predictions.parquet", index=True)

    comparison_rows = []
    for alternative in ["embedding_mlp", "xgboost", "ebm"]:
        gain, low, high = paired_loss_gain(y_test, predictions.graphsage.to_numpy(), predictions[alternative].to_numpy())
        comparison_rows.append({"reference": "graphsage", "alternative": alternative,
                                "graphsage_log_loss_gain": gain, "ci_low": low, "ci_high": high})
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(REPORTS / "gnn_paired_comparisons.csv", index=False)
    torch.save({"states": fitted, "airport_map": airport_map, "airline_map": airline_map,
                "numeric_features": NUMERIC}, MODELS / "gnn_prototype.pt")

    report = "# GNN prototype results\n\n"
    report += "Research-only edge-classification experiment; not included in the case-study notebook.\n\n"
    report += results.round(6).to_markdown(index=False)
    report += "\n\n## Paired log-loss comparisons\n\n" + comparisons.round(7).to_markdown(index=False) + "\n"
    (REPORTS / "gnn_prototype_results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
