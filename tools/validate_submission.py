"""Audit the saved HW8 results without training or executing the notebook.

Checks use the provider response, chronological dates, saved predictions and model
weights. A missing or failed result is reported as a failure, never as a pass.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import nbformat
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STEM = "e89-Schwartzberg-Paul-HW08"


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def error_metrics(rows):
    actual = np.asarray([float(row["actual_open"]) for row in rows])
    predicted = np.asarray([float(row["predicted_open"]) for row in rows])
    baseline = np.asarray([float(row["persistence_open"]) for row in rows])
    return {
        "mae_usd": float(np.mean(np.abs(predicted - actual))),
        "rmse_usd": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "baseline_mae_usd": float(np.mean(np.abs(baseline - actual))),
        "baseline_rmse_usd": float(np.sqrt(np.mean((baseline - actual) ** 2))),
    }


class Audit:
    def __init__(self, root):
        self.root = root
        self.out = root / "outputs"
        self.stock = self.out / "stocks"
        self.checks = []

    def check(self, name, condition, detail=""):
        self.checks.append({"check": name, "passed": bool(condition), "detail": str(detail)})

    def section(self, name, function):
        try:
            function()
        except Exception as exc:
            self.check(name + ": evidence can be read", False, f"{type(exc).__name__}: {exc}")

    def load(self, relative):
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    def run(self):
        for name, function in [
            ("Notebook and HTML", self.notebook),
            ("Microsoft data provenance", self.market_data),
            ("Stock split and scalers", self.stock_split),
            ("Twelve stock experiments", self.stock_runs),
            ("Stock metrics and selection", self.stock_tables),
            ("Vocabulary experiment", self.nmt),
        ]:
            self.section(name, function)
        passed = sum(row["passed"] for row in self.checks)
        return {"checked_utc": datetime.now(timezone.utc).isoformat(),
                "notebook": STEM + ".ipynb", "training_performed": False,
                "notebook_execution_performed": False,
                "passed": passed == len(self.checks),
                "checks_passed": passed, "checks_total": len(self.checks),
                "checks": self.checks}

    def notebook(self):
        path = self.root / (STEM + ".ipynb")
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        self.nb = nb
        self.markdown = "\n\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
        self.code = [c for c in nb.cells if c.cell_type == "code"]
        self.sources = "\n\n".join(c.source for c in self.code)
        self.check("Notebook schema is valid", True)
        self.check("Author's name is present", "Schwartzberg, Paul" in self.markdown)
        self.check("All three problem headings are present",
                   all(re.search(rf"(?im)^#+\s+Problem {i}\b", self.markdown) for i in (1, 2, 3)))
        self.check("All code cells contain valid Python", all(ast.parse(c.source) for c in self.code))
        counts = [c.execution_count for c in self.code]
        self.check("All code cells executed in order", counts == list(range(1, len(counts) + 1)) and bool(counts), counts)
        errors = [o.get("ename", "error") for c in self.code for o in c.outputs if o.output_type == "error"]
        self.check("Notebook has no error output", not errors, errors)
        descriptions = []
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                described = any(line.strip().startswith("#") for line in cell.source.splitlines())
                described |= index > 0 and nb.cells[index - 1].cell_type == "markdown" and len(nb.cells[index - 1].source.split()) > 5
                descriptions.append(described)
        self.check("Every code cell has comments or adjacent explanation", all(descriptions))
        images = sum("image/png" in o.get("data", {}) for c in self.code for o in c.outputs)
        self.check("Result figures are embedded", images >= 5, images)
        record = self.load("outputs/execution_log.json")
        self.check("Execution used a fresh kernel", record.get("fresh_kernel_run_all") is True)
        self.check("Execution record matches code-cell count", record.get("code_cells") == len(self.code))
        self.check("Execution duration is positive", record.get("elapsed_seconds", 0) > 0)
        self.check("Notebook and saved execution records match", dict(nb.metadata.get("hw8_execution", {})) == record)
        environment = self.load("outputs/environment.json")
        self.check("Run used a CUDA GPU", str(environment.get("device", "")).startswith("cuda") and bool(environment.get("gpu")))
        exported = path.with_suffix(".html")
        body = exported.read_text(encoding="utf-8")
        self.check("HTML export includes author and all three problems",
                   "Schwartzberg, Paul" in html.unescape(body)
                   and all(f"Problem {i}" in html.unescape(body) for i in (1, 2, 3)))
        html_cells = len(re.findall(r'class="[^\"]*\bjp-CodeCell\b[^\"]*"', body))
        self.check("HTML contains every notebook code cell", html_cells == len(self.code), f"{html_cells} / {len(self.code)}")
        self.check("HTML embeds result figures", body.count("data:image/png;base64,") >= 5)

    def market_data(self):
        meta = self.load("data/MSFT_provenance.json")
        raw_path = self.root / "data/MSFT_yahoo_raw.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))["chart"]["result"][0]
        self.check("Provider response is genuinely for Microsoft", meta["symbol"] == raw["meta"]["symbol"] == "MSFT")
        self.check("Raw provider file matches its checksum", digest(raw_path) == meta["raw_sha256"])
        self.check("Provenance identifies the unadjusted Open field", meta["source_field"] == "chart.result[0].indicators.quote[0].open")
        quotes = raw["indicators"]["quote"][0]["open"]
        provider = {datetime.fromtimestamp(t, ZoneInfo(meta["exchange_timezone"])).date().isoformat(): float(value)
                    for t, value in zip(raw["timestamp"], quotes) if value is not None}
        frames = {}
        for name, info in meta["files"].items():
            path = self.root / "data" / info["filename"]
            rows = csv_rows(path)
            frames[name] = rows
            self.check(f"{name}: CSV checksum matches", digest(path) == info["sha256"])
            self.check(f"{name}: row count and endpoints match provenance",
                       len(rows) == info["rows"] and rows[0]["Date"] == info["first_date"] and rows[-1]["Date"] == info["last_date"])
            dates = [row["Date"] for row in rows]
            self.check(f"{name}: dates are unique and chronological", dates == sorted(set(dates)))
            self.check(f"{name}: every Open matches the raw provider response",
                       all(math.isfinite(float(row["Open"])) and float(row["Open"]) > 0
                           and math.isclose(float(row["Open"]), provider[row["Date"]], abs_tol=1e-7) for row in rows))
        self.market = frames
        development, test, all_rows = frames["development"], frames["test"], frames["all"]
        self.dates = [row["Date"] for row in all_rows]
        self.positions = {date: i for i, date in enumerate(self.dates)}
        self.opens = np.asarray([float(row["Open"]) for row in all_rows], dtype=np.float32)
        self.n_development = len(development)
        self.n_train = math.floor(0.8 * self.n_development)
        self.test_positions = np.asarray([self.positions[row["Date"]] for row in test], dtype=np.int64)
        self.check("Development covers the requested trading-date endpoints",
                   development[0]["Date"] == "2016-01-04" and development[-1]["Date"] == "2026-09-04")
        self.check("Test covers the requested trading-date endpoints",
                   test[0]["Date"] == "2026-09-09" and test[-1]["Date"] == "2026-10-09")
        self.check("Development precedes all test targets", development[-1]["Date"] < test[0]["Date"])
        self.check("Full history contains development unchanged", all_rows[:len(development)] == development)
        gap = [row["Date"] for row in all_rows if development[-1]["Date"] < row["Date"] < test[0]["Date"]]
        self.check("September 8 is retained as input-only gap context", gap == ["2026-09-08"])

    def scaler_checks(self, scaler, stop, label):
        values = self.opens[:stop]
        minimum, maximum = float(values.min()), float(values.max())
        expected_scale = 1.0 / (maximum - minimum)
        self.check(label + ": scaler sees only its declared fitting rows", scaler["n_samples_seen_"] == stop and scaler["n_features_in_"] == 1)
        self.check(label + ": scaler extrema independently match raw fitting prices",
                   np.allclose(scaler["data_min_"], [minimum], atol=1e-6)
                   and np.allclose(scaler["data_max_"], [maximum], atol=1e-6))
        self.check(label + ": scaler scale and offset independently recompute",
                   np.allclose(scaler["scale_"], [expected_scale], rtol=1e-6, atol=1e-8)
                   and np.allclose(scaler["min_"], [-minimum * expected_scale], rtol=1e-6, atol=1e-8))

    def stock_split(self):
        split = self.load("outputs/stocks/split_manifest.json")
        self.stock_manifest = split
        n, cut = self.n_development, self.n_train
        self.check("Problem 2 uses the exact chronological 80/20 row split",
                   split["development_rows"] == n and split["training_rows"] == cut and split["validation_rows"] == n - cut)
        self.check("Recorded training and validation dates are exact chronological slices",
                   split["training_dates"] == self.dates[:cut] and split["validation_dates"] == self.dates[cut:n])
        self.check("Recorded test dates exactly match the requested test file",
                   split["test_dates"] == [r["Date"] for r in self.market["test"]])
        self.check("Gap context is outside every target split",
                   split["context_only_dates"] == ["2026-09-08"]
                   and "2026-09-08" not in split["training_dates"] + split["validation_dates"] + split["test_dates"])
        self.scaler_checks(split["p2_scaler"], cut, "Problem 2")
        self.check("All one-step lookbacks use the same train target origins",
                   split["one_step_train_origins"] == list(range(120, cut)))
        self.check("One-step validation origins start at the validation boundary",
                   split["one_step_validation_origins"] == list(range(cut, n)))
        self.check("One-step test origins include every test trading date",
                   split["one_step_test_origins"] == self.test_positions.tolist())
        five = self.load("outputs/stocks/five_step_window_manifest.json")
        self.check("Five-step training targets never cross the train boundary",
                   five["lookback"] == 60 and five["horizon"] == 5 and five["train_origins"] == list(range(60, cut - 4)))
        self.check("Five-step validation targets never cross the validation boundary",
                   five["validation_origins"] == list(range(cut, n - 4)))
        valid_test = set(self.test_positions.tolist())
        expected = [i for i in self.test_positions.tolist() if all(i + h in valid_test for h in range(5))]
        self.check("All five-step test targets stay inside the test period", five["test_origins"] == expected)
        self.check("Five-step examples have the required input and target shapes",
                   five["train_input_shape"] == [cut - 64, 60, 1]
                   and five["train_target_shape"] == [cut - 64, 5])
        self.five_manifest = five

    def expected_runs(self):
        specs = [(f"p1_{kind.lower()}_60", kind, 60, 1, True) for kind in ("RNN", "LSTM")]
        specs += [(f"p2_{kind.lower()}_{lookback}", kind, lookback, 1, False)
                  for kind in ("RNN", "LSTM") for lookback in (10, 30, 60, 120)]
        specs += [(f"p2_{kind.lower()}_60_five", kind, 60, 5, False) for kind in ("RNN", "LSTM")]
        return specs

    def prediction_checks(self, name, lookback, horizon, split, expected_origins):
        rows = csv_rows(self.stock / f"{name}_{split}_predictions.csv")
        self.check(f"{name} {split}: expected number of origin-horizon pairs",
                   len(rows) == len(expected_origins) * horizon)
        expected_pairs = {(int(i), h + 1) for i in expected_origins for h in range(horizon)}
        actual_pairs = set()
        dates_correct, prices_correct, finite = True, True, True
        for row in rows:
            origin = self.positions[row["origin_date"]] + 1
            h = int(row["horizon"])
            actual_pairs.add((origin, h))
            dates_correct &= (row["input_start_date"] == self.dates[origin - lookback]
                              and row["target_date"] == self.dates[origin + h - 1]
                              and row["origin_date"] < row["target_date"])
            prices_correct &= math.isclose(float(row["actual_open"]), float(self.opens[origin + h - 1]), abs_tol=0.002)
            prices_correct &= math.isclose(float(row["persistence_open"]), float(self.opens[origin - 1]), abs_tol=0.001)
            finite &= math.isfinite(float(row["predicted_open"]))
        self.check(f"{name} {split}: all expected forecasts appear exactly once",
                   actual_pairs == expected_pairs and len(actual_pairs) == len(rows))
        self.check(f"{name} {split}: input dates precede every target and match lookback", dates_correct)
        self.check(f"{name} {split}: targets and persistence independently match raw prices", prices_correct)
        self.check(f"{name} {split}: every model forecast is finite", finite)
        return rows

    def stock_runs(self):
        import torch
        self.stock_predictions = {}
        self.stock_records = {}
        specs = self.expected_runs()
        self.check("Exactly twelve stock run records are saved", {p.stem[:-4] for p in self.stock.glob("*_run.json")} == {s[0] for s in specs})
        for name, kind, lookback, horizon, p1 in specs:
            report = self.load(f"outputs/stocks/{name}_run.json")
            history = self.load(f"outputs/stocks/{name}_history.json")
            checkpoint = torch.load(self.stock / report["checkpoint"], map_location="cpu", weights_only=True)
            self.stock_records[name] = report
            self.check(name + ": supplied recurrent architecture and settings retained",
                       checkpoint["model_config"] == {"kind": kind, "horizon": horizon, "hidden_size": 50, "dropout": 0.2}
                       and report["epochs"] == checkpoint["epochs"] == 100
                       and checkpoint["batch_size"] == 32 and checkpoint["learning_rate"] == 0.001)
            self.check(name + ": input and target shapes match the experiment",
                       report["input_shape"] == [lookback, 1] and report["target_shape"] == [horizon])
            self.check(name + ": all one hundred training epochs completed",
                       [row["epoch"] for row in history] == list(range(1, 101))
                       and all(math.isfinite(row["train_mse_scaled"]) and row["seconds"] > 0 for row in history))
            fit_stop = self.n_development if p1 else self.n_train
            self.scaler_checks(checkpoint["scaler"], fit_stop, name)
            train_expected = (list(range(60, self.n_development)) if p1 else
                              list(range(120, self.n_train)) if horizon == 1 else list(range(60, self.n_train - 4)))
            val_expected = [] if p1 else (list(range(self.n_train, self.n_development)) if horizon == 1 else list(range(self.n_train, self.n_development - 4)))
            self.check(name + ": exact chronological training/validation origins are saved",
                       report["train_origins"] == checkpoint["training_origins"] == train_expected
                       and report["validation_origins"] == checkpoint["validation_origins"] == val_expected)
            if p1:
                selected, loss = history[-1]["epoch"], history[-1]["train_mse_scaled"]
                self.check(name + ": baseline has no validation fitting", all(row["val_mse_scaled"] is None for row in history))
            else:
                selected_row = min(history, key=lambda row: row["val_mse_scaled"])
                selected, loss = selected_row["epoch"], selected_row["val_mse_scaled"]
                self.check(name + ": validation losses are finite", all(math.isfinite(row["val_mse_scaled"]) for row in history))
            self.check(name + ": checkpoint epoch follows the stated selection rule",
                       report["selected_epoch"] == checkpoint["selected_epoch"] == selected
                       and math.isclose(report["selection_loss_scaled"], loss, abs_tol=1e-10))
            state = checkpoint["model_state_dict"]
            expected_parameters = (18001 if kind == "RNN" else 71851) + (horizon - 1) * 51
            self.check(name + ": checkpoint holds the expected finite CPU parameter tensors",
                       sum(v.numel() for v in state.values()) == expected_parameters
                       and all(v.device.type == "cpu" and bool(torch.isfinite(v).all()) for v in state.values()))
            self.check(name + ": model and scaler reload checks passed", report["reload_verified"] is True)
            test_origins = self.test_positions.tolist() if horizon == 1 else self.five_manifest["test_origins"]
            self.stock_predictions[(name, "test")] = self.prediction_checks(name, lookback, horizon, "test", test_origins)
            if not p1:
                self.stock_predictions[(name, "validation")] = self.prediction_checks(name, lookback, horizon, "validation", val_expected)

    def check_metrics(self, report, predictions, prefix, label):
        metrics = error_metrics(predictions)
        self.check(label + ": dollar errors independently recompute",
                   all(math.isclose(float(report[prefix + key]), value, rel_tol=1e-7, abs_tol=1e-5)
                       for key, value in metrics.items()))

    def stock_tables(self):
        p1 = csv_rows(self.stock / "problem1_metrics.csv")
        self.check("Problem 1 reports both recurrent models", {row["model"] for row in p1} == {"RNN", "LSTM"})
        for row in p1:
            self.check_metrics(row, self.stock_predictions[(f"p1_{row['model'].lower()}_60", "test")], "", "Problem 1 " + row["model"])
        p2 = csv_rows(self.stock / "problem2_lookback_metrics.csv")
        self.check("Problem 2 reports all eight lookback experiments",
                   len(p2) == 8 and {(r["model"], int(r["lookback"])) for r in p2} == {(m, k) for m in ("RNN", "LSTM") for k in (10, 30, 60, 120)})
        selection = self.load("outputs/stocks/lookback_selection.json")
        self.check("Lookback selection excludes test results", selection["test_results_used"] is False and selection["selection_metric"] == "validation RMSE in USD")
        for row in p2:
            for split in ("validation", "test"):
                self.check_metrics(row, self.stock_predictions[(row["run_name"], split)], split + "_", row["run_name"] + " " + split)
        for kind in ("RNN", "LSTM"):
            best = min((r for r in p2 if r["model"] == kind), key=lambda r: (float(r["validation_rmse_usd"]), int(r["lookback"])))
            selected_rows = [r for r in p2 if r["model"] == kind and str(r["selected_by_validation"]).lower() == "true"]
            self.check(kind + ": selected lookback is the validation minimum",
                       selection["selected_lookbacks"][kind] == int(best["lookback"])
                       and len(selected_rows) == 1 and selected_rows[0]["run_name"] == best["run_name"])
        five = csv_rows(self.stock / "problem2_five_step_metrics.csv")
        self.check("Both models report direct five-step results", len(five) == 2 and {r["model"] for r in five} == {"RNN", "LSTM"})
        for row in five:
            name = f"p2_{row['model'].lower()}_60_five"
            for split in ("validation", "test"):
                self.check_metrics(row, self.stock_predictions[(name, split)], split + "_", name + " " + split)
        horizons = csv_rows(self.stock / "problem2_horizon_metrics.csv")
        self.check("Five-step errors cover all horizons for both models", len(horizons) == 10 and {(r["model"], int(r["horizon"])) for r in horizons} == {(m, h) for m in ("RNN", "LSTM") for h in range(1, 6)})
        for row in horizons:
            predictions = [r for r in self.stock_predictions[(f"p2_{row['model'].lower()}_60_five", "test")] if r["horizon"] == row["horizon"]]
            self.check_metrics(row, predictions, "", row["model"] + " horizon " + row["horizon"])

    def nmt(self):
        raise FileNotFoundError("NMT artifact schema is still being prepared; this check cannot pass yet.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report = Audit(root).run()
    destination = args.report or root / "outputs/submission_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['checks_passed']} of {report['checks_total']} checks passed.")
    for row in report["checks"]:
        if not row["passed"]:
            print("FAIL:", row["check"], row["detail"])
    print("Report:", destination)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
