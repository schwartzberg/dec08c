"""Self-contained notebook sections for the stock-price experiments."""
from textwrap import dedent, indent
import re
import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


def make_cells():
    cells = [
        md("""
        ## Problem 1. Predict Microsoft's opening prices

        I use Microsoft (`MSFT`) in place of Apple.<br>
        The development period starts in January 2016 and ends on September 6th 2026.<br>
        The test period runs from September 9th to October 10th 2026.<br>
        These are calendar limits; the tables below show the trading dates actually present in the downloaded data.

        Both supplied notebooks use PyTorch.<br>
        I retain their four recurrent layers, 50 hidden units per layer, dropout of 0.20, Adam learning rate of 0.001, batch size of 32 and 100 training epochs.<br>
        The RNN also retains gradient clipping at a norm of 1.0.<br>
        The first experiment uses the original 60-observation input and predicts one opening price.<br>
        I save the learned weights with `torch.save`, the PyTorch equivalent of saving a trained Keras model, and check the restored predictions.
        """),
        code(r'''
        # Load the two requested periods and retain any observed gap day as context.
        from sklearn.preprocessing import MinMaxScaler
        from torch.utils.data import TensorDataset, DataLoader
        import time
        import json
        import gc

        STOCK_DIR = OUTPUT_DIR / "stocks"
        STOCK_DIR.mkdir(parents=True, exist_ok=True)
        STOCK_EPOCHS = 100
        STOCK_BATCH_SIZE = 32
        STOCK_LEARNING_RATE = 0.001
        # Enable only to recover completed runs after an interrupted execution.
        # A fresh, complete submission run leaves this False.
        STOCK_REUSE_COMPLETED = False

        def load_stock_prices(path):
            frame = pd.read_csv(path)
            assert {"Date", "Open"}.issubset(frame.columns), path
            frame = frame[["Date", "Open"]].copy()
            frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
            frame["Open"] = pd.to_numeric(frame["Open"], errors="raise")
            frame = frame.sort_values("Date").reset_index(drop=True)
            assert len(frame) and not frame["Date"].duplicated().any()
            assert np.isfinite(frame["Open"]).all() and (frame["Open"] > 0).all()
            return frame

        stock_development = load_stock_prices(DATA_DIR / "MSFT_development.csv")
        stock_test = load_stock_prices(DATA_DIR / "MSFT_test.csv")
        if (DATA_DIR / "MSFT_all.csv").exists():
            stock_all = load_stock_prices(DATA_DIR / "MSFT_all.csv")
            stock_all = stock_all.loc[
                stock_all["Date"].between(stock_development["Date"].min(), stock_test["Date"].max())
            ].reset_index(drop=True)
        else:
            stock_all = pd.concat([stock_development, stock_test], ignore_index=True)
        assert stock_development["Date"].min() >= pd.Timestamp("2016-01-01")
        assert stock_development["Date"].max() <= pd.Timestamp("2026-09-06")
        assert stock_test["Date"].min() >= pd.Timestamp("2026-09-09")
        assert stock_test["Date"].max() <= pd.Timestamp("2026-10-10")
        assert stock_development["Date"].max() < stock_test["Date"].min()
        n_development = len(stock_development)
        assert n_development > 120 and len(stock_test) >= 5
        pd.testing.assert_frame_equal(
            stock_all.iloc[:n_development].reset_index(drop=True), stock_development,
            check_dtype=False)
        stock_dates = pd.DatetimeIndex(stock_all["Date"])
        stock_raw = stock_all[["Open"]].to_numpy(dtype=np.float32)
        test_positions = stock_dates.get_indexer(stock_test["Date"])
        assert (test_positions >= n_development).all()
        np.testing.assert_allclose(stock_raw[test_positions, 0], stock_test["Open"].to_numpy())
        stock_test_position_set = set(test_positions.tolist())
        context_only_positions = [i for i in range(n_development, len(stock_all))
                                  if i not in stock_test_position_set]
        print("Development:", len(stock_development), "rows;",
              stock_development["Date"].min().date(), "to", stock_development["Date"].max().date())
        print("Test:", len(stock_test), "rows;",
              stock_test["Date"].min().date(), "to", stock_test["Date"].max().date())
        print("Context-only dates:", [stock_dates[i].strftime("%Y-%m-%d") for i in context_only_positions])
        display(stock_development.head(3))
        display(stock_test.head(3))

        # September 6th is a Sunday; September 7th is the US Labor Day holiday.
        # If present, September 8th is used only as known input before September 9th.
        # It is never used to fit a scaler or as a training, validation or test target.
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.plot(stock_development["Date"], stock_development["Open"], label="Development period")
        ax.plot(stock_test["Date"], stock_test["Open"], label="Test period")
        ax.set(title="Microsoft opening prices", xlabel="Date", ylabel="Opening price (USD)")
        ax.legend()
        fig.tight_layout()
        save_figure(fig, "stock_price_periods.png")
        '''),
        md("""
        ### Model, windows and saved weights

        An input window contains only observations available before its target date.<br>
        Each recurrent layer passes its sequence to the next layer; the fourth layer's final time step feeds the output layer.<br>
        A one-step model has one output, while the five-step model below has five outputs.<br>
        Hidden states start at zero for each window, as in the supplied notebooks.

        Training minimises mean squared error on scaled prices.<br>
        Each batch starts with cleared gradients.<br>
        Back-propagation calculates the gradients, and Adam uses them to update the weights.<br>
        The RNN's gradient clipping acts before the update.<br>
        Evaluation disables dropout and gradient tracking.

        Test evaluation rolls forward through the observed prices.<br>
        Later test forecasts can therefore use earlier test opening prices as inputs, while the weights remain fixed.<br>
        This evaluates forecasts made as each new observation arrives.<br>
        The persistence baseline repeats the most recent observed opening price.<br>
        Errors are reported in dollars after reversing the training scale.
        """),
        code(r'''
        # These helpers are shared by all RNN and LSTM experiments.
        class StockSequenceModel(nn.Module):
            def __init__(self, kind, horizon=1, hidden_size=50, dropout=0.20):
                super().__init__()
                if kind not in {"RNN", "LSTM"}:
                    raise ValueError(kind)
                self.kind = kind
                recurrent = nn.RNN if kind == "RNN" else nn.LSTM
                self.layers = nn.ModuleList([
                    recurrent(1 if number == 0 else hidden_size, hidden_size,
                              num_layers=1, batch_first=True)
                    for number in range(4)])
                self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(4)])
                self.output = nn.Linear(hidden_size, horizon)

            def forward(self, values):
                for number, layer in enumerate(self.layers):
                    values, _ = layer(values)
                    if number == 3:
                        values = values[:, -1, :]
                    values = self.dropouts[number](values)
                return self.output(values)

        def stock_windows(scaled, origins, lookback, horizon):
            """Origin i predicts rows i through i+horizon-1 from rows i-lookback:i."""
            origins = np.asarray(origins, dtype=np.int64)
            assert origins.ndim == 1 and len(origins)
            assert origins.min() >= lookback and origins.max() + horizon <= len(scaled)
            inputs = np.stack([scaled[i - lookback:i] for i in origins]).astype(np.float32)
            targets = np.stack([scaled[i:i + horizon, 0] for i in origins]).astype(np.float32)
            assert inputs.shape == (len(origins), lookback, 1)
            assert targets.shape == (len(origins), horizon)
            return inputs, targets

        def stock_test_origins(horizon):
            # Every future target must belong to the requested test period.
            return np.array([int(i) for i in test_positions
                             if all(i + h in stock_test_position_set for h in range(horizon))],
                            dtype=np.int64)

        def stock_unscale(values, fitted_scaler):
            shape = np.asarray(values).shape
            return fitted_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).reshape(shape)

        def stock_error_metrics(actual, predicted):
            error = np.asarray(predicted, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
            assert error.shape == np.asarray(actual).shape and np.isfinite(error).all()
            return {"mae_usd": float(np.abs(error).mean()),
                    "rmse_usd": float(np.sqrt(np.mean(error ** 2)))}

        def stock_scaler_record(fitted_scaler):
            return {"scale_": fitted_scaler.scale_.tolist(), "min_": fitted_scaler.min_.tolist(),
                    "data_min_": fitted_scaler.data_min_.tolist(),
                    "data_max_": fitted_scaler.data_max_.tolist(),
                    "data_range_": fitted_scaler.data_range_.tolist(),
                    "n_features_in_": int(fitted_scaler.n_features_in_),
                    "n_samples_seen_": int(fitted_scaler.n_samples_seen_)}

        def restore_stock_scaler(record):
            fitted = MinMaxScaler()
            for key in ("scale_", "min_", "data_min_", "data_max_", "data_range_"):
                setattr(fitted, key, np.asarray(record[key], dtype=np.float32))
            fitted.n_features_in_ = record["n_features_in_"]
            fitted.n_samples_seen_ = record["n_samples_seen_"]
            return fitted

        def stock_predict(model, inputs):
            model.eval()
            batches = []
            with torch.inference_mode():
                for start in range(0, len(inputs), 256):
                    batch = torch.from_numpy(inputs[start:start + 256]).to(device)
                    batches.append(model(batch).cpu().numpy())
            return np.concatenate(batches)

        def stock_epoch(model, loader, optimizer=None):
            model.train(optimizer is not None)
            sum_squared_error, target_count = 0.0, 0
            with torch.set_grad_enabled(optimizer is not None):
                for inputs, expected in loader:
                    inputs, expected = inputs.to(device), expected.to(device)
                    if optimizer is not None:
                        optimizer.zero_grad(set_to_none=True)
                    predicted = model(inputs)
                    assert predicted.shape == expected.shape
                    loss = F.mse_loss(predicted, expected)
                    if optimizer is not None:
                        loss.backward()
                        if model.kind == "RNN":
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    sum_squared_error += float(F.mse_loss(
                        predicted.detach(), expected, reduction="sum").item())
                    target_count += expected.numel()
            return sum_squared_error / target_count

        def fit_stock_model(run_name, kind, lookback, horizon, fitted_scaler,
                            train_origins, validation_origins=None):
            """Save immediately after training or whenever validation loss improves."""
            checkpoint_path = STOCK_DIR / f"{run_name}.pt"
            history_path = STOCK_DIR / f"{run_name}_history.json"
            record_path = STOCK_DIR / f"{run_name}_run.json"
            if STOCK_REUSE_COMPLETED and all(path.exists() for path in
                                             (checkpoint_path, history_path, record_path)):
                saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                run_record = json.loads(record_path.read_text(encoding="utf-8"))
                assert run_record.get("completed") and saved["epochs"] == STOCK_EPOCHS
                assert saved["lookback"] == lookback and saved["horizon"] == horizon
                assert saved["model_config"]["kind"] == kind
                assert saved["scaler"] == stock_scaler_record(fitted_scaler)
                assert saved["training_origins"] == np.asarray(train_origins).tolist()
                expected_validation = [] if validation_origins is None else np.asarray(validation_origins).tolist()
                assert saved["validation_origins"] == expected_validation
                restored = StockSequenceModel(**saved["model_config"]).to(device)
                restored.load_state_dict(saved["model_state_dict"])
                restored.eval()
                history = json.loads(history_path.read_text(encoding="utf-8"))
                assert len(history) == STOCK_EPOCHS
                print(f"{run_name}: explicitly reusing its completed, checked checkpoint")
                return restored, run_record, history
            torch.manual_seed(SEED)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(SEED)
            scaled = fitted_scaler.transform(stock_raw).astype(np.float32)
            train_x, train_y = stock_windows(scaled, train_origins, lookback, horizon)
            train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                                      batch_size=STOCK_BATCH_SIZE, shuffle=True, num_workers=0,
                                      generator=torch.Generator().manual_seed(SEED))
            validation_loader = None
            if validation_origins is not None:
                val_x, val_y = stock_windows(scaled, validation_origins, lookback, horizon)
                validation_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
                                               batch_size=256, shuffle=False, num_workers=0)
            config = {"kind": kind, "horizon": horizon, "hidden_size": 50, "dropout": 0.20}
            model = StockSequenceModel(**config).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=STOCK_LEARNING_RATE)
            history, best_loss, selected_epoch = [], float("inf"), None
            started = time.perf_counter()
            print(f"{run_name}: inputs {train_x.shape}; targets {train_y.shape}; "
                  f"parameters {sum(p.numel() for p in model.parameters()):,}")
            for epoch in range(1, STOCK_EPOCHS + 1):
                tick = time.perf_counter()
                train_mse = stock_epoch(model, train_loader, optimizer)
                val_mse = stock_epoch(model, validation_loader) if validation_loader is not None else None
                history.append({"epoch": epoch, "train_mse_scaled": train_mse,
                                "val_mse_scaled": val_mse, "seconds": time.perf_counter() - tick})
                history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
                should_save = (val_mse is not None and val_mse < best_loss) or (
                    val_mse is None and epoch == STOCK_EPOCHS)
                if should_save:
                    best_loss = val_mse if val_mse is not None else train_mse
                    selected_epoch = epoch
                    torch.save({
                        "model_state_dict": {key: value.detach().cpu().clone()
                                             for key, value in model.state_dict().items()},
                        "model_config": config, "lookback": lookback, "horizon": horizon,
                        "scaler": stock_scaler_record(fitted_scaler),
                        "selected_epoch": epoch, "selection_loss_scaled": best_loss,
                        "selection_rule": "minimum validation MSE" if val_mse is not None else "final epoch",
                        "seed": SEED, "epochs": STOCK_EPOCHS, "batch_size": STOCK_BATCH_SIZE,
                        "learning_rate": STOCK_LEARNING_RATE,
                        "training_origins": np.asarray(train_origins).tolist(),
                        "validation_origins": [] if validation_origins is None else np.asarray(validation_origins).tolist(),
                    }, checkpoint_path)
                if epoch in {1, 50, STOCK_EPOCHS}:
                    validation_text = "" if val_mse is None else f"; validation MSE {val_mse:.6f}"
                    print(f"  epoch {epoch:03d}: training MSE {train_mse:.6f}{validation_text}; "
                          f"{history[-1]['seconds']:.2f} s")
                    if epoch == 1:
                        print(f"  first-epoch estimate: {STOCK_EPOCHS * history[-1]['seconds'] / 60:.1f} minutes")
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(saved["model_state_dict"])
            model.eval()
            # Rebuild a separate model and scaler; compare predictions before returning.
            restored = StockSequenceModel(**saved["model_config"]).to(device)
            restored.load_state_dict(saved["model_state_dict"])
            restored.eval()
            check_x = train_x[:min(8, len(train_x))]
            np.testing.assert_allclose(stock_predict(model, check_x), stock_predict(restored, check_x),
                                       rtol=1e-5, atol=1e-6)
            restored_scaler = restore_stock_scaler(saved["scaler"])
            np.testing.assert_allclose(restored_scaler.transform(stock_raw), scaled, rtol=1e-5, atol=1e-6)
            duration = time.perf_counter() - started
            run_record = {"run_name": run_name, "kind": kind, "lookback": lookback, "horizon": horizon,
                          "selected_epoch": selected_epoch, "selection_loss_scaled": best_loss,
                          "selection_rule": saved["selection_rule"], "epochs": STOCK_EPOCHS,
                          "batch_size": STOCK_BATCH_SIZE, "learning_rate": STOCK_LEARNING_RATE,
                          "train_examples": len(train_x),
                          "validation_examples": 0 if validation_origins is None else len(validation_origins),
                          "input_shape": list(train_x.shape[1:]), "target_shape": list(train_y.shape[1:]),
                          "scaler": saved["scaler"], "checkpoint": checkpoint_path.name,
                          "reload_verified": True, "seconds": duration,
                          "completed": True,
                          "train_origins": np.asarray(train_origins).tolist(),
                          "validation_origins": saved["validation_origins"]}
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
            record_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
            print(f"  selected epoch {selected_epoch}; reload verified; total {duration / 60:.1f} minutes")
            del optimizer, model
            gc.collect()
            return restored, run_record, history

        def evaluate_stock(model, fitted_scaler, origins, lookback, horizon, run_name, split):
            scaled = fitted_scaler.transform(stock_raw).astype(np.float32)
            inputs, targets_scaled = stock_windows(scaled, origins, lookback, horizon)
            predicted = stock_unscale(stock_predict(model, inputs), fitted_scaler)
            actual = stock_unscale(targets_scaled, fitted_scaler)
            expected_raw = np.stack([stock_raw[i:i + horizon, 0] for i in origins])
            np.testing.assert_allclose(actual, expected_raw, rtol=1e-5, atol=1e-3)
            baseline = np.repeat(stock_raw[np.asarray(origins) - 1, 0:1], horizon, axis=1)
            rows = []
            for number, origin in enumerate(origins):
                for h in range(horizon):
                    rows.append({"run_name": run_name, "split": split,
                                 "origin_date": stock_dates[origin - 1].strftime("%Y-%m-%d"),
                                 "input_start_date": stock_dates[origin - lookback].strftime("%Y-%m-%d"),
                                 "target_date": stock_dates[origin + h].strftime("%Y-%m-%d"),
                                 "horizon": h + 1, "actual_open": float(actual[number, h]),
                                 "predicted_open": float(predicted[number, h]),
                                 "persistence_open": float(baseline[number, h])})
            predictions = pd.DataFrame(rows)
            predictions.to_csv(STOCK_DIR / f"{run_name}_{split}_predictions.csv", index=False)
            return {**stock_error_metrics(actual, predicted),
                    **{"baseline_" + key: value for key, value in stock_error_metrics(actual, baseline).items()},
                    "origins": len(origins), "target_cells": int(actual.size)}, predictions
        '''),
        code(r'''
        # Problem 1 follows the original full-development training formulation.
        p1_stock_scaler = MinMaxScaler().fit(stock_raw[:n_development])
        p1_train_origins = np.arange(60, n_development)
        p1_test_origins = stock_test_origins(1)
        p1_stock_results, p1_stock_histories, p1_stock_predictions = [], {}, {}
        for stock_kind in ("RNN", "LSTM"):
            run_name = f"p1_{stock_kind.lower()}_60"
            fitted_model, run_record, history = fit_stock_model(
                run_name, stock_kind, 60, 1, p1_stock_scaler, p1_train_origins)
            test_metrics, predictions = evaluate_stock(
                fitted_model, p1_stock_scaler, p1_test_origins, 60, 1, run_name, "test")
            p1_stock_results.append({"model": stock_kind, **test_metrics})
            p1_stock_histories[stock_kind] = history
            p1_stock_predictions[stock_kind] = predictions
            del fitted_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        p1_stock_table = pd.DataFrame(p1_stock_results)
        p1_stock_table.to_csv(STOCK_DIR / "problem1_metrics.csv", index=False)
        display(p1_stock_table.round(3))

        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
        for ax, kind in zip(axes, ("RNN", "LSTM")):
            rows = p1_stock_predictions[kind]
            dates = pd.to_datetime(rows["target_date"])
            ax.plot(dates, rows["actual_open"], label="Actual")
            ax.plot(dates, rows["predicted_open"], label=kind)
            ax.plot(dates, rows["persistence_open"], "--", label="Previous opening price")
            ax.set(title=f"{kind}: 60-observation input", ylabel="Opening price (USD)")
            ax.tick_params(axis="x", rotation=40)
            ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, "problem1_stock_predictions.png")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for kind, history in p1_stock_histories.items():
            ax.plot([row["epoch"] for row in history], [row["train_mse_scaled"] for row in history], label=kind)
        ax.set(title="Problem 1 training loss", xlabel="Epoch", ylabel="MSE on scaled prices")
        ax.legend()
        fig.tight_layout()
        save_figure(fig, "problem1_stock_training.png")
        for row in p1_stock_results:
            verdict = "lower" if row["rmse_usd"] < row["baseline_rmse_usd"] else "higher"
            display(Markdown(f"The **{row['model']}** has test RMSE **\\${row['rmse_usd']:.2f}**, "
                             f"{verdict} than persistence at **\\${row['baseline_rmse_usd']:.2f}**.<br>\n"
                             "Following the price trend alone does not establish a forecasting improvement."))
        '''),
        md("""
        ## Problem 2. Use chronological validation and change the forecast

        I divide the original development rows chronologically: the first 80% are training data and the final 20% are validation data.<br>
        A new scaler is fitted only to that earlier training portion.<br>
        All Problem 2 models start with fresh weights; none uses the Problem 1 weights fitted to the whole development period.<br>
        Validation chooses the checkpoint and the one-step lookback separately for each model type.<br>
        The test period is reserved for the final comparison.

        For the one-step experiment, I use lookbacks of 10, 30, 60 and 120 trading observations.<br>
        All eight runs use the same training target dates after a common 120-observation warm-up, and the same validation and test target dates.<br>
        This keeps differences in the number of training examples out of the lookback comparison.<br>
        A validation input may contain earlier training observations, because those observations were already known before its target date.
        """),
        code(r'''
        # Set boundaries before building examples, so targets cannot cross a split.
        stock_training_rows = int(np.floor(0.80 * n_development))
        assert stock_training_rows > 120
        p2_stock_scaler = MinMaxScaler().fit(stock_raw[:stock_training_rows])
        p2_train_origins = np.arange(120, stock_training_rows)
        p2_validation_origins = np.arange(stock_training_rows, n_development)
        p2_test_origins = stock_test_origins(1)
        assert p2_train_origins.max() < stock_training_rows
        assert p2_validation_origins.min() == stock_training_rows
        assert p2_validation_origins.max() < n_development
        assert set(p2_train_origins).isdisjoint(p2_validation_origins)
        assert p2_stock_scaler.n_samples_seen_ == stock_training_rows
        np.testing.assert_allclose(p2_stock_scaler.data_min_, stock_raw[:stock_training_rows].min(axis=0))
        np.testing.assert_allclose(p2_stock_scaler.data_max_, stock_raw[:stock_training_rows].max(axis=0))
        stock_split_manifest = {
            "company": "Microsoft", "ticker": "MSFT", "development_rows": n_development,
            "training_rows": stock_training_rows, "validation_rows": n_development - stock_training_rows,
            "split_fraction_requested": 0.80, "common_one_step_warmup": 120,
            "training_dates": stock_dates[:stock_training_rows].strftime("%Y-%m-%d").tolist(),
            "validation_dates": stock_dates[stock_training_rows:n_development].strftime("%Y-%m-%d").tolist(),
            "test_dates": stock_dates[test_positions].strftime("%Y-%m-%d").tolist(),
            "context_only_dates": stock_dates[context_only_positions].strftime("%Y-%m-%d").tolist(),
            "p2_scaler": stock_scaler_record(p2_stock_scaler),
            "one_step_train_origins": p2_train_origins.tolist(),
            "one_step_validation_origins": p2_validation_origins.tolist(),
            "one_step_test_origins": p2_test_origins.tolist(),
        }
        (STOCK_DIR / "split_manifest.json").write_text(json.dumps(stock_split_manifest, indent=2), encoding="utf-8")
        print(f"Chronological split: {stock_training_rows} training rows; "
              f"{n_development - stock_training_rows} validation rows.")
        print("Training ends:", stock_dates[stock_training_rows - 1].date(),
              "; validation starts:", stock_dates[stock_training_rows].date())
        print("Scaler fitted to:", p2_stock_scaler.n_samples_seen_, "training prices only.")
        print("One-step example counts:", len(p2_train_origins), "training;",
              len(p2_validation_origins), "validation;", len(p2_test_origins), "test.")
        '''),
        code(r'''
        # Train all lookbacks and select them using validation results only.
        p2_lookback_results = []
        p2_lookback_histories = {}
        for stock_kind in ("RNN", "LSTM"):
            for stock_lookback in (10, 30, 60, 120):
                run_name = f"p2_{stock_kind.lower()}_{stock_lookback}"
                fitted_model, run_record, history = fit_stock_model(
                    run_name, stock_kind, stock_lookback, 1, p2_stock_scaler,
                    p2_train_origins, p2_validation_origins)
                val_metrics, _ = evaluate_stock(
                    fitted_model, p2_stock_scaler, p2_validation_origins,
                    stock_lookback, 1, run_name, "validation")
                p2_lookback_results.append({"run_name": run_name, "model": stock_kind,
                                            "lookback": stock_lookback,
                                            "selected_epoch": run_record["selected_epoch"],
                                            **{"validation_" + key: value for key, value in val_metrics.items()}})
                p2_lookback_histories[run_name] = history
                del fitted_model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        p2_validation_table = pd.DataFrame(p2_lookback_results)
        selected_stock_lookbacks = {
            kind: int(p2_validation_table.loc[p2_validation_table["model"].eq(kind)]
                      .sort_values(["validation_rmse_usd", "lookback"]).iloc[0]["lookback"])
            for kind in ("RNN", "LSTM")}
        (STOCK_DIR / "lookback_selection.json").write_text(json.dumps({
            "selection_metric": "validation RMSE in USD", "test_results_used": False,
            "selected_lookbacks": selected_stock_lookbacks,
            "candidates": p2_lookback_results}, indent=2), encoding="utf-8")
        print("Lookbacks selected before test evaluation:", selected_stock_lookbacks)
        display(p2_validation_table[["model", "lookback", "selected_epoch", "validation_mae_usd",
                                      "validation_rmse_usd"]].round(3))
        '''),
        code(r'''
        # With selection fixed, restore each checkpoint and evaluate the test period.
        p2_all_lookback_results, p2_selected_predictions = [], {}
        for row in p2_lookback_results:
            saved = torch.load(STOCK_DIR / f"{row['run_name']}.pt", map_location="cpu", weights_only=True)
            test_model = StockSequenceModel(**saved["model_config"]).to(device)
            test_model.load_state_dict(saved["model_state_dict"])
            test_scaler = restore_stock_scaler(saved["scaler"])
            metrics, predictions = evaluate_stock(test_model, test_scaler, p2_test_origins,
                                                  row["lookback"], 1, row["run_name"], "test")
            p2_all_lookback_results.append({**row, **{"test_" + key: value for key, value in metrics.items()},
                                           "selected_by_validation": row["lookback"] == selected_stock_lookbacks[row["model"]]})
            if row["lookback"] == selected_stock_lookbacks[row["model"]]:
                p2_selected_predictions[row["model"]] = predictions
            del test_model
        p2_stock_table = pd.DataFrame(p2_all_lookback_results)
        p2_stock_table.to_csv(STOCK_DIR / "problem2_lookback_metrics.csv", index=False)
        display(p2_stock_table[["model", "lookback", "selected_epoch", "validation_rmse_usd",
                                 "test_mae_usd", "test_rmse_usd", "test_baseline_rmse_usd",
                                 "selected_by_validation"]].round(3))
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        for kind in ("RNN", "LSTM"):
            rows = p2_stock_table.loc[p2_stock_table["model"].eq(kind)].sort_values("lookback")
            axes[0].plot(rows["lookback"], rows["validation_rmse_usd"], marker="o", label=kind)
            axes[1].plot(rows["lookback"], rows["test_rmse_usd"], marker="o", label=kind)
        axes[0].set(title="Validation chooses the lookback", xlabel="Lookback (trading observations)", ylabel="RMSE (USD)")
        axes[1].axhline(p2_stock_table.iloc[0]["test_baseline_rmse_usd"], color="gray", linestyle="--", label="Persistence")
        axes[1].set(title="Test comparison after selection", xlabel="Lookback (trading observations)", ylabel="RMSE (USD)")
        for ax in axes:
            ax.set_xticks([10, 30, 60, 120])
            ax.legend()
        fig.tight_layout()
        save_figure(fig, "problem2_lookback_comparison.png")
        for kind, lookback in selected_stock_lookbacks.items():
            row = p2_stock_table.loc[p2_stock_table["model"].eq(kind) & p2_stock_table["lookback"].eq(lookback)].iloc[0]
            verdict = "beats" if row["test_rmse_usd"] < row["test_baseline_rmse_usd"] else "does not beat"
            display(Markdown(f"For **{kind}**, validation selects **{lookback} observations**, "
                             f"with validation RMSE **\\${row['validation_rmse_usd']:.2f}**.<br>\n"
                             f"Its test RMSE is **\\${row['test_rmse_usd']:.2f}**; it {verdict} "
                             f"persistence (**\\${row['test_baseline_rmse_usd']:.2f}**) on this test period."))
        '''),
        md("""
        ### Predict five future opening prices from a `(60, 1)` input

        The next experiment fixes the input at 60 past trading observations and changes the target to a vector of five future opening prices.<br>
        The output layer now has five units, so one forward pass predicts all five prices directly.<br>
        A later predicted price is not fed back as an input.

        All five targets must lie within the same training, validation or test period.<br>
        I drop examples whose targets would cross a boundary.<br>
        Past inputs can still include observations from an earlier period.<br>
        Five steps means five trading observations, which can span more than five calendar days.<br>
        The persistence baseline repeats the last observed opening price at every horizon.<br>
        I show errors separately for horizons one through five and an aggregate across all origin–horizon pairs.<br>
        Adjacent forecasts overlap, so these aggregate errors do not describe independent observations.
        """),
        code(r'''
        # Construct direct five-step targets without crossing a target split boundary.
        p2_five_train_origins = np.arange(60, stock_training_rows - 5 + 1)
        p2_five_validation_origins = np.arange(stock_training_rows, n_development - 5 + 1)
        p2_five_test_origins = stock_test_origins(5)
        assert p2_five_train_origins.max() + 4 < stock_training_rows
        assert p2_five_validation_origins.min() >= stock_training_rows
        assert p2_five_validation_origins.max() + 4 < n_development
        assert all(all(i + h in stock_test_position_set for h in range(5))
                   for i in p2_five_test_origins)
        five_scaled = p2_stock_scaler.transform(stock_raw).astype(np.float32)
        five_x, five_y = stock_windows(five_scaled, p2_five_train_origins, 60, 5)
        print("Five-step training input shape:", five_x.shape)
        print("Five-step training target shape:", five_y.shape)
        print("First training target:", stock_unscale(five_y[:1], p2_stock_scaler).round(2))
        (STOCK_DIR / "five_step_window_manifest.json").write_text(json.dumps({
            "lookback": 60, "horizon": 5, "train_origins": p2_five_train_origins.tolist(),
            "validation_origins": p2_five_validation_origins.tolist(),
            "test_origins": p2_five_test_origins.tolist(),
            "train_input_shape": list(five_x.shape), "train_target_shape": list(five_y.shape),
            "target_boundaries_checked": True}, indent=2), encoding="utf-8")
        p2_five_results, p2_five_horizon_rows, p2_five_predictions = [], [], {}
        for kind in ("RNN", "LSTM"):
            run_name = f"p2_{kind.lower()}_60_five"
            model, run_record, history = fit_stock_model(
                run_name, kind, 60, 5, p2_stock_scaler,
                p2_five_train_origins, p2_five_validation_origins)
            validation_metrics, _ = evaluate_stock(model, p2_stock_scaler, p2_five_validation_origins,
                                                   60, 5, run_name, "validation")
            test_metrics, predictions = evaluate_stock(model, p2_stock_scaler, p2_five_test_origins,
                                                       60, 5, run_name, "test")
            p2_five_predictions[kind] = predictions
            p2_five_results.append({"model": kind, "lookback": 60, "horizon": 5,
                                    "selected_epoch": run_record["selected_epoch"],
                                    **{"validation_" + key: value for key, value in validation_metrics.items()},
                                    **{"test_" + key: value for key, value in test_metrics.items()}})
            for horizon, group in predictions.groupby("horizon", sort=True):
                metric = stock_error_metrics(group["actual_open"], group["predicted_open"])
                baseline_metric = stock_error_metrics(group["actual_open"], group["persistence_open"])
                p2_five_horizon_rows.append({"model": kind, "horizon": int(horizon), "forecast_origins": len(group),
                                             **metric, **{"baseline_" + key: value for key, value in baseline_metric.items()}})
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        p2_five_table = pd.DataFrame(p2_five_results)
        p2_horizon_table = pd.DataFrame(p2_five_horizon_rows)
        p2_five_table.to_csv(STOCK_DIR / "problem2_five_step_metrics.csv", index=False)
        p2_horizon_table.to_csv(STOCK_DIR / "problem2_horizon_metrics.csv", index=False)
        display(p2_five_table[["model", "selected_epoch", "validation_rmse_usd", "test_mae_usd",
                                "test_rmse_usd", "test_baseline_rmse_usd", "test_origins", "test_target_cells"]].round(3))
        display(p2_horizon_table.round(3))

        fig, ax = plt.subplots(figsize=(8, 3.5))
        for kind in ("RNN", "LSTM"):
            rows = p2_horizon_table.loc[p2_horizon_table["model"].eq(kind)]
            ax.plot(rows["horizon"], rows["rmse_usd"], marker="o", label=kind)
        baseline_rows = p2_horizon_table.loc[p2_horizon_table["model"].eq("RNN")]
        ax.plot(baseline_rows["horizon"], baseline_rows["baseline_rmse_usd"], "--", marker="o", label="Persistence")
        ax.set(title="Direct five-step test errors", xlabel="Forecast horizon (trading observations)", ylabel="RMSE (USD)", xticks=[1, 2, 3, 4, 5])
        ax.legend()
        fig.tight_layout()
        save_figure(fig, "problem2_five_step_horizons.png")
        for row in p2_five_results:
            verdict = "lower" if row["test_rmse_usd"] < row["test_baseline_rmse_usd"] else "higher"
            display(Markdown(f"The direct five-step **{row['model']}** has aggregate test RMSE "
                             f"**\\${row['test_rmse_usd']:.2f}**, {verdict} than persistence at "
                             f"**\\${row['test_baseline_rmse_usd']:.2f}**.<br>\n"
                             f"This uses **{row['test_origins']} forecast origins**, each with five targets.<br>\n"
                             "The horizon table shows whether errors differ as the forecast moves further ahead."))
        '''),
    ]
    # Scope temporary models, tensors and plotting variables inside each experiment.
    # Only the named stock results remain available to later notebook sections.
    wrappers = {
        4: ("stock_problem1_experiment", "p1_stock_results, p1_stock_histories, p1_stock_predictions, p1_stock_table"),
        7: ("stock_train_lookback_experiments", "p2_lookback_results, p2_lookback_histories, selected_stock_lookbacks, p2_validation_table"),
        8: ("stock_test_lookback_experiments", "p2_stock_table, p2_selected_predictions"),
        10: ("stock_five_step_experiment", "p2_five_table, p2_horizon_table, p2_five_predictions, p2_five_results"),
    }
    for index, (function_name, results) in wrappers.items():
        source = cells[index].source
        cells[index].source = (f"def {function_name}():\n" + indent(source, "    ")
                               + f"\n    return {results}\n\n{results} = {function_name}()")
    names = {
        "n_development": "stock_development_count",
        "test_positions": "stock_test_positions",
        "context_only_positions": "stock_context_only_positions",
        "load_stock_prices": "stock_load_prices",
        "restore_stock_scaler": "stock_restore_scaler",
        "fit_stock_model": "stock_fit_model",
        "evaluate_stock": "stock_evaluate",
    }
    for cell in cells:
        if cell.cell_type == "code":
            for old, new in names.items():
                cell.source = re.sub(r"\b" + old + r"\b", new, cell.source)
    cells[1].source = re.sub(r"\bfig\b", "stock_period_figure", cells[1].source)
    cells[1].source = re.sub(r"\bax\b", "stock_period_axis", cells[1].source)
    return cells
