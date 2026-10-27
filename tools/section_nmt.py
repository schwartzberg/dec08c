"""Self-contained notebook cells for the controlled NMT vocabulary experiment."""
from textwrap import dedent
import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


def make_cells():
    return [
        md(r"""
        ## Problem 3. Increase the translator's vocabulary

        I compare vocabulary budgets of **1,000** and **3,000** tokens in the supplied `3_NMT.ipynb` model.<br>
        Each language has its own vocabulary, and each budget includes the four reserved tokens.<br>
        All other settings remain unchanged.<br>
        Both runs use the full filtered corpus, seed 42, a 50-token limit and batches of 64.<br>
        Embeddings have 128 values, and the encoder and decoder each have one 512-unit LSTM.<br>
        Adam uses a learning rate of 0.001, the gradient norm is clipped at 1.0, and each model trains for 10 epochs.<br>
        `QUICK_RUN` remains false.<br>
        Each model is restored to its lowest validation-loss epoch.

        The data, tokenisation and grouped split follow the starter notebook.<br>
        Normalised English sentences cannot cross the training, validation and test boundaries.<br>
        The vocabulary is fitted on training text only.<br>
        I select the same five held-out English inputs for both models before training, using the starter's short-to-long example selection.
        """),
        code(r'''
        # Reproduce the starter's corpus, tokenisation and grouped random split.
        import csv
        import hashlib
        import math
        import random
        import re
        import time
        import unicodedata
        import urllib.request
        import zipfile
        from collections import Counter
        from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
        from torch.utils.data import DataLoader, Dataset

        NMT_DATA_DIR = DATA_DIR / "nmt"
        NMT_OUTPUT_DIR = OUTPUT_DIR / "nmt"
        NMT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        NMT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        NMT_SETTINGS = {
            "QUICK_RUN": False, "MAX_PAIRS": None, "MAX_WORDS": 50,
            "BATCH_SIZE": 64, "EMBED_DIM": 128, "HIDDEN_DIM": 512,
            "EPOCHS": 10, "LEARNING_RATE": 0.001, "CLIP_NORM": 1.0,
            "SEED": 42, "vocabulary_budgets": [1000, 3000],
            "checkpoint_selection": "minimum validation token cross-entropy",
        }
        assert SEED == NMT_SETTINGS["SEED"]
        NMT_URL = "https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
        NMT_ARCHIVE_SHA256 = "0c4fd7825d3d872488d38c157f784e035edab4f2ff0df25a2a208790ecb5b58f"
        nmt_archive_path = NMT_DATA_DIR / "spa-eng.zip"
        if not nmt_archive_path.exists():
            nmt_temporary_path = nmt_archive_path.with_suffix(".download")
            urllib.request.urlretrieve(NMT_URL, nmt_temporary_path)
            nmt_temporary_path.replace(nmt_archive_path)
        assert hashlib.sha256(nmt_archive_path.read_bytes()).hexdigest() == NMT_ARCHIVE_SHA256
        with zipfile.ZipFile(nmt_archive_path) as nmt_archive:
            nmt_text_members = [name for name in nmt_archive.namelist()
                                if Path(name).name == "spa.txt"]
            assert len(nmt_text_members) == 1
            nmt_raw_text = nmt_archive.read(nmt_text_members[0]).decode("utf-8-sig")
            nmt_about = [name for name in nmt_archive.namelist()
                         if Path(name).name == "_about.txt"]
            if nmt_about:
                (NMT_OUTPUT_DIR / "corpus_attribution.txt").write_bytes(nmt_archive.read(nmt_about[0]))

        def nmt_tokenize(text):
            text = unicodedata.normalize("NFC", text).lower()
            return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", text, flags=re.UNICODE)

        def nmt_detokenize(tokens):
            text = " ".join(tokens)
            text = re.sub(r"\s+([.,!?;:])", r"\1", text)
            return re.sub(r"([¿¡])\s+", r"\1", text)

        nmt_raw_pairs = []
        for nmt_line in nmt_raw_text.splitlines():
            nmt_fields = nmt_line.split("\t")
            if len(nmt_fields) >= 2 and nmt_fields[0].strip() and nmt_fields[1].strip():
                nmt_raw_pairs.append((nmt_fields[0].strip(), nmt_fields[1].strip()))
        nmt_pairs, nmt_seen = [], set()
        for nmt_source, nmt_target in nmt_raw_pairs:
            nmt_source_tokens, nmt_target_tokens = nmt_tokenize(nmt_source), nmt_tokenize(nmt_target)
            nmt_key = (tuple(nmt_source_tokens), tuple(nmt_target_tokens))
            if (0 < len(nmt_source_tokens) <= NMT_SETTINGS["MAX_WORDS"]
                    and 0 < len(nmt_target_tokens) <= NMT_SETTINGS["MAX_WORDS"]
                    and nmt_key not in nmt_seen):
                nmt_pairs.append((nmt_source_tokens, nmt_target_tokens))
                nmt_seen.add(nmt_key)

        # Keep both shuffles and their order identical to the starter.
        nmt_rng = random.Random(SEED)
        nmt_rng.shuffle(nmt_pairs)
        nmt_groups = {}
        for nmt_source, nmt_target in nmt_pairs:
            nmt_groups.setdefault(tuple(nmt_source), []).append((nmt_source, nmt_target))
        nmt_source_keys = list(nmt_groups)
        nmt_rng.shuffle(nmt_source_keys)
        nmt_train_end = int(0.8 * len(nmt_source_keys))
        nmt_valid_end = int(0.9 * len(nmt_source_keys))

        def nmt_collect_pairs(keys):
            return [pair for key in keys for pair in nmt_groups[key]]

        nmt_train_pairs = nmt_collect_pairs(nmt_source_keys[:nmt_train_end])
        nmt_valid_pairs = nmt_collect_pairs(nmt_source_keys[nmt_train_end:nmt_valid_end])
        nmt_test_pairs = nmt_collect_pairs(nmt_source_keys[nmt_valid_end:])
        nmt_split_keys = [set(tuple(pair[0]) for pair in split)
                          for split in (nmt_train_pairs, nmt_valid_pairs, nmt_test_pairs)]
        assert all(nmt_split_keys[a].isdisjoint(nmt_split_keys[b])
                   for a, b in ((0, 1), (0, 2), (1, 2)))
        assert sum(map(len, (nmt_train_pairs, nmt_valid_pairs, nmt_test_pairs))) == len(nmt_pairs)
        with (NMT_OUTPUT_DIR / "split_manifest.csv").open("w", encoding="utf-8", newline="") as nmt_stream:
            nmt_writer = csv.writer(nmt_stream)
            nmt_writer.writerow(["split", "english_tokens", "spanish_tokens"])
            for nmt_split_name, nmt_rows in (("train", nmt_train_pairs), ("validation", nmt_valid_pairs),
                                             ("test", nmt_test_pairs)):
                nmt_writer.writerows((nmt_split_name, " ".join(source), " ".join(target))
                                     for source, target in nmt_rows)
        nmt_manifest = {
            "source_url": NMT_URL, "archive_sha256": NMT_ARCHIVE_SHA256,
            "archive_date": "2018-05-13", "raw_pairs": len(nmt_raw_pairs),
            "retained_pairs": len(nmt_pairs), "source_groups": len(nmt_groups),
            "split_pair_counts": {"train": len(nmt_train_pairs), "validation": len(nmt_valid_pairs),
                                  "test": len(nmt_test_pairs)},
            "split_group_counts": {name: len(keys) for name, keys in
                                   zip(("train", "validation", "test"), nmt_split_keys)},
            "source_notebook_sha256": "d7ea900529283d49a280e09da2438a4f6e531053e93d4efe3115b123f628224f",
        }
        (NMT_OUTPUT_DIR / "settings.json").write_text(json.dumps(NMT_SETTINGS, indent=2), encoding="utf-8")
        (NMT_OUTPUT_DIR / "data_provenance.json").write_text(json.dumps(nmt_manifest, indent=2), encoding="utf-8")

        # The source notebook samples five positions after sorting by English length.
        nmt_ordered_test_indices = sorted(range(len(nmt_test_pairs)),
                                          key=lambda index: len(nmt_test_pairs[index][0]))
        nmt_positions = np.linspace(0, len(nmt_ordered_test_indices) - 1, num=5, dtype=int)
        nmt_five_indices = [nmt_ordered_test_indices[int(position)] for position in nmt_positions]
        nmt_five_examples = [
            {"test_index": index, "english": nmt_detokenize(nmt_test_pairs[index][0]),
             "reference": nmt_detokenize(nmt_test_pairs[index][1]),
             "source_tokens": nmt_test_pairs[index][0], "target_tokens": nmt_test_pairs[index][1]}
            for index in nmt_five_indices]
        assert len({tuple(row["source_tokens"]) for row in nmt_five_examples}) == 5
        (NMT_OUTPUT_DIR / "five_sources.json").write_text(
            json.dumps(nmt_five_examples, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Read {len(nmt_raw_pairs):,} pairs; retained {len(nmt_pairs):,} distinct normalised pairs.")
        print("Pair counts:", nmt_manifest["split_pair_counts"])
        print("No normalised English source appears in more than one split.")
        display(pd.DataFrame(nmt_five_examples)[["test_index", "english", "reference"]])
        '''),
        md(r"""
        ### Vocabulary, model and measurements

        Both vocabularies use the starter's frequency ranking and alphabetical tie-breaking.<br>
        The encoder reads the packed English sequence and passes its hidden and cell states to the Spanish decoder.<br>
        Training uses the correct preceding Spanish token; inference feeds back the model's own prediction.<br>
        The decoder uses the same greedy search and 51-step limit as the starter.

        The unknown-token percentage is the number of `<unk>` occurrences divided by the number of original content tokens.<br>
        It includes punctuation, as in the supplied notebook, and excludes padding and boundary markers.<br>
        I also save word-only unknown counts, with punctuation excluded, to make this distinction explicit.

        The requested validation accuracy follows the starter: padding is excluded, while `<eos>` and matching `<unk>` labels count.<br>
        This can reward a small vocabulary for predicting an unknown marker.<br>
        I therefore also report accuracy with every `<unk>` target counted as an error, keeping the same denominator of non-padding target positions.<br>
        Neither teacher-forced accuracy is a percentage of sentences translated correctly.<br>
        Losses and perplexities from different vocabulary sizes should not be read as directly comparable translation-quality scores.
        """),
        code(r'''
        # Keep the encoder-decoder architecture and all training settings unchanged.
        NMT_PAD, NMT_UNK, NMT_BOS, NMT_EOS = 0, 1, 2, 3
        NMT_SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

        class NmtVocabulary:
            def __init__(self, token_sequences, max_size):
                counts = Counter(token for sequence in token_sequences for token in sequence)
                ranked = sorted(counts, key=lambda token: (-counts[token], token))
                self.itos = NMT_SPECIAL_TOKENS + [token for token in ranked if token not in NMT_SPECIAL_TOKENS][
                    :max_size - len(NMT_SPECIAL_TOKENS)]
                self.stoi = {token: index for index, token in enumerate(self.itos)}

            def __len__(self):
                return len(self.itos)

            def encode(self, tokens):
                return [self.stoi.get(token, NMT_UNK) for token in tokens]

            def decode(self, ids):
                return [self.itos[index] for index in ids]

        def nmt_unknown_counts(pairs, side, vocabulary):
            tokens = [token for pair in pairs for token in pair[side]]
            unknown = sum(token not in vocabulary.stoi for token in tokens)
            words = [token for token in tokens if re.fullmatch(r"\w+(?:['’]\w+)*", token)]
            unknown_words = sum(token not in vocabulary.stoi for token in words)
            return {"unknown_tokens": unknown, "content_tokens": len(tokens),
                    "unknown_rate": unknown / len(tokens), "unknown_words": unknown_words,
                    "words": len(words), "word_unknown_rate": unknown_words / len(words)}

        class NmtTranslationDataset(Dataset):
            def __init__(self, pairs, source_vocab, target_vocab):
                self.examples = []
                for source, target in pairs:
                    source_ids = torch.tensor(source_vocab.encode(source), dtype=torch.long)
                    target_ids = torch.tensor([NMT_BOS] + target_vocab.encode(target) + [NMT_EOS], dtype=torch.long)
                    self.examples.append((source_ids, target_ids[:-1], target_ids[1:]))

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, index):
                return self.examples[index]

        def nmt_collate_batch(examples):
            sources, decoder_inputs, labels = zip(*examples)
            lengths = torch.tensor([len(source) for source in sources], dtype=torch.long)
            return (pad_sequence(sources, batch_first=True, padding_value=NMT_PAD), lengths,
                    pad_sequence(decoder_inputs, batch_first=True, padding_value=NMT_PAD),
                    pad_sequence(labels, batch_first=True, padding_value=NMT_PAD))

        class NmtEncoder(nn.Module):
            def __init__(self, vocab_size, embed_dim, hidden_dim):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=NMT_PAD)
                self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

            def forward(self, source_ids, source_lengths):
                embedded = self.embedding(source_ids)
                packed = pack_padded_sequence(embedded, source_lengths.cpu(), batch_first=True, enforce_sorted=False)
                _, state = self.lstm(packed)
                return state

        class NmtDecoder(nn.Module):
            def __init__(self, vocab_size, embed_dim, hidden_dim):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=NMT_PAD)
                self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
                self.output = nn.Linear(hidden_dim, vocab_size)

            def forward(self, decoder_ids, state):
                embedded = self.embedding(decoder_ids)
                hidden_sequence, state = self.lstm(embedded, state)
                return self.output(hidden_sequence), state

        class NmtSeq2Seq(nn.Module):
            def __init__(self, source_size, target_size, embed_dim, hidden_dim):
                super().__init__()
                self.encoder = NmtEncoder(source_size, embed_dim, hidden_dim)
                self.decoder = NmtDecoder(target_size, embed_dim, hidden_dim)

            def forward(self, source_ids, source_lengths, decoder_ids):
                state = self.encoder(source_ids, source_lengths)
                logits, _ = self.decoder(decoder_ids, state)
                return logits

        def nmt_run_epoch(model, loader, optimizer=None):
            training = optimizer is not None
            model.train(training)
            total_loss, total_correct, resolved_correct, total_tokens, unknown_targets = 0.0, 0, 0, 0, 0
            with torch.set_grad_enabled(training):
                for source_ids, source_lengths, decoder_ids, labels in loader:
                    source_ids, decoder_ids, labels = source_ids.to(device), decoder_ids.to(device), labels.to(device)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    logits = model(source_ids, source_lengths, decoder_ids)
                    loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                                               ignore_index=NMT_PAD, reduction="sum")
                    mask = labels.ne(NMT_PAD)
                    token_count = int(mask.sum().item())
                    if training:
                        (loss_sum / token_count).backward()
                        nn.utils.clip_grad_norm_(model.parameters(), NMT_SETTINGS["CLIP_NORM"])
                        optimizer.step()
                    correct = logits.detach().argmax(-1).eq(labels) & mask
                    total_loss += float(loss_sum.detach().item())
                    total_correct += int(correct.sum().item())
                    resolved_correct += int((correct & labels.ne(NMT_UNK)).sum().item())
                    unknown_targets += int((labels.eq(NMT_UNK) & mask).sum().item())
                    total_tokens += token_count
            return {"loss": total_loss / total_tokens, "accuracy": total_correct / total_tokens,
                    "accuracy_unk_as_error": resolved_correct / total_tokens,
                    "correct_tokens": total_correct, "resolved_correct_tokens": resolved_correct,
                    "tokens": total_tokens, "unknown_targets": unknown_targets}

        @torch.inference_mode()
        def nmt_translate(sentence, model, source_vocab, target_vocab, max_steps=51):
            tokens = nmt_tokenize(sentence)
            if not tokens or len(tokens) > NMT_SETTINGS["MAX_WORDS"]:
                raise ValueError("Use one to 50 source tokens, including punctuation.")
            if max_steps < 1:
                raise ValueError("max_steps must be positive.")
            model.eval()
            model_device = next(model.parameters()).device
            source_ids = torch.tensor([source_vocab.encode(tokens)], dtype=torch.long, device=model_device)
            lengths = torch.tensor([len(tokens)], dtype=torch.long)
            state = model.encoder(source_ids, lengths)
            next_input = torch.tensor([[NMT_BOS]], dtype=torch.long, device=model_device)
            generated, ended = [], False
            for _ in range(max_steps):
                logits, state = model.decoder(next_input, state)
                scores = logits[0, -1].clone()
                scores[[NMT_PAD, NMT_BOS]] = -torch.inf
                next_id = int(scores.argmax().item())
                if next_id == NMT_EOS:
                    ended = True
                    break
                generated.append(next_id)
                next_input = torch.tensor([[next_id]], dtype=torch.long, device=model_device)
            return {"translation": nmt_detokenize(target_vocab.decode(generated)),
                    "ended_with_eos": ended,
                    "unknown_source_tokens": [token for token in tokens if token not in source_vocab.stoi]}

        def nmt_restore_vocab(tokens):
            vocabulary = NmtVocabulary([], len(tokens))
            vocabulary.itos = list(tokens)
            vocabulary.stoi = {token: index for index, token in enumerate(tokens)}
            return vocabulary
        '''),
        md("""
        ### Train both vocabulary budgets

        Each experiment starts from a new model and the same seed.<br>
        The batches use a separate generator with the same seed, so the examples occur in the same order for both vocabulary budgets.<br>
        The initial batch inspection is retained from the source notebook.<br>
        I save a CPU checkpoint when validation loss improves and save the completed training history.<br>
        A fresh model and its saved vocabularies must reproduce all five generated translations after reloading.
        """),
        code(r'''
        # Train the full corpus for all ten epochs at each vocabulary budget.
        def nmt_train_experiment(vocabulary_budget):
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
            source_vocab = NmtVocabulary((source for source, _ in nmt_train_pairs), vocabulary_budget)
            target_vocab = NmtVocabulary((target for _, target in nmt_train_pairs), vocabulary_budget)
            assert len(source_vocab) == len(target_vocab) == vocabulary_budget
            coverage = {split_name: {language: nmt_unknown_counts(rows, side, vocabulary)
                                    for language, side, vocabulary in
                                    (("english", 0, source_vocab), ("spanish", 1, target_vocab))}
                        for split_name, rows in (("train", nmt_train_pairs), ("validation", nmt_valid_pairs),
                                                 ("test", nmt_test_pairs))}
            datasets = [NmtTranslationDataset(rows, source_vocab, target_vocab)
                        for rows in (nmt_train_pairs, nmt_valid_pairs, nmt_test_pairs)]
            loader_generator = torch.Generator().manual_seed(SEED)
            train_loader = DataLoader(datasets[0], batch_size=NMT_SETTINGS["BATCH_SIZE"], shuffle=True,
                                      collate_fn=nmt_collate_batch, num_workers=0, generator=loader_generator)
            valid_loader = DataLoader(datasets[1], batch_size=NMT_SETTINGS["BATCH_SIZE"],
                                      collate_fn=nmt_collate_batch, num_workers=0)
            test_loader = DataLoader(datasets[2], batch_size=NMT_SETTINGS["BATCH_SIZE"],
                                     collate_fn=nmt_collate_batch, num_workers=0)
            # This consumes the same initial shuffled batch as the starter.
            batch = next(iter(train_loader))
            model_config = {"source_size": len(source_vocab), "target_size": len(target_vocab),
                            "embed_dim": NMT_SETTINGS["EMBED_DIM"], "hidden_dim": NMT_SETTINGS["HIDDEN_DIM"]}
            model = NmtSeq2Seq(**model_config).to(device)
            with torch.no_grad():
                example_logits = model(batch[0].to(device), batch[1], batch[2].to(device))
                assert example_logits.shape == (*batch[3].shape, len(target_vocab))
            del example_logits
            optimizer = torch.optim.Adam(model.parameters(), lr=NMT_SETTINGS["LEARNING_RATE"])
            history, best_loss, best_state, best_epoch = [], float("inf"), None, None
            checkpoint_path = NMT_OUTPUT_DIR / f"seq2seq_vocab{vocabulary_budget}.pt"
            started_all = time.perf_counter()
            for epoch in range(1, NMT_SETTINGS["EPOCHS"] + 1):
                started = time.perf_counter()
                train_metrics = nmt_run_epoch(model, train_loader, optimizer)
                valid_metrics = nmt_run_epoch(model, valid_loader)
                elapsed = time.perf_counter() - started
                history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics, "seconds": elapsed})
                if valid_metrics["loss"] < best_loss:
                    best_loss, best_epoch = valid_metrics["loss"], epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    torch.save({"model_state": best_state, "model_config": model_config,
                                "source_tokens": source_vocab.itos, "target_tokens": target_vocab.itos,
                                "settings": NMT_SETTINGS, "vocabulary_budget": vocabulary_budget,
                                "seed": SEED, "best_epoch": best_epoch, "history": list(history)}, checkpoint_path)
                (NMT_OUTPUT_DIR / f"history_vocab{vocabulary_budget}.json").write_text(
                    json.dumps(history, indent=2), encoding="utf-8")
                print(f"Vocabulary {vocabulary_budget:,} | epoch {epoch:02d}/10 | "
                      f"train loss {train_metrics['loss']:.3f}, accuracy {train_metrics['accuracy']:.1%} | "
                      f"validation loss {valid_metrics['loss']:.3f}, accuracy {valid_metrics['accuracy']:.1%} | {elapsed:.1f} s")
                if epoch == 1:
                    print(f"First-epoch estimate for this ten-epoch model: {10 * elapsed / 60:.1f} minutes.")
            assert best_state is not None
            model.load_state_dict(best_state)
            model.eval()
            # Save the selected weights immediately after training, with all ten history rows.
            checkpoint = {"model_state": best_state, "model_config": model_config,
                          "source_tokens": source_vocab.itos, "target_tokens": target_vocab.itos,
                          "settings": NMT_SETTINGS, "vocabulary_budget": vocabulary_budget,
                          "max_words": NMT_SETTINGS["MAX_WORDS"], "seed": SEED,
                          "best_epoch": best_epoch, "history": history,
                          "tokenizer": "NFC lowercase; words with internal apostrophes; separate punctuation"}
            torch.save(checkpoint, checkpoint_path)
            validation_metrics = nmt_run_epoch(model, valid_loader)
            test_metrics = nmt_run_epoch(model, test_loader)
            translations = [{**example, **nmt_translate(example["english"], model, source_vocab, target_vocab)}
                            for example in nmt_five_examples]

            # Restore a genuinely new instance and check every displayed translation.
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            loaded_source_vocab = nmt_restore_vocab(saved["source_tokens"])
            loaded_target_vocab = nmt_restore_vocab(saved["target_tokens"])
            loaded_model = NmtSeq2Seq(**saved["model_config"]).to(device)
            loaded_model.load_state_dict(saved["model_state"])
            loaded_model.eval()
            reloaded_translations = [nmt_translate(example["english"], loaded_model,
                                                   loaded_source_vocab, loaded_target_vocab)
                                      for example in nmt_five_examples]
            assert reloaded_translations == [{key: row[key] for key in
                    ("translation", "ended_with_eos", "unknown_source_tokens")} for row in translations]
            result = {"vocabulary_budget": vocabulary_budget, "model_config": model_config,
                      "parameters": sum(parameter.numel() for parameter in model.parameters()),
                      "settings": dict(NMT_SETTINGS), "coverage": coverage, "history": history,
                      "best_epoch": best_epoch, "validation_metrics": validation_metrics,
                      "test_metrics": test_metrics, "translations": translations,
                      "reloaded_five_match": True, "checkpoint": checkpoint_path.name,
                      "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                      "elapsed_seconds": time.perf_counter() - started_all}
            (NMT_OUTPUT_DIR / f"results_vocab{vocabulary_budget}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Selected epoch {best_epoch} for vocabulary {vocabulary_budget:,}; "
                  "saved model and five-translation reload check passed.")
            return result

        nmt_results = []
        for nmt_budget in NMT_SETTINGS["vocabulary_budgets"]:
            nmt_results.append(nmt_train_experiment(nmt_budget))
            # Only plain result records are returned; model and optimizer are local.
            import gc
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        assert [result["vocabulary_budget"] for result in nmt_results] == [1000, 3000]
        '''),
        md("""
        ### Compare unknown words, accuracy and the five translations

        The first table compares coverage and validation accuracy at each model's selected epoch.<br>
        The second uses identical English inputs and Spanish references for both models.<br>
        `<unk>` remains visible in generated text, and the saved results record whether each decoder reached its end marker.<br>
        These five examples illustrate particular successes and errors; they are not a representative sentence-accuracy estimate.
        """),
        code(r'''
        # Report percentages with their definitions, then show the same five inputs side by side.
        nmt_summary_rows, nmt_coverage_rows = [], []
        for nmt_result in nmt_results:
            nmt_summary_rows.append({
                "vocabulary": nmt_result["vocabulary_budget"], "selected_epoch": nmt_result["best_epoch"],
                "English validation unknown %": 100 * nmt_result["coverage"]["validation"]["english"]["unknown_rate"],
                "Spanish validation unknown %": 100 * nmt_result["coverage"]["validation"]["spanish"]["unknown_rate"],
                "validation accuracy %": 100 * nmt_result["validation_metrics"]["accuracy"],
                "accuracy with UNK as error %": 100 * nmt_result["validation_metrics"]["accuracy_unk_as_error"],
            })
            for nmt_split_name, nmt_languages in nmt_result["coverage"].items():
                for nmt_language, nmt_values in nmt_languages.items():
                    nmt_coverage_rows.append({"vocabulary": nmt_result["vocabulary_budget"],
                                              "split": nmt_split_name, "language": nmt_language, **nmt_values})
        nmt_summary_table = pd.DataFrame(nmt_summary_rows)
        nmt_summary_table.to_csv(NMT_OUTPUT_DIR / "vocabulary_comparison.csv", index=False)
        pd.DataFrame(nmt_coverage_rows).to_csv(NMT_OUTPUT_DIR / "unknown_token_counts.csv", index=False)
        display(nmt_summary_table.round(2))

        nmt_comparison_rows = []
        for nmt_example_index, nmt_example in enumerate(nmt_five_examples):
            nmt_smaller, nmt_larger = (result["translations"][nmt_example_index] for result in nmt_results)
            assert nmt_smaller["english"] == nmt_larger["english"] == nmt_example["english"]
            assert nmt_smaller["reference"] == nmt_larger["reference"] == nmt_example["reference"]
            nmt_comparison_rows.append({"English": nmt_example["english"], "Spanish reference": nmt_example["reference"],
                                        "vocabulary 1000": nmt_smaller["translation"],
                                        "vocabulary 3000": nmt_larger["translation"],
                                        "1000 reached EOS": nmt_smaller["ended_with_eos"],
                                        "3000 reached EOS": nmt_larger["ended_with_eos"]})
        nmt_comparison_table = pd.DataFrame(nmt_comparison_rows)
        nmt_comparison_table.to_csv(NMT_OUTPUT_DIR / "five_translation_comparison.csv", index=False, encoding="utf-8")
        with pd.option_context("display.max_colwidth", None):
            display(nmt_comparison_table[["English", "Spanish reference", "vocabulary 1000", "vocabulary 3000"]])

        nmt_figure, nmt_axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        for nmt_column, nmt_plot_result in enumerate(nmt_results):
            nmt_epochs = [row["epoch"] for row in nmt_plot_result["history"]]
            for nmt_curve_split, nmt_curve_label in (("train", "Training"), ("valid", "Validation")):
                nmt_axes[0, nmt_column].plot(nmt_epochs, [row[nmt_curve_split]["loss"] for row in nmt_plot_result["history"]], label=nmt_curve_label)
                nmt_axes[1, nmt_column].plot(nmt_epochs, [row[nmt_curve_split]["accuracy"] for row in nmt_plot_result["history"]], label=nmt_curve_label)
            nmt_axes[0, nmt_column].set(title=f"Vocabulary {nmt_plot_result['vocabulary_budget']:,}: token loss", ylabel="Cross-entropy")
            nmt_axes[1, nmt_column].set(title=f"Vocabulary {nmt_plot_result['vocabulary_budget']:,}: token accuracy",
                                xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
            for nmt_plot_row in (0, 1):
                nmt_axes[nmt_plot_row, nmt_column].axvline(nmt_plot_result["best_epoch"], color="grey", linestyle="--", linewidth=1,
                                          label="Selected epoch")
                nmt_axes[nmt_plot_row, nmt_column].legend()
        nmt_figure.tight_layout()
        save_figure(nmt_figure, "problem3_nmt_training.png")
        nmt_changed_examples = sum(row["vocabulary 1000"] != row["vocabulary 3000"]
                                   for row in nmt_comparison_rows)
        print(f"Generated wording changed on {nmt_changed_examples} of the same five inputs.")
        print("Coverage counts, complete histories, references, translations and both selected models are saved.")
        '''),
    ]
