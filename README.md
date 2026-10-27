# Assignment 08: RNNs, LSTMs and sequence-to-sequence translation

Schwartzberg, Paul

The submission is `e89-Schwartzberg-Paul-HW08.ipynb` and its matching HTML export.
All three solutions are in this notebook.

## Run

[Open in Colab](https://colab.research.google.com/github/schwartzberg/dec08c/blob/main/e89-Schwartzberg-Paul-HW08.ipynb),
select a GPU runtime and run the cells in order.
The notebook retrieves the saved Microsoft price snapshot and the supplied
English–Spanish corpus when those files are missing.
Its solution code is inline; it does not need the assembly scripts in `tools/`.

To run a fresh notebook kernel and export HTML:

```text
python run_all.py --kernel python3
```

To refresh the HTML after editing only the text:

```text
python run_all.py --html-only
```

## Data and saved results

`data/MSFT_provenance.json` records the provider, retrieval time, requested dates
and checksums for the Microsoft data.
The provider response is saved in `data/MSFT_yahoo_raw.json`.
`tools/fetch_msft.py` records how those prices were downloaded.
The small stock files are included so the experiment uses the same observations
when rerun later.

`outputs/stocks/` contains the RNN and LSTM checkpoints, fitted scalers,
training histories, prediction tables and experiment results.
`outputs/nmt/` contains both translation checkpoints, vocabularies, data split,
unknown-token counts, histories and translations.
`outputs/figures/` contains the figures embedded in the submission.

`tools/validate_submission.py` checks the saved evidence without training again.
