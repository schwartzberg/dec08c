"""Build the single HW8 submission with all solution code inline."""
from pathlib import Path
import ast
import sys
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
NOTEBOOK = ROOT / 'e89-Schwartzberg-Paul-HW08.ipynb'


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def build():
    from section_stocks import make_cells as stock_cells
    from section_nmt import make_cells as nmt_cells
    cells = [md('''
    # Assignment 08: RNNs, LSTMs and sequence-to-sequence translation
    **Schwartzberg, Paul**

    Problems 1 and 2 use Microsoft opening prices to compare recurrent forecasting models.<br>
    Problem 3 compares the supplied English-to-Spanish translator with vocabulary budgets of 1,000 and 3,000 tokens.<br>
    The notebook contains the code, figures, measured results and explanations for all three problems.

    Run the cells in order in a Colab GPU runtime.<br>
    The first run downloads the small data files; running the training cells trains the models again.<br>
    The stock experiments retain 100 epochs per model, and each translator retains 10 epochs.
    '''), md('''
    ## Setup and data sources

    I use the supplied PyTorch implementations and save reloadable PyTorch checkpoints.<br>
    Seed 42 controls the data splits, initial weights and shuffled training batches.<br>
    Results can still differ between devices and software versions, so I record the environment used for this run.

    The Microsoft data were downloaded from Yahoo Finance on October 27th 2026.<br>
    I use its daily `Open` field directly, without applying dividend adjustments.<br>
    The repository keeps the provider response, the requested periods and file checksums.<br>
    This notebook retrieves that saved copy when the local files are missing.
    '''), code('''
    # Import the numerical tools and install only a missing helper library.
    import importlib.util
    import subprocess
    import sys
    for module, package in [("numpy", "numpy"), ("pandas", "pandas"),
                            ("matplotlib", "matplotlib"), ("sklearn", "scikit-learn")]:
        if importlib.util.find_spec(module) is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)

    from pathlib import Path
    from datetime import datetime, timezone
    from collections import Counter
    import copy
    import gc
    import hashlib
    import json
    import math
    import platform
    import random
    import re
    import time
    import unicodedata
    import urllib.request
    import zipfile

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.nn import functional as F
    from IPython.display import display, Markdown

    get_ipython().run_line_magic("matplotlib", "inline")
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path("outputs")
    FIGURE_DIR = OUTPUT_DIR / "figures"
    DATA_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.max_colwidth", None)
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

    def save_figure(fig, filename):
        """Keep each figure on disk and embed it in the notebook output."""
        fig.savefig(FIGURE_DIR / filename, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "numpy": np.__version__, "pandas": pd.__version__, "seed": SEED,
                   "device": str(device),
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    (OUTPUT_DIR / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(environment)
    '''), code('''
    # Retrieve the exact Microsoft data snapshot if this notebook is opened on its own.
    STOCK_DATA_COMMIT = "29a161691055455a1fccfa5b03dadb07dcba13c6"
    STOCK_DATA_BASE = f"https://raw.githubusercontent.com/schwartzberg/dec08c/{STOCK_DATA_COMMIT}/data/"
    stock_file_checksums = {
        "MSFT_all.csv": "ba6c8d2c4f33d1aa7b9e0d1e8d75231e3a868067752e1c95d5b9bccc2712ff68",
        "MSFT_development.csv": "694bf16d174118e12d58ee77eff27c595556ad7cbd564db12e282fd916b96f37",
        "MSFT_test.csv": "499eb5077c8078e9469129249beb857e2bfa8608d4f1f5aee0f2f0f3137a468c",
        "MSFT_yahoo_raw.json": "5a682a7a774760cdddcee3ad2d510bbe16352c3920cfa0578781f516eaf51b57",
    }
    for filename, expected_hash in stock_file_checksums.items():
        path = DATA_DIR / filename
        if not path.exists():
            urllib.request.urlretrieve(STOCK_DATA_BASE + filename, path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash, filename
    provenance_path = DATA_DIR / "MSFT_provenance.json"
    if not provenance_path.exists():
        urllib.request.urlretrieve(STOCK_DATA_BASE + provenance_path.name, provenance_path)
    stock_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert stock_provenance["symbol"] == "MSFT"
    assert stock_provenance["raw_sha256"] == stock_file_checksums["MSFT_yahoo_raw.json"]
    print("Provider:", stock_provenance["provider"], "| downloaded:", stock_provenance["retrieved_utc"])
    print("Verified four data-file checksums. The source response is retained for checking.")
    ''')]
    cells += stock_cells() + nmt_cells()
    cells.append(md('''
    ## Sources

    The assignment, lecture slides and supplied RNN, LSTM and `3_NMT.ipynb` notebooks guided these solutions.<br>
    The stock-price snapshot comes from [Yahoo Finance](https://finance.yahoo.com/quote/MSFT/history/).<br>
    The parallel sentences come from [Tatoeba through ManyThings](https://www.manythings.org/anki/), using the archive and attribution recorded in Problem 3.<br>
    [PyTorch's RNN and LSTM documentation](https://docs.pytorch.org/docs/stable/nn.html#recurrent-layers) describes the recurrent layers used here.<br>
    [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) describes the fitted price transformation.
    '''))
    for i, cell in enumerate(cells):
        if cell.cell_type == 'code':
            ast.parse(cell.source, filename=f'cell {i + 1}')
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'},
        'language_info': {'name':'python'}, 'accelerator':'GPU',
        'colab': {'name':NOTEBOOK.name, 'provenance':[]}})
    if NOTEBOOK.exists():
        raise FileExistsError('Preserve the existing notebook before rebuilding.')
    nbf.write(nb, NOTEBOOK)
    print(f'Built {len(cells)} cells; all Python cells parse.')


if __name__ == '__main__':
    build()
