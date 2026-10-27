"""Insert reviewed observations without rerunning or changing the experiments."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'e89-Schwartzberg-Paul-HW08.ipynb'
NOTES = ROOT / 'tools/notes'


def main():
    fingerprints = json.loads((NOTES / 'reviewed_results.json').read_text(encoding='utf-8'))
    for relative, expected in fingerprints.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, (
            'Review the changed experiment results before reusing these observations: ' + relative)
    nmt = [json.loads((ROOT / f'outputs/nmt/results_vocab{budget}.json').read_text(encoding='utf-8'))
           for budget in (1000, 3000)]
    small, large = nmt
    metrics = [result['validation_metrics'] for result in nmt]
    word_rates = [[100 * result['coverage']['validation'][language]['word_unknown_rate']
                   for language in ('english', 'spanish')] for result in nmt]
    quantitative = f'''### Reading the vocabulary results

Both models completed all ten epochs, and validation loss selected epoch **4** for each.<br>
Their validation loss then rises while training loss continues to fall, which indicates overfitting.<br>
I therefore use the saved epoch-4 weights for the comparison.

The main table follows the supplied notebook's unknown-token calculation, which includes punctuation.<br>
Counting words alone, with punctuation left out, English validation unknowns fall from **{word_rates[0][0]:.2f}%** to **{word_rates[1][0]:.2f}%**.<br>
Spanish validation unknown words fall from **{word_rates[0][1]:.2f}%** to **{word_rates[1][1]:.2f}%**.

The original validation token accuracy falls from **{100 * metrics[0]['accuracy']:.2f}%** to **{100 * metrics[1]['accuracy']:.2f}%**.<br>
This measure gives credit when a predicted `<unk>` matches an unknown target token.<br>
When `<unk>` is always counted as an error, accuracy instead rises from **{100 * metrics[0]['accuracy_unk_as_error']:.2f}%** to **{100 * metrics[1]['accuracy_unk_as_error']:.2f}%**, using the same target positions.<br>
The smaller vocabulary replaces more target words with `<unk>`, which helps explain why the two accuracy measures tell different stories.<br>
Both exclude padding and include the end marker; neither is the percentage of complete sentences translated correctly.

Increasing the vocabulary expands the word lookup tables and output layer.<br>
The model grows from **{small['parameters']:,}** to **{large['parameters']:,}** trainable parameters, while the embedding width, LSTM width and other training settings stay fixed.
'''
    stock_text = (NOTES / 'stock_discussion.md').read_text(encoding='utf-8').strip()
    nmt_text = quantitative.strip() + '\n\n' + (NOTES / 'nmt_discussion.md').read_text(encoding='utf-8').strip()
    original = NOTEBOOK.read_bytes()
    nb = nbformat.reads(original.decode('utf-8'), as_version=4)
    original_code = [cell.source for cell in nb.cells if cell.cell_type == 'code']
    for note_id, text, next_heading in [
            ('hw8-stock-observations', stock_text, '## Problem 3.'),
            ('hw8-nmt-observations', nmt_text, '## Sources')]:
        nb.cells = [cell for cell in nb.cells if cell.id != note_id]
        position = next(index for index, cell in enumerate(nb.cells)
                        if cell.cell_type == 'markdown' and cell.source.startswith(next_heading))
        nb.cells.insert(position, nbformat.v4.new_markdown_cell(text, id=note_id))
    sources = next(cell for cell in nb.cells if cell.cell_type == 'markdown'
                   and cell.source.startswith('## Sources'))
    if 'nasdaqtrader.com' not in sources.source:
        sources.source += ('<br>\nThe [Nasdaq trading calendar](https://nasdaqtrader.com/Trader.aspx?id=Calendar) '
                           'lists the September 7th market holiday.')
    assert original_code == [cell.source for cell in nb.cells if cell.cell_type == 'code']
    backup = Path(tempfile.gettempdir()) / 'hw08_notebook_backups'
    backup.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    shutil.copy2(NOTEBOOK, backup / f'before_notes_{stamp}.ipynb')
    assert hashlib.sha256(NOTEBOOK.read_bytes()).digest() == hashlib.sha256(original).digest()
    nbformat.write(nb, NOTEBOOK)
    print('Added reviewed stock and translation discussions; all code and results preserved.')


if __name__ == '__main__':
    main()
