"""Execute HW8 in a fresh kernel and export its saved results as HTML."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import shutil
import tempfile
import time
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / 'e89-Schwartzberg-Paul-HW08.ipynb'


class ProgressClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg.get('msg_type') == 'stream':
            print(msg['content']['text'], end='', flush=True)
        return super().process_message(msg, cell, cell_index)


def export_html(notebook):
    html, _ = HTMLExporter().from_notebook_node(notebook, resources={
        'metadata': {'name':'Assignment 08 - Paul Schwartzberg'}})
    NOTEBOOK.with_suffix('.html').write_text(
        '\n'.join(line.rstrip() for line in html.splitlines()) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html-only', action='store_true')
    parser.add_argument('--kernel', default=os.environ.get('HW8_KERNEL', 'python3'))
    args = parser.parse_args()
    original = NOTEBOOK.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    nb = nbformat.reads(original.decode('utf-8'), as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
    if args.html_only:
        export_html(nb)
        print('HTML exported without running notebook cells.')
        return
    backup = Path(tempfile.gettempdir()) / 'hw08_notebook_backups'
    backup.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    shutil.copy2(NOTEBOOK, backup / f'{NOTEBOOK.stem}_{stamp}.ipynb')
    output_dir = ROOT / 'outputs'
    output_dir.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    tick = time.perf_counter()

    def save_progress(**kwargs):
        nbformat.write(nb, output_dir / 'execution_in_progress.ipynb')

    client = ProgressClient(nb, kernel_name=args.kernel, timeout=14400,
                            resources={'metadata':{'path':str(ROOT)}},
                            on_cell_executed=save_progress)
    try:
        client.execute()
    except Exception:
        nbformat.write(nb, output_dir / 'failed_execution.ipynb')
        print('Failed run preserved in outputs/failed_execution.ipynb', flush=True)
        raise
    if hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest() != digest:
        nbformat.write(nb, backup / f'executed_{stamp}.ipynb')
        raise RuntimeError('The notebook changed during execution; its edits were preserved.')
    code_cells = [c for c in nb.cells if c.cell_type == 'code']
    assert [c.execution_count for c in code_cells] == list(range(1, len(code_cells) + 1))
    assert not any(o.output_type == 'error' for c in code_cells for o in c.outputs)
    for cell in code_cells:
        last = ast.parse(cell.source).body[-1]
        if isinstance(last, ast.Expr) and isinstance(last.value, ast.Call) and isinstance(last.value.func, ast.Attribute) and last.value.func.attr == 'write_text':
            cell.outputs = [o for o in cell.outputs if o.output_type != 'execute_result']
    record = {'started_utc':started, 'finished_utc':datetime.now(timezone.utc).isoformat(),
              'elapsed_seconds':time.perf_counter()-tick, 'kernel':args.kernel,
              'fresh_kernel_run_all':True, 'code_cells':len(code_cells)}
    nb.metadata['hw8_execution'] = record
    nbformat.write(nb, NOTEBOOK)
    export_html(nb)
    (output_dir / 'execution_log.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'All {len(code_cells)} code cells passed; notebook and HTML saved.', flush=True)


if __name__ == '__main__':
    main()
