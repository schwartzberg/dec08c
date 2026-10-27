"""Keep currency amounts as text instead of accidental Markdown mathematics."""
from pathlib import Path
import ast
import re
import nbformat

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r'(?<!\\)\$(?=\d)')


def escaped(text):
    return PATTERN.sub(r'\\$', text)


def prose_currency(text):
    return re.sub(r'(?:\\)?\$(?=\d)', 'USD ', text)


def main():
    path = ROOT / 'e89-Schwartzberg-Paul-HW08.ipynb'
    nb = nbformat.read(path, as_version=4)
    replacements = 0
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            replacements += len(PATTERN.findall(cell.source))
            cell.source = prose_currency(cell.source)
        elif cell.cell_type == 'code':
            # Two source backslashes produce one literal escape in the Markdown output.
            cell.source = cell.source.replace('**$', '**' + '\\\\' + '$')
            ast.parse(cell.source)
            for output in cell.get('outputs', []):
                data = output.get('data', {})
                if isinstance(data.get('text/markdown'), str):
                    replacements += len(PATTERN.findall(data['text/markdown']))
                    data['text/markdown'] = escaped(data['text/markdown'])
    nb.metadata['presentation_edits_after_execution'] = [
        'Formatted currency as literal text; numerical results and model computations are unchanged.']
    nbformat.write(nb, path)
    section = ROOT / 'tools/section_stocks.py'
    source = section.read_text(encoding='utf-8').replace('**$', '**' + '\\\\' + '$')
    ast.parse(source)
    section.write_text(source, encoding='utf-8')
    for note in (ROOT / 'tools/notes').glob('*.md'):
        note.write_text(prose_currency(note.read_text(encoding='utf-8')), encoding='utf-8')
    print(f'Corrected {replacements} currency amounts for Markdown display.')


if __name__ == '__main__':
    main()
