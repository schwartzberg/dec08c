"""Download Microsoft's daily Open prices for the two assignment periods."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
START = '2016-01-01'
END_EXCLUSIVE = '2026-10-11'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-raw', type=Path)
    args = parser.parse_args()
    start = int(datetime.fromisoformat(START).replace(tzinfo=timezone.utc).timestamp())
    end = int(datetime.fromisoformat(END_EXCLUSIVE).replace(tzinfo=timezone.utc).timestamp())
    url = 'https://query1.finance.yahoo.com/v8/finance/chart/MSFT?' + urllib.parse.urlencode(
        {'period1': start, 'period2': end, 'interval': '1d',
         'events': 'history', 'includeAdjustedClose': 'true'})
    if args.from_raw:
        raw = args.from_raw.read_bytes()
        retrieved = datetime.fromtimestamp(args.from_raw.stat().st_mtime, timezone.utc).isoformat()
    else:
        with urllib.request.urlopen(urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0'}), timeout=45) as response:
            raw = response.read()
        retrieved = datetime.now(timezone.utc).isoformat()
    result = json.loads(raw)['chart']['result'][0]
    assert result['meta']['symbol'] == 'MSFT'
    quotes = result['indicators']['quote'][0]
    rows = []
    for timestamp, opening in zip(result['timestamp'], quotes['open']):
        if opening is not None:
            # All daily MSFT opening timestamps fall on the same UTC and New York date.
            date = datetime.fromtimestamp(timestamp, timezone.utc).strftime('%Y-%m-%d')
            if START <= date < END_EXCLUSIVE:
                rows.append({'Date': date, 'Open': f'{opening:.8f}'})
    assert rows and len({r['Date'] for r in rows}) == len(rows)
    assert rows == sorted(rows, key=lambda row: row['Date'])
    assert rows[0]['Date'] == '2016-01-04' and rows[-1]['Date'] == '2026-10-09'
    development = [r for r in rows if r['Date'] <= '2026-09-06']
    test = [r for r in rows if '2026-09-09' <= r['Date'] <= '2026-10-10']
    gap = [r for r in rows if '2026-09-06' < r['Date'] < '2026-09-09']
    DATA.mkdir(exist_ok=True)
    (DATA / 'MSFT_yahoo_raw.json').write_bytes(raw)
    csv_records = {}
    for name, subset in [('all', rows), ('development', development), ('test', test)]:
        text = io.StringIO(newline='')
        writer = csv.DictWriter(text, fieldnames=['Date', 'Open'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(subset)
        content = text.getvalue().encode('utf-8')
        filename = f'MSFT_{name}.csv'
        (DATA / filename).write_bytes(content)
        csv_records[name] = {'filename': filename, 'rows': len(subset),
                             'first_date': subset[0]['Date'], 'last_date': subset[-1]['Date'],
                             'sha256': sha256(content)}
    record = {'symbol': 'MSFT', 'company': 'Microsoft', 'provider': 'Yahoo Finance',
              'source_url': url, 'source_page': 'https://finance.yahoo.com/quote/MSFT/history/',
              'retrieved_utc': retrieved, 'raw_sha256': sha256(raw),
              'source_field': 'chart.result[0].indicators.quote[0].open',
              'price_handling': 'Open field used directly; no dividend adjustment is applied.',
              'currency': result['meta'].get('currency'),
              'exchange_timezone': result['meta'].get('exchangeTimezoneName'),
              'requested_development': ['2016-01-01', '2026-09-06'],
              'requested_test': ['2026-09-09', '2026-10-10'],
              'gap_context_only': gap, 'files': csv_records}
    (DATA / 'MSFT_provenance.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
