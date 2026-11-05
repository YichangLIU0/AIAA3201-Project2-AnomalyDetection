"""Resumable parallel download of the official VisA archive (no model weights)."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import time

import requests

URL = 'https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar'
ROOT = Path(__file__).resolve().parents[1]


def download(destination, workers=8):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.head(URL, timeout=60)
    response.raise_for_status()
    size = int(response.headers['Content-Length'])
    etag = response.headers.get('ETag')
    chunk_size = 16 * 1024 * 1024
    partial = destination.with_suffix('.tar.partial')
    progress_file = destination.with_suffix('.download.json')
    if destination.exists():
        if destination.stat().st_size != size:
            raise ValueError('Existing archive has unexpected size')
        print('Archive already present', flush=True)
        return
    state = {'url': URL, 'size': size, 'etag': etag, 'chunk_size': chunk_size, 'completed': []}
    if progress_file.exists():
        old = json.loads(progress_file.read_text())
        if any(old[k] != state[k] for k in ['url', 'size', 'etag', 'chunk_size']):
            raise ValueError('Remote object or chunk size changed; use a new destination')
        if not partial.exists() or partial.stat().st_size != size:
            raise ValueError('Partial archive missing or wrong size')
        state = old
    else:
        with partial.open('wb') as stream:
            stream.truncate(size)
    completed = set(state['completed'])
    count = (size + chunk_size - 1) // chunk_size

    def fetch(index):
        first = index * chunk_size
        last = min(size - 1, first + chunk_size - 1)
        for attempt in range(6):
            try:
                with requests.get(URL, headers={'Range': f'bytes={first}-{last}', 'If-Match': etag},
                                  timeout=(30, 120), stream=True) as result:
                    result.raise_for_status()
                    if result.status_code != 206 or result.headers.get('Content-Range') != f'bytes {first}-{last}/{size}':
                        raise ValueError('Server did not honor exact byte range')
                    received = 0
                    with partial.open('r+b') as output:
                        output.seek(first)
                        for block in result.iter_content(1024 * 1024):
                            if received + len(block) > last - first + 1:
                                raise ValueError('Response exceeds requested range')
                            output.write(block)
                            received += len(block)
                    if received != last - first + 1:
                        raise ValueError('Incomplete range')
                return index
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 20))

    print(f'Downloading official archive: {size / 1e9:.2f} GB; {len(completed)}/{count} chunks cached', flush=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, i) for i in range(count) if i not in completed]
        for future in as_completed(futures):
            completed.add(future.result())
            state['completed'] = sorted(completed)
            progress_file.write_text(json.dumps(state), encoding='utf-8')
            print(f'DOWNLOAD {len(completed)}/{count} chunks, elapsed {time.monotonic() - started:.0f}s', flush=True)
    partial.replace(destination)
    checksum = hashlib.sha256()
    with destination.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            checksum.update(block)
    digest = checksum.hexdigest()
    metadata = {**state, 'sha256': digest, 'downloaded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    destination.with_suffix('.provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('DOWNLOAD COMPLETE', destination, digest, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT / 'data/downloads/VisA_20220922.tar')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    download(args.destination, args.workers)
