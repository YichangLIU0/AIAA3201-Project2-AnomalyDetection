"""Explicitly freeze and verify the shared handoff's source/document versions."""
from datetime import datetime, timezone
import argparse
import hashlib
import json

from .protocol import PACKAGE, PROTOCOL_ID, sha256, save_json
from .predictions import source_version


def release_files():
    paths = set()
    for pattern in ['*.py', '*.md', 'requirements*.txt', 'protocol.json', 'tests/*.py', 'upstream/*']:
        paths.update(p for p in PACKAGE.glob(pattern) if p.is_file())
    paths.add(PACKAGE / 'manifests/manifest_lock.json')
    return sorted(paths)


def freeze():
    files = {p.relative_to(PACKAGE).as_posix(): sha256(p) for p in release_files()}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode('utf-8')).hexdigest()
    snapshot = {'protocol_id': PROTOCOL_ID, 'release_id': identity, 'runtime_source_sha256': source_version(),
                'created_utc': datetime.now(timezone.utc).isoformat(), 'files': files}
    save_json(PACKAGE / 'release.json', snapshot)
    return snapshot


def verify():
    snapshot = json.loads((PACKAGE / 'release.json').read_text(encoding='utf-8'))
    current = {p.relative_to(PACKAGE).as_posix(): sha256(p) for p in release_files()}
    if snapshot.get('protocol_id') != PROTOCOL_ID or current != snapshot['files']:
        raise ValueError('Handoff source/docs/manifest lock changed after release; review and explicitly freeze a new release')
    identity = hashlib.sha256(json.dumps(current, sort_keys=True).encode('utf-8')).hexdigest()
    if identity != snapshot['release_id'] or source_version() != snapshot['runtime_source_sha256']:
        raise ValueError('Release identity does not match source content')
    return snapshot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true', help='Maintainer only: freeze after all validation succeeds')
    args = parser.parse_args()
    result = freeze() if args.freeze else verify()
    print(result['protocol_id'], result['release_id'])
