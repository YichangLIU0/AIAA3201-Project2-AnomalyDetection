"""Same-path, byte-identical hardlink storage for immutable prediction maps."""
from pathlib import Path
import os
import uuid

from shared_protocol.protocol import sha256


class MapStorage:
    def __init__(self, baseline_root, output_root):
        import json
        self.output = Path(output_root).resolve(strict=True)
        self.roots = [Path(baseline_root).resolve(strict=True), self.output]
        self.index, self.verified = {}, {}
        self.shared_files = self.saved_bytes = self.new_files = 0
        for root in self.roots:
            for path in sorted(root.glob('*/*/run.json')):
                run = json.loads(path.read_text(encoding='utf-8'))
                if run['status'] != 'complete':
                    continue
                for row in run['test']:
                    candidate = (path.parent / row['map']).resolve(strict=True)
                    if not candidate.is_relative_to(root):
                        raise ValueError('Prediction map outside declared root')
                    self.index.setdefault(row['map_sha256'], candidate)

    @staticmethod
    def fingerprint(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def share(self, path, expected_sha):
        """Called only after PredictionWriter saved and hashed a fresh map."""
        original = Path(path)
        if original.is_symlink():
            raise ValueError('Refuse symbolic links')
        target = original.resolve(strict=True)
        if target.suffix != '.npy' or not target.is_relative_to(self.output):
            raise ValueError('Storage replacement outside new prediction root')
        state = self.fingerprint(target)
        source = self.index.get(expected_sha)
        if source is None:
            self.index[expected_sha] = target
            self.verified[expected_sha] = (target, state)
            self.new_files += 1
            return
        source_state = self.fingerprint(source)
        if source_state[:2] == state[:2]:
            return
        if source_state[0] != state[0]:
            raise ValueError('Hardlink requires a common filesystem')
        if self.verified.get(expected_sha) != (source, source_state):
            if sha256(source) != expected_sha or self.fingerprint(source) != source_state:
                raise ValueError('Canonical prediction content changed')
            self.verified[expected_sha] = (source, source_state)
        if source_state[2] != state[2] or self.fingerprint(target) != state:
            raise ValueError('Unexpected prediction size or concurrent modification')
        temp = target.with_name(target.name + '.link-' + uuid.uuid4().hex + '.tmp')
        try:
            os.link(source, temp)
            if self.fingerprint(temp)[:3] != source_state[:3] or self.fingerprint(target) != state:
                raise ValueError('Temporary hardlink or target changed')
            os.replace(temp, target)
        finally:
            if temp.exists():
                temp.unlink()
        if self.fingerprint(target)[:3] != source_state[:3]:
            raise ValueError('Replacement did not preserve canonical bytes')
        self.shared_files += 1
        self.saved_bytes += state[2]

    def stats(self):
        return dict(shared_files=self.shared_files, saved_bytes=self.saved_bytes,
                    new_unique_files=self.new_files)
