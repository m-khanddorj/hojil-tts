"""Local artifact and explicitly public download utilities."""
import hashlib
import json
import shutil
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_rows(path, rows):
    with Path(path).open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def read_rows(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def external(path):
    """Require generated data outside any Git working tree, including this source tree."""
    p = Path(path).expanduser().resolve()
    for parent in (p, *p.parents):
        if (parent / '.git').exists():
            raise ValueError(f'Data must be outside Git: {p}')
    return p


def fresh_dir(path):
    p = external(path)
    p.mkdir(parents=True, exist_ok=False)
    return p


def child(root, relative):
    p = (Path(root) / relative).resolve()
    if not p.is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes source directory')
    return p


def extract(archive, dest, max_bytes=100 * 1024**3):
    """Only regular files/directories; no links, devices, traversal, or overwrites."""
    dest = Path(dest)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        if sum(m.size for m in members) > max_bytes:
            raise ValueError('Archive exceeds extraction size limit')
        seen = set()
        for m in members:
            if Path(m.name).is_absolute():
                raise ValueError("Absolute archive path")
            target = child(dest, m.name)
            if target in seen or not (m.isfile() or m.isdir()) or target.exists():
                raise ValueError(f'Unsafe or duplicate archive member: {m.name}')
            seen.add(target)
        file_paths = {child(dest, m.name) for m in members if m.isfile()}
        if any(parent in file_paths for path in seen for parent in path.parents):
            raise ValueError('Archive file/directory collision')
        for m in members:
            target = child(dest, m.name)
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, target.open('xb') as out:
                    shutil.copyfileobj(src, out)


def download(url, target, expected=None):
    """Public HTTPS only: no auth headers, cookie jar, token lookup, or signed URLs."""
    u = urllib.parse.urlsplit(url)
    if u.scheme != 'https' or u.username or u.password or u.query or u.fragment:
        raise ValueError('Use a public HTTPS URL without credentials/query/fragment')
    target = Path(target)
    if target.exists():
        if expected and digest(target) == expected:
            return
        raise ValueError(f'Existing download needs matching checksum: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + '.partial')
    try:
        with urllib.request.urlopen(url, timeout=60) as src, tmp.open('xb') as out:
            shutil.copyfileobj(src, out)
            size = src.headers.get('Content-Length')
            if size is not None and out.tell() != int(size):
                raise ValueError('Incomplete download: Content-Length mismatch')
        if expected and digest(tmp) != expected:
            raise ValueError('Download checksum mismatch')
        tmp.rename(target)
    finally:
        tmp.unlink(missing_ok=True)
