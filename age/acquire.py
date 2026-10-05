"""Acquire public age datasets and verify archives without replacing source files."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
from .common import save_json, sha256
from .prep import extract_archives


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    partial = path.with_suffix(path.suffix + '.partial')
    request = urllib.request.Request(url, headers={'User-Agent': 'age-research-preparation'})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as out:
        shutil.copyfileobj(response, out)
    partial.replace(path)


def extract_tar(archive, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile() and not member.isdir():
                raise ValueError('Archive contains a nonregular entry')
            tar.extract(member, output, filter='data')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=['utkface', 'afad'], required=True)
    p.add_argument('--root', type=Path, default=Path('data/raw'))
    p.add_argument('--utk-url', help='Current official Drive file URL (not folder); or use --archive')
    p.add_argument('--archive', type=Path, help='Already downloaded UTKFace ZIP/tar.gz')
    p.add_argument('--parts-dir', type=Path, help='Already downloaded AFAD-Full.tar.xz* split files')
    args = p.parse_args()
    root = args.root / args.source
    root.mkdir(parents=True, exist_ok=True)
    archives = root / 'archives'
    archives.mkdir(exist_ok=True)
    if args.source == 'utkface':
        if args.archive:
            archive = args.archive
        elif args.utk_url:
            import gdown
            archive = archives / 'UTKFace.tar.gz'
            if not archive.exists():
                result = gdown.download(args.utk_url, str(archive), fuzzy=True)
                if not result:
                    raise RuntimeError('Official Drive download failed; no data imported')
        else:
            p.error('Supply --archive or the official --utk-url')
        if tarfile.is_tarfile(archive):
            extract_tar(archive, root / 'images')
        else:
            extract_archives(archive, root / 'images')
        files = [archive]
    else:
        if args.parts_dir:
            files = sorted(args.parts_dir.glob('AFAD-Full.tar.xz??'))
        else:
            url = 'https://api.github.com/repos/John-niu-07/tarball/contents/'
            with urllib.request.urlopen(url, timeout=60) as response:
                entries = json.load(response)
            files = []
            for entry in sorted(entries, key=lambda e: e['name']):
                if entry['name'].startswith('AFAD-Full.tar.xz'):
                    dest = archives / entry['name']
                    download(entry['download_url'], dest)
                    files.append(dest)
        if not files:
            raise ValueError('No AFAD split archives found')
        archive = archives / 'AFAD-Full.tar.xz'
        partial = archive.with_suffix('.partial')
        with partial.open('wb') as out:
            for part in files:
                with part.open('rb') as f:
                    shutil.copyfileobj(f, out)
        # A missing/truncated piece fails tar/xz integrity verification here.
        extract_tar(partial, root / 'images')
        partial.replace(archive)
    save_json(root / 'acquisition.json', dict(source=args.source,
              archives=[dict(name=f.name, bytes=f.stat().st_size, sha256=sha256(f)) for f in files]))
    print(f'Imported {args.source} into {root / "images"}')


if __name__ == '__main__':
    main()
