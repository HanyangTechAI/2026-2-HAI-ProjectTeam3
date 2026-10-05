"""Prepare official InsightFace v0.7 ONNX runtime subset, without face3d C++ extensions.

Official Python source files are downloaded unchanged, with their MIT license.
The detection/attribute weights are exactly those in buffalo_l.
"""
import argparse
from pathlib import Path
import zipfile
from .acquire import download
from .common import save_json, sha256

SOURCE = 'https://raw.githubusercontent.com/deepinsight/insightface/v0.7/'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root', type=Path, default=Path('models/insightface'))
    p.add_argument('--archive', type=Path, help='Already downloaded official buffalo_l.zip')
    args = p.parse_args()
    root = args.model_root
    runtime = root / 'python_runtime'
    records = []
    for sub in ('', 'model_zoo', 'utils'):
        dest = runtime / sub
        dest.mkdir(parents=True, exist_ok=True)
        (dest / '__init__.py').touch(exist_ok=True)
    for relative in ('model_zoo/scrfd.py', 'model_zoo/attribute.py', 'utils/face_align.py'):
        url = SOURCE + 'python-package/insightface/' + relative
        path = runtime / relative
        download(url, path)
        records.append(dict(url=url, sha256=sha256(path)))
    license_path = runtime / 'LICENSE'
    download(SOURCE + 'LICENSE', license_path)
    archive = args.archive or root / 'buffalo_l.zip'
    model_url = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'
    if not args.archive:
        download(model_url, archive)
    dest = root / 'models/buffalo_l'
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for basename in ('det_10g.onnx', 'genderage.onnx'):
            names = [n for n in z.namelist() if Path(n).name == basename]
            if len(names) != 1:
                raise ValueError(f'Expected one {basename} in official buffalo_l archive')
            contents = z.read(names[0])
            path = dest / basename
            if not path.exists():
                path.write_bytes(contents)
            elif path.read_bytes() != contents:
                raise ValueError(f'Existing weights differ: {path}')
            records.append(dict(model=basename, sha256=sha256(path)))
    save_json(root / 'provenance.json', dict(source_version='v0.7', source_files=records,
              weights_url=model_url, archive_sha256=sha256(archive), license='MIT for code; pretrained weights for noncommercial research'))
    print('Official buffalo_l runtime prepared:', root)


if __name__ == '__main__':
    main()
