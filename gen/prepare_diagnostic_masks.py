"""Rasterize explicit source-space diagnostic polygons; not human ground truth."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from age.common import save_json, sha256
from .compare_lora import load_manifest
from .pipeline import read_image


def polygon_mask(size, polygons):
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    for points in polygons:
        if len(points) < 3 or any(len(p) != 2 or any(type(v) is not int for v in p) or
                                  not 0 <= p[0] < size[0] or not 0 <= p[1] < size[1] for p in points):
            raise ValueError('Polygons need at least three in-bounds integer coordinate pairs')
        draw.polygon([tuple(point) for point in points], fill=255)
    return mask


def prepare(manifest, annotations, output):
    samples = {s['sample_id']: s for s in load_manifest(manifest)['samples']}
    recipe = json.loads(Path(annotations).read_text(encoding='utf-8'))
    if recipe.get('author') != 'assistant_visual_diagnostic' or recipe.get('human_verified') is not False:
        raise ValueError('This utility records AI diagnostic annotations, not human-approved masks')
    if not recipe.get('samples') or set(recipe['samples']) - set(samples):
        raise ValueError('Annotations must select known validation sample IDs')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    prepared = {}
    for identifier, annotation in recipe['samples'].items():
        sample = samples[identifier]
        original = read_image(sample['input_path'])
        if list(original.size) != annotation['source_size']:
            raise ValueError('Annotation coordinates do not match the source dimensions')
        edit = polygon_mask(original.size, annotation['editable'])
        if not edit.getbbox():
            raise ValueError('Editable polygon is empty')
        protect = polygon_mask(original.size, annotation['protected'])
        # Feather inward only: zero support outside the explicitly drawn polygon.
        edit = Image.fromarray(np.minimum(np.asarray(edit), np.asarray(edit.filter(ImageFilter.GaussianBlur(3)))))
        entry = dict(input_sha256=sample['sha256'], source_size=list(original.size),
                     negative_prompt=annotation['negative_prompt'], notes=annotation['notes'])
        for name, mask in (('source_edit_mask', edit), ('source_protect_mask', protect)):
            path = output / f'{identifier}_{name}.png'
            mask.save(path)
            entry[name] = dict(path=str(path.resolve()), sha256=sha256(path))
        pixels = np.array(original).astype(np.float32)
        editable = np.asarray(edit) > 0
        protected = np.asarray(protect) > 0
        pixels[editable] = pixels[editable] * 0.65 + np.array([0, 255, 0]) * 0.35
        pixels[protected] = pixels[protected] * 0.5 + np.array([255, 0, 0]) * 0.5
        Image.fromarray(pixels.clip(0, 255).astype(np.uint8)).save(output / f'{identifier}_overlay.png')
        prepared[identifier] = entry
    record = dict(schema=1, coordinate_space='source', author=recipe['author'], human_verified=False,
                  annotations_sha256=sha256(annotations), source_manifest_sha256=sha256(manifest),
                  note='Diagnostic approximations only; green=editable, red=protected. Pipeline still applies face/other-face guards.',
                  samples=prepared)
    save_json(output / 'annotations.json', recipe)
    save_json(output / 'manifest.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.annotations, args.output), indent=2))


if __name__ == '__main__':
    main()
