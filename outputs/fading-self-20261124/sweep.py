"""Run from a fresh output directory with /venv/main/bin/python."""
import os
import shlex
import sys
from pathlib import Path

from PIL import Image, ImageDraw

prior = Path('/workspace/hai/outputs/age-edit-20261124/fading')
entry = Path('/root/.cache/fading-src/age_editing.py')
sys.path.insert(0, str(entry.parent))
values = (.50, .70, .76, .80, .84, .90)
setup, edits = entry.read_text().split('#%% age editing')
assert edits.count('self_replace_steps = .5') == 1
edits = edits.replace('self_replace_steps = .5', 'self_replace_steps = value')
out = Path.cwd() / 'images'
out.mkdir(exist_ok=False)
for line in (prior / 'commands.sh').read_text().splitlines():
    args = shlex.split(line)
    sys.argv = [str(entry), *args[args.index('--image_path'):]]
    src = Path(sys.argv[sys.argv.index('--image_path') + 1])
    pid = src.stem[:3]
    sys.argv[sys.argv.index('--save_aged_dir') + 1] = str(out / pid)
    ns = {'__name__': '__main__', '__file__': str(entry)}
    exec(compile(setup, str(entry), 'exec'), ns)
    assert ns['device'].type == 'cuda'
    sheet = Image.new('RGB', (4*192, len(values)*216), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, value in enumerate(values):
        dest = out / pid / f'{value:.2f}'
        dest.mkdir()
        ns.update(value=value, save_aged_dir=str(dest))
        exec(compile(edits, str(entry), 'exec'), ns)
        for col, (label, file) in enumerate([('source', src)] + [(str(a), dest / f'{src.stem}_{a}.png') for a in (10,30,50)]):
            with Image.open(file) as im:
                if col:
                    assert im.size == (512,512) and im.mode == 'RGB'
                tile = im.convert('RGB'); tile.thumbnail((192,192))
                xy = (col*192, row*216)
                sheet.paste(tile, xy)
                draw.text((xy[0]+3, xy[1]+194), f'{value:.2f} | {label}', fill='black')
    sheet.save(out.parent / f'{pid}.jpg', quality=92)
    del ns
