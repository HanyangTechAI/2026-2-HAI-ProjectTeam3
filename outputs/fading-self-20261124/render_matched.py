"""Render matched framing from existing PNGs; no diffusion inference."""
import shlex
from pathlib import Path

from PIL import Image, ImageDraw
from null_inversion import load_512

root = Path(__file__).resolve().parent
out = root / 'matched'
out.mkdir(exist_ok=True)
preview = Image.new('RGB', (768, 3 * 216), 'white')
commands = root.parent / 'age-edit-20261124/fading/commands.sh'
for index, line in enumerate(commands.read_text().splitlines()):
    args = shlex.split(line)
    src = Path(args[args.index('--image_path') + 1])
    pid = src.stem[:3]
    source = Image.fromarray(load_512(str(src)))
    sheet = Image.new('RGB', (768, 6 * 216), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, value in enumerate(('0.50', '0.70', '0.76', '0.80', '0.84', '0.90')):
        for col, label in enumerate(('input', '10', '30', '50')):
            if col:
                with Image.open(root / 'images' / pid / value / f'{src.stem}_{label}.png') as im:
                    tile = im.resize((192, 192))
            else:
                tile = source.resize((192, 192))
            sheet.paste(tile, (col * 192, row * 216))
            draw.text((col * 192 + 3, row * 216 + 194), f'{pid} | {value} | {label}', fill='black')
        if value == '0.70':
            preview.paste(sheet.crop((0, row * 216, 768, (row + 1) * 216)), (0, index * 216))
    sheet.save(out / f'{pid}.png')
preview.save(out / 'booth-070.png')
