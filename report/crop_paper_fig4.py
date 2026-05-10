"""Crop the Fig 4 (right top) ASR vs poison budget panel from page 7."""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / 'figures'

src = OUT / 'paper_p7.png'
img = Image.open(src)
print(f'Source: {img.size}')  # (1700, 2200)

# Crop Fig 4 top-right (ASR vs budget)
# The figure occupies ~ left=920, right=1700, top=10, bottom=600
left, top, right, bottom = 940, 200, 1700, 590
fig4_right = img.crop((left, top, right, bottom))
out = OUT / 'paper_fig4_right_asr.png'
fig4_right.save(out)
print(f'Cropped Fig 4 right -> {out} ({fig4_right.size})')
