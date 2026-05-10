"""Render the page containing Fig 4 from the paper PDF and crop to the figure region.

Paper Fig 4 spans most of page 7 (top: poisoned image samples; bottom-left: bird vote bar chart;
bottom-right: ASR vs poison budget). We want the bottom-right "ASR vs poison budget" panel.
"""
from pathlib import Path
import fitz  # pymupdf

ROOT = Path(__file__).parent.parent
PDF = ROOT / '2004.00225v2.pdf'
OUT = Path(__file__).parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

doc = fitz.open(str(PDF))
print(f'PDF has {len(doc)} pages')

# Fig 4 caption is around page index 6 (page 7 in 1-indexed)
# Render at high DPI for cropping
DPI = 200
zoom = DPI / 72  # PDF default 72 dpi
mat = fitz.Matrix(zoom, zoom)

# Render candidate pages
for page_idx in range(5, 8):
    page = doc[page_idx]
    pix = page.get_pixmap(matrix=mat)
    img_path = OUT / f'paper_p{page_idx+1}.png'
    pix.save(str(img_path))
    print(f'Rendered page {page_idx+1} -> {img_path} ({pix.width}×{pix.height})')

doc.close()
