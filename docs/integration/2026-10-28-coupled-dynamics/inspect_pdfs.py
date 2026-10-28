"""Render the affected manuscript pages and check physical text bounds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


def inspect(mode: str, stem: str, pages: list[int]) -> dict:
    """Render the selected one-based pages and inspect all text-span bounds."""
    pdf = ROOT / "build" / mode / f"{stem}.pdf"
    output = ROOT / "rendered" / mode
    output.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(pdf)
    violations = []
    for number, page in enumerate(document, 1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    rectangle = pymupdf.Rect(span["bbox"])
                    if (rectangle.x0 < -0.5 or rectangle.y0 < -0.5
                            or rectangle.x1 > page.rect.width + 0.5
                            or rectangle.y1 > page.rect.height + 0.5):
                        violations.append({"page": number, "text": span["text"],
                                           "bbox": span["bbox"]})
    assert not violations, violations
    thumbnails = []
    for number in pages:
        page = document[number - 1]
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2), alpha=False)
        pixmap.save(output / f"page-{number}.png")
        thumbnail = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        thumbnails.append((number, thumbnail))
    sheets = []
    for start in range(0, len(thumbnails), 4):
        group = thumbnails[start:start + 4]
        width = max(image.width for _, image in group)
        height = max(image.height for _, image in group) + 28
        sheet = Image.new("RGB", (2 * width, 2 * height), "#dddddd")
        draw = ImageDraw.Draw(sheet)
        for position, (number, thumbnail) in enumerate(group):
            left = (position % 2) * width
            top = (position // 2) * height
            draw.text((left + 12, top + 7), f"{mode}: PDF page {number}", fill="black")
            sheet.paste(thumbnail, (left, top + 28))
        filename = output / f"contact-{start // 4 + 1:02d}.png"
        sheet.save(filename)
        sheets.append(str(filename))
    return {"pdf": str(pdf), "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "pages": len(document), "rendered_pages": pages, "contact_sheets": sheets,
            "text_span_bounds_violations": violations}


def main() -> None:
    """Inspect each compiled PDF sequentially without any model computation."""
    result = {
        "main": inspect("main", "MAgent_Theory_2.0", [3, 4, 5, 6, 7, 8, *range(243, 262)]),
        "companion": inspect("companion", "MAgent_Theory_2.0_companion", [3, 4, 5, *range(98, 109)]),
    }
    (ROOT / "pdf-validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({mode: {"pages": item["pages"], "rendered": len(item["rendered_pages"]),
                             "sheets": len(item["contact_sheets"]),
                             "bounds_violations": len(item["text_span_bounds_violations"])}
                      for mode, item in result.items()}, indent=2))


if __name__ == "__main__":
    main()
