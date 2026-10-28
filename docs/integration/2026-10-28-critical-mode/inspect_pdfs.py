"""Render affected pages and check document bounds and final TeX diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
MANUSCRIPT = Path(
    "C:/Users/chris and christine/.codex/worktrees/"
    "coupled-dynamics-publication/Research/manuscripts/MAgent_Theory_2.0"
)


def inspect(stem: str, pages: list[int]) -> dict:
    """Check all physical text bounds and render selected one-based pages."""
    pdf = MANUSCRIPT / "build" / f"{stem}.pdf"
    document = pymupdf.open(pdf)
    violations = []
    for number, page in enumerate(document, 1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    x0, y0, x1, y1 = span["bbox"]
                    if x0 < -0.5 or y0 < -0.5 or x1 > page.rect.width + 0.5 or y1 > page.rect.height + 0.5:
                        violations.append({"page": number, "text": span["text"]})
    rendered = ROOT / "rendered" / stem
    rendered.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    images = []
    for number in pages:
        prefix = rendered / f"page-{number}"
        subprocess.run(
            ["pdftoppm", "-f", str(number), "-l", str(number), "-singlefile", "-r", "85", "-png", str(pdf), str(prefix)],
            check=True, capture_output=True, env=environment,
        )
        images.append((number, Image.open(prefix.with_suffix(".png")).convert("RGB")))
    contacts = []
    for start in range(0, len(images), 4):
        group = images[start:start + 4]
        width = max(im.width for _, im in group)
        height = max(im.height for _, im in group) + 25
        sheet = Image.new("RGB", (2 * width, 2 * height), "#dddddd")
        draw = ImageDraw.Draw(sheet)
        for offset, (number, im) in enumerate(group):
            x, y = (offset % 2) * width, (offset // 2) * height
            draw.text((x + 10, y + 6), f"PDF page {number}", fill="black")
            sheet.paste(im, (x, y + 25))
        target = rendered / f"contact-{start // 4 + 1}.png"
        sheet.save(target)
        contacts.append(str(target))
    log = (MANUSCRIPT / "build" / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
    unresolved = [line for line in log.splitlines() if re.search(r"undefined|multiply defined", line)]
    result = {
        "pdf": str(pdf), "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "pages": len(document), "rendered_pages": pages, "contact_sheets": contacts,
        "physical_bounds_violations": violations, "unresolved_references": unresolved,
        "overfull_count": len(re.findall(r"Overfull ", log)),
        "underfull_count": len(re.findall(r"Underfull ", log)),
    }
    assert not violations and not unresolved, result
    return result


def main() -> None:
    """Inspect the two PDFs sequentially, without numerical model computation."""
    result = {
        "main": inspect("MAgent_Theory_2.0", [8, *range(262, 267), 275, 301, 302]),
        "companion": inspect("MAgent_Theory_2.0_companion", [4, *range(109, 113), 120, 121]),
    }
    (ROOT / "pdf-validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: {k: v for k, v in item.items() if k not in ("contact_sheets", "pdf")} for key, item in result.items()}, indent=2))


if __name__ == "__main__":
    main()
