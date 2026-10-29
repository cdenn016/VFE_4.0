r"""Hard-coded chapter numbers of the main manuscript in the working documents, compared with the printed
chapter numbers of the compiled main PDF (input order in MAgent_Theory_2.0.tex).  Read-only."""
import re

from texscan import doc_files, read_lines

main_files = [f for f in doc_files("main") if f.startswith("chapters/")]
printed = {f: i + 1 for i, f in enumerate(main_files)}  # chapters print in input order 1..21
file_number = {}
for f in main_files:
    m = re.match(r"chapters/(\d+)_", f)
    file_number.setdefault(int(m.group(1)), []).append(f)

out = ["file number -> files (printed chapter):"]
for k in sorted(file_number):
    out.append(f"  {k:02d}: " + ", ".join(f"{f} (Ch {printed[f]})" for f in file_number[k]))

pat = re.compile(r"[Cc]hapters?~(\d+)(?:\s*(?:to|and)~(\d+))?\s+of the main manuscript")
total = 0
for doc in ("item1", "item2", "item3", "item7"):
    for f in doc_files(doc):
        for n, l in enumerate(read_lines(f), 1):
            for m in pat.finditer(l):
                total += 1
                nums = [int(x) for x in m.groups() if x]
                tgt = []
                for x in nums:
                    tgt.append("/".join(f"{g.split('/')[1]}=Ch {printed[g]}" for g in file_number.get(x, [])))
                out.append(f"{f}:{n}: '{m.group(0)}' -> file-number reading: {tgt}")
out.append(f"total hard-coded main-manuscript chapter references: {total}")
txt = "\n".join(out)
print(txt)
open(__file__.replace(".py", ".out.txt"), "w", encoding="utf-8").write(txt + "\n")
