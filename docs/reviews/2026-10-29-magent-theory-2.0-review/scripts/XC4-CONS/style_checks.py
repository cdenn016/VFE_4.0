r"""Style consistency checks across the seven documents (read-only).

1. Sentence-initial lowercase \cref (prints 'proposition 3.2 ...' at the start of a sentence).
2. British spellings in the scientific text (AUTHORING: American English).
3. Banned promotional words from the user's writing rules (counts only).
4. Chapter-title capitalization (title case versus sentence case).
"""
import collections
import pathlib
import re

from texscan import DOCS, doc_files, read_lines

files = []
for d in DOCS:
    for f in doc_files(d):
        if f not in files and f.endswith(".tex"):
            files.append(f)
out = []

# 1. sentence-initial \cref
hits = []
for f in files:
    text = "\n".join(read_lines(f))
    for m in re.finditer(r"(^|[.?!:]\s+|\n\n)\\cref\{([^}]*)\}", text):
        line = text.count("\n", 0, m.start()) + 1
        hits.append(f"{f}:{line}: \\cref{{{m.group(2)}}}")
out.append(f"1. sentence-initial lowercase \\cref: {len(hits)}")
out.extend("   " + h for h in hits[:60])

# 2. British spellings
brit = r"\b(behaviour\w*|colour\w*|favour\w*|neighbour\w*|centre\w*|organis\w*|normalis\w*|realis\w*|recognis\w*|characteris\w*|generalis\w*|parameteris\w*|minimis\w*|maximis\w*|optimis\w*|summaris\w*|analys(e|es|ed|ing)\b|modell\w*|labell\w*|travell\w*|programme\w*|licence|defence|marginalis\w*|symmetris\w*|regularis\w*|linearis\w*|stabilis\w*|randomis\w*|emphasis(e|ed|es)\b|whilst|amongst)"
bh = collections.Counter()
ex = []
for f in files:
    for n, line in enumerate(read_lines(f), 1):
        clean = re.sub(r"\\cite[a-z]*(\[[^\]]*\])?\{[^}]*\}", "", line)
        for m in re.finditer(brit, clean, re.I):
            bh[f] += 1
            ex.append(f"{f}:{n}: {m.group(0)} ...{clean[max(0, m.start() - 60):m.end() + 30]}...")
out.append(f"2. British spellings: {sum(bh.values())} in {dict(bh)}")
out.extend("   " + e for e in ex[:40])

# 3. banned words
banned = r"\b(key insight|crucially|critically|notably|importantly|fundamentally|essentially|leverage\w*|underscore\w*|delve\w*|tapestry|landscape|realm|pivotal|paramount|intricate|multifaceted|nuanced|holistic|seamless\w*|showcase\w*|foster\w*|bolster\w*|harness\w*|unlock\w*|elevate\w*|in essence|at its core|it is important to note|it's worth noting|worth noting|a testament to|plays a crucial role|shed light)\b"
wh = collections.Counter()
wex = []
for f in files:
    for n, line in enumerate(read_lines(f), 1):
        for m in re.finditer(banned, line, re.I):
            wh[(f, m.group(0).lower())] += 1
            wex.append(f"{f}:{n}: {m.group(0)} ...{line[max(0, m.start() - 60):m.end() + 40]}...")
out.append(f"3. banned words: {sum(wh.values())}")
out.extend("   " + e for e in wex[:40])

# 4. chapter titles
small = {"a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "under", "with", "from", "by", "as", "at", "its", "is", "not", "that", "than", "into", "across", "through"}
out.append("4. chapter titles in sentence case (a non-initial content word starts lowercase):")
for f in files:
    for n, line in enumerate(read_lines(f), 1):
        m = re.search(r"\\chapter\*?\{([^}]*)\}", line)
        if not m:
            continue
        words = re.findall(r"[A-Za-z][A-Za-z\-']*", m.group(1))
        lower = [w for w in words[1:] if w[0].islower() and w.lower() not in small]
        if lower:
            out.append(f"   {f}:{n}: '{m.group(1)}' lowercase words {lower}")

txt = "\n".join(out)
pathlib.Path(__file__).with_name("style_checks.out.txt").write_text(txt, encoding="utf-8")
print(txt)
