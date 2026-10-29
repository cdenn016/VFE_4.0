"""Lint B-FOUND files: environment balance, banned spacing macros, em dashes, control chars, duplicate labels across the manuscript."""
import glob
import re
import sys
from collections import Counter

root = sys.argv[1]
files = sys.argv[2:]
for f in files:
    s = open(f"{root}/{f}", encoding='utf8').read()
    begins = Counter(re.findall(r'\\begin\{(\w+\*?)\}', s))
    ends = Counter(re.findall(r'\\end\{(\w+\*?)\}', s))
    bad = [c for c in s if ord(c) < 32 and c not in '\n\r\t']
    print(f, 'env-imbalance:', {k: begins[k] - ends[k] for k in set(begins) | set(ends) if begins[k] != ends[k]},
          'spacing:', len(re.findall(r'\\[;,!]', s)), 'emdash:', s.count('\u2014'), 'ctrl:', len(bad),
          'braces:', s.count('{') - s.count('}'), 'dollars odd:', s.count('$') % 2)
    for m in re.finditer(r'.{25}\\[;,!].{10}', s):
        print('   spacing at:', repr(m.group()))
labels = Counter()
for f in glob.glob(f"{root}/**/*.tex", recursive=True):
    txt = open(f, encoding='utf8').read()
    for l in re.findall(r'\\label\{([^}]+)\}', txt) + re.findall(r'\\[a-z]+heading\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\{([^}]+)\}', txt):
        labels[l] += 1
dups = {k: v for k, v in labels.items() if v > 1}
print('duplicate labels (all files):', len(dups))
mine = ['def:observable-algebra', 'prop:agent-fe-invariance', 'prop:predictive-equivalence', 'eq:model-display',
        'eq:composite-record-kernel', 'eq:shared-model-frame', 'def:adaptive-history', 'eq:record-flow-kernel', 'geo:class-curvature']
for m in mine:
    print(m, labels[m])
refs = set()
for f in glob.glob(f"{root}/**/*.tex", recursive=True):
    for grp in re.findall(r'\\[cC]ref\{([^}]+)\}', open(f, encoding='utf8').read()):
        refs.update(x.strip() for x in grp.split(','))
missing = sorted(r for r in refs if r not in labels)
print('refs without label:', missing)
