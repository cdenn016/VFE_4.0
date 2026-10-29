"""Lint B-PA files: environment balance, banned spacing macros and dashes,
dangling or duplicate labels, cite keys used. Usage: python lint.py ROOT FILE..."""
import re, glob, sys, os
root = sys.argv[1]; mine = sys.argv[2:]
alltex = [f for f in glob.glob(root + '/**/*.tex', recursive=True)]
pat = re.compile(r'\\label\{([^}]+)\}|heading\{[^}]*\}\{([^}]+)\}')
labels = {}
for f in alltex:
    for m in pat.finditer(open(f, encoding='utf-8', errors='ignore').read()):
        l = m.group(1) or m.group(2)
        labels.setdefault(l, []).append(os.path.relpath(f, root))
cites = set()
for f in mine:
    s = open(os.path.join(root, f), encoding='utf-8').read()
    for env in ['equation', 'aligned', 'proof', 'gathered', 'enumerate', 'align', 'pmatrix', 'smallmatrix']:
        a = s.count('\\begin{%s}' % env); b = s.count('\\end{%s}' % env)
        if a != b: print(f, 'unbalanced', env, a, b)
    for bad in ['\\;', '\\,', '\\!', '\u2014', '---', '\x08']:
        if bad in s: print(f, 'contains', repr(bad))
    print(f, 'brace balance', s.count('{') - s.count('}'), 'dollar count parity', s.count('$') % 2)
    for m in re.finditer(r'\\[cC]ref\{([^}]+)\}', s):
        for l in m.group(1).split(','):
            if l.strip() not in labels: print(f, 'dangling', l)
    for m in pat.finditer(s):
        l = m.group(1) or m.group(2)
        if len(labels[l]) > 1: print(f, 'duplicate label', l, labels[l])
    for m in re.finditer(r'\\cite[pt]?(?:\[[^]]*\])?\{([^}]+)\}', s):
        for k in m.group(1).split(','): cites.add(k.strip())
bib = open(os.path.join(root, 'references.bib'), encoding='utf-8', errors='ignore').read()
for k in sorted(cites):
    print('cite', k, 'in references.bib' if ('{' + k + ',') in bib else 'NEW')
