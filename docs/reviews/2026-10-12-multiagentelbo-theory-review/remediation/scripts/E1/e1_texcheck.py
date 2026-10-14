"""E1: check cref targets, brace balance, and label uniqueness in the E1 files."""
import glob
import re
import sys

root = sys.argv[1]
HEAD = re.compile(r'heading\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}\{([^}]+)\}')
labels: dict[str, int] = {}
for f in glob.glob(root + '/*.tex'):
    txt = open(f, encoding='utf-8').read()
    for l in re.findall(r'\\label\{([^}]+)\}', txt) + HEAD.findall(txt):
        labels[l] = labels.get(l, 0) + 1
print('duplicate labels:', [l for l, c in labels.items() if c > 1 and not l.startswith('#')])
for name in ['c02_geometry.tex', 'c03_probability.tex', 'c06_attention.tex', 'ca_proofs_c06.tex', 'g03_interaction_family.tex']:
    t = open(root + '/' + name, encoding='utf-8').read()
    refs = set()
    for a, b in re.findall(r'\\(?:c|C)ref\{([^}]+)\}|\\eqref\{([^}]+)\}', t):
        for g in (a, b):
            for r in g.split(','):
                if r.strip():
                    refs.add(r.strip())
    miss = sorted(r for r in refs if r not in labels)
    u = re.sub(r'\\[{}]', '', t)
    depth, neg = 0, False
    for ch in u:
        depth += (ch == '{') - (ch == '}')
        neg |= depth < 0
    bad = [w for w in ['crucially', 'notably', 'importantly', 'fundamentally', 'leverage', 'underscore',
                       'key insight', 'robust', 'delve', '---', '\\;', '\\,', '\\!'] if w in t]
    print(name, 'missing refs:', miss, 'brace depth:', depth, 'negative:', neg, 'banned:', bad)
