"""Check cref targets, label uniqueness, and environment balance for B-FUS files."""
import re, glob, collections
ROOT = 'C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/'
files = [f for f in glob.glob(ROOT + '**/*.tex', recursive=True) if '/build/' not in f.replace('\\', '/')]
labels = collections.Counter()
heading = re.compile(r'\\[a-z]+heading\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\{([^{}]+)\}')
for f in files:
    t = open(f, encoding='utf-8', errors='replace').read()
    for m in re.finditer(r'\\label\{([^}]+)\}', t):
        labels[m.group(1)] += 1
    for m in heading.finditer(t):
        labels[m.group(1)] += 1
mine = ['chapters/06_fusion.tex', 'chapters/07_smooth_closure.tex', 'chapters/08_smooth_erasure.tex',
        'chapters/09_public_compression.tex', 'companion/c07_fusion.tex', 'companion/c08_smooth.tex',
        'companion/c09_public_compression.tex']
for rel in mine:
    t = open(ROOT + rel, encoding='utf-8').read()
    for m in re.finditer(r'\\[cC]ref\{([^}]+)\}', t):
        for k in m.group(1).split(','):
            if labels[k.strip()] == 0:
                print('UNDEFINED', rel, k)
    for m in re.finditer(r'\\label\{([^}]+)\}', t):
        if labels[m.group(1)] > 1:
            print('DUP', rel, m.group(1), labels[m.group(1)])
    for env in ['equation', 'proof', 'enumerate', 'align', 'center', 'tabular']:
        b = t.count('\\begin{%s}' % env); e = t.count('\\end{%s}' % env)
        if b != e:
            print('UNBALANCED', rel, env, b, e)
    for bad in ['\\,', '\\;', '\\!', '\\eqref', '---']:
        if bad in t:
            print('STYLE', rel, bad)
print('done')
