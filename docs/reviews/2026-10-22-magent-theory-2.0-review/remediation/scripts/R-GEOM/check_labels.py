"""Check label definitions (\\label and heading macros) and references in the owned files."""
import glob
import re

ROOT = "C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/"
p = ROOT + 'chapters/02_interactions.tex'
s = open(p, encoding='utf-8').read()
a = "\\bigr\\Vert_F^2\n\\label{eq:int-loop-defect}"
if a in s:
    s = s.replace(a, "\\bigr\\Vert_F^2,\n\\label{eq:int-loop-defect}")
    open(p, 'w', encoding='utf-8').write(s)
    print('punctuated loop-defect display')

defs = {}
heading = re.compile(r'\\(?:definition|construction|lemma|proposition|theorem|corollary|remark|openproblem|hypothesis)heading\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\{([^{}]+)\}')
for f in glob.glob(ROOT + '**/*.tex', recursive=True):
    t = open(f, encoding='utf-8').read()
    for lab in re.findall(r'\\label\{([^}]+)\}', t) + heading.findall(t) + re.findall(r'\\chapter\{[^}]*\}\s*\\label\{([^}]+)\}', t):
        defs.setdefault(lab, set()).add(f.replace(ROOT, ''))
dups = {k: v for k, v in defs.items() if len(v) > 1}
print('labels defined in more than one file:', dups)
OWN = ['chapters/02_interactions.tex', 'chapters/02_geometry.tex', 'appendices/base_connection.tex',
       'appendices/determinant_example.tex', 'supplements/s02_geometry.tex', 'companion/c02_links.tex',
       'companion/c06_geometry.tex']
for f in OWN:
    t = open(ROOT + f, encoding='utf-8').read()
    for grp in re.findall(r'\\[cC]ref\{([^}]+)\}', t):
        for r in grp.split(','):
            if r.strip() not in defs:
                print('MISSING', f, r)
    for lab in re.findall(r'\\label\{([^}]+)\}', t) + heading.findall(t):
        n = sum(len(re.findall(r'(?:\\label\{|heading\{[^{}]*\}\{)' + re.escape(lab) + r'\}', open(g, encoding='utf-8').read())) for g in glob.glob(ROOT + '**/*.tex', recursive=True))
        if n != 1:
            print('label count', lab, n)
bib = open(ROOT + 'references.bib', encoding='utf-8').read()
keys = set(re.findall(r'@\w+\{([^,]+),', bib))
for f in OWN:
    t = open(ROOT + f, encoding='utf-8').read()
    for grp in re.findall(r'\\cite[pt]?(?:\[[^\]]*\])?\{([^}]+)\}', t):
        for k in grp.split(','):
            if k.strip() not in keys:
                print('NEW BIB KEY', f, k.strip())
