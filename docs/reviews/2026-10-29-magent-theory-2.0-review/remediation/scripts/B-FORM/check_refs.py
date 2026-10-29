import re, glob, os
M = r'C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/'
files = glob.glob(M + '**/*.tex', recursive=True)
labels = set()
for f in files:
    s = open(f, encoding='utf-8', errors='replace').read()
    labels |= set(re.findall(r'\\label\{([^}]*)\}', s))
    labels |= set(re.findall(r'heading\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\{([^}]*)\}', s))
bib = open(M + 'references.bib', encoding='utf-8', errors='replace').read()
keys = set(re.findall(r'@\w+\{([^,]+),', bib))
mine = ['chapters/10_boundary_closure.tex', 'chapters/11_conditional_formation.tex', 'chapters/12_open_formation.tex',
        'companion/c10_boundary.tex', 'companion/c11_formation.tex', 'companion/c12_open_formation.tex']
for f in mine:
    s = open(M + f, encoding='utf-8').read()
    for grp in re.findall(r'\\[cC]ref\{([^}]*)\}', s):
        for l in grp.split(','):
            if l.strip() not in labels:
                print(f, 'MISSING LABEL', l)
    for grp in re.findall(r'\\cite[pt]?(?:\[[^\]]*\])?\{([^}]*)\}', s):
        for k in grp.split(','):
            if k.strip() not in keys:
                print(f, 'MISSING KEY', k)
    print(f, 'begin/end', s.count('\\begin{'), s.count('\\end{'), 'braces', s.count('{') - s.count('}'))
