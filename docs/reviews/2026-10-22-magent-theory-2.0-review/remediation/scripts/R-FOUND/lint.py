import re, glob

def labels(files):
    L = {}
    for f in files:
        s = open(f, encoding='utf8').read()
        for m in re.findall(r'\\label\{([^}]*)\}', s):
            L.setdefault(m, []).append(f)
        for m in re.findall(r'heading\{[^}]*\}\{([^}]*)\}', s):
            L.setdefault(m, []).append(f)
    return L

main = ['frontmatter.tex'] + sorted(glob.glob('chapters/*.tex')) + sorted(glob.glob('appendices/*.tex'))
sup = sorted(glob.glob('supplements/*.tex'))
comp = sorted(glob.glob('companion/*.tex'))
allL = labels(main + sup + comp)
print('dups', {k: v for k, v in allL.items() if len(v) > 1})
for group, files in (('main', main), ('sup', sup), ('comp', comp)):
    L = labels(files)
    for f in files:
        s = open(f, encoding='utf8').read()
        for m in re.findall(r'\\[cC]ref\{([^}]*)\}|\\eqref\{([^}]*)\}', s):
            for k in (m[0] or m[1]).split(','):
                if k.strip() not in L:
                    print('MISSING', group, f, k)
mine = ['chapters/01_foundations.tex', 'chapters/03_dynamics.tex', 'chapters/04_prediction.tex',
        'appendices/record_process.tex', 'appendices/notation.tex', 'supplements/s01_probability.tex',
        'supplements/s03_variation.tex', 'companion/c01_objects.tex', 'companion/c03_dynamics.tex',
        'companion/c04_communication.tex', 'companion/glossary.tex']
cites = set()
for f in mine:
    s = open(f, encoding='utf8').read()
    b = re.findall(r'\\begin\{(\w+)\}', s); e = re.findall(r'\\end\{(\w+)\}', s)
    if sorted(b) != sorted(e):
        print('UNBALANCED', f)
    if s.count('{') != s.count('}'):
        print('BRACES', f, s.count('{'), s.count('}'))
    for w in ['crucially', 'notably', 'importantly', 'essentially', 'leverage', 'robust', 'comprehensive',
              'key insight', 'delve', 'fundamentally', '\\;', '\\,', '\\!', 'landscape', 'nuanced', 'pivotal']:
        if w in s:
            print('STYLE', f, w)
    for m in re.findall(r'\\cite[pt]?(?:\[[^\]]*\])?\{([^}]*)\}', s):
        cites.update(k.strip() for k in m.split(','))
bib = open('references.bib', encoding='utf8').read()
print('NEW KEYS', sorted(k for k in cites if '{' + k + ',' not in bib))
