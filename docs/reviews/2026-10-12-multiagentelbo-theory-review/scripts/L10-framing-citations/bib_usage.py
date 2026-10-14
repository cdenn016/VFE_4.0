"""List bib entries by author pattern and the main-manuscript files that cite them."""
import re, glob, sys, os
d = sys.argv[1]
pats = sys.argv[2].split(',')
bib = open(os.path.join(d, 'references.bib'), encoding='utf8').read()
entries = re.split(r'\n(?=@)', bib)
tex = {os.path.basename(f): open(f, encoding='utf8').read() for f in glob.glob(os.path.join(d, '*.tex'))}
cites = {}
for f, t in tex.items():
    for m in re.finditer(chr(92)*2+r"cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", t):
        for k in m.group(1).split(','):
            cites.setdefault(k.strip(), set()).add(f)
for e in entries:
    m = re.match(r'@\w+\{([^,]+),', e)
    if not m:
        continue
    key = m.group(1).strip()
    au = re.search(r'author\s*=\s*[{"](.*?)[}"]\s*,', e, re.S)
    ti = re.search(r'title\s*=\s*[{"](.*?)[}"]\s*,', e, re.S)
    a = au.group(1) if au else ''
    if any(p.lower() in a.lower() for p in pats):
        print(key, '|', (ti.group(1) if ti else '')[:80].replace('\n', ' '), '|', sorted(cites.get(key, [])))
