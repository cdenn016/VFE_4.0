"""Repair control characters introduced by shell escaping, and lint the owned files."""
import re
import sys

ROOT = "C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/"
FILES = ['chapters/02_interactions.tex', 'chapters/02_geometry.tex', 'appendices/base_connection.tex',
         'appendices/determinant_example.tex', 'supplements/s02_geometry.tex', 'companion/c02_links.tex',
         'companion/c06_geometry.tex', 'companion/figures/link_transport.tex']
p = ROOT + 'companion/c06_geometry.tex'
s = open(p, encoding='utf-8').read()
s = s.replace('\x09', '\\t').replace('\x08', '\\b')
open(p, 'w', encoding='utf-8').write(s)
for f in FILES:
    t = open(ROOT + f, encoding='utf-8').read()
    bad = len(re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', t)) + t.count('\t')
    spacing = sum(t.count(x) for x in ['\\,', '\\;', '\\!'])
    envs = {e: (t.count('\\begin{' + e + '}'), t.count('\\end{' + e + '}')) for e in set(re.findall(r'\\begin\{(\w+\*?)\}', t))}
    unbalanced = {e: v for e, v in envs.items() if v[0] != v[1]}
    banned = [w for w in ['key insight', 'crucially', 'critically', 'notably', 'importantly', 'fundamentally', 'essentially',
                          'leverage', 'underscore', 'delve', 'comprehensive', 'seamless', 'pivotal', 'paramount', 'nuanced',
                          'landscape', 'realm', 'robust'] if w in t.lower()]
    print(f, 'ctrl', bad, 'spacing', spacing, 'unbalanced', unbalanced, 'banned', banned, 'braces', t.count('{') - t.count('}'))
