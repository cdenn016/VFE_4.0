import re

p = 'appendices/interacting_memory.tex'
s = open(p, encoding='utf-8', newline='').read()
a = s.index('\\propositionheading{A reachable collision of local-only retention}')
b = s.index('For comparison, if a local label/result posterior')
blk = s[a:b]
pat = re.compile(r'(?<![A-Za-z\\])C(?![A-Za-z])')
print('matches', len(pat.findall(blk)))
blk = pat.sub(lambda m: '\\chi', blk)
s = s[:a] + blk + s[b:]
open(p, 'w', encoding='utf-8', newline='').write(s)
print('tabs', s.count('\t'), 'ff', s.count('\x0c'))
