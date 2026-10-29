import re

p = 'chapters/04_prediction.tex'
s = open(p, encoding='utf-8', newline='').read()
a = s.index('\\propositionheading{Common recovery by likelihood descent}')
b = s.index('\\section{A predictive coding criterion}')
blk = s[a:b]
pat = re.compile(r'(?<![A-Za-z\\])C(?![A-Za-z])')
print('matches', len(pat.findall(blk)))
blk = pat.sub(lambda m: '\\tau', blk)
s = s[:a] + blk + s[b:]
open(p, 'w', encoding='utf-8', newline='').write(s)
print('tabs', s.count('\t'))
