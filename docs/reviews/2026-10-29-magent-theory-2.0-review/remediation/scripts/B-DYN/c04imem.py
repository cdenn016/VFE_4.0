import re

p = 'companion/c04_interacting_memory.tex'
s = open(p, encoding='utf-8', newline='').read()


def sub_block(s, start, end, pat, rep):
    a = s.index(start)
    b = s.index(end, a) if end else len(s)
    blk = s[a:b]
    n = len(re.findall(pat, blk))
    blk = re.sub(pat, lambda m: rep, blk)
    print(start[:30], n)
    return s[:a] + blk + s[b:]


cap = r'(?<![A-Za-z\\])C(?![A-Za-z])'
s = sub_block(s, 'There is a corresponding exact free-energy reduction', '\\section{An exchange that changes the reply}',
              r'(?<![A-Za-z\\])Z_t', '\\mathcal Z_t')
s = sub_block(s, '\\section{When Gaussian communication closes}', '\\section{Why complete local beliefs', cap, 'N')
s = sub_block(s, '\\section{Why complete local beliefs', 'Finally, the protocol redistributes', cap, '\\chi')
s = s.replace('$308/959$', '$44/137$').replace('\\frac{651}{959}\\frac16+\\frac{308}{959}\\frac5{14}', '\\frac{93}{137}\\frac16+\\frac{44}{137}\\frac5{14}')
open(p, 'w', encoding='utf-8', newline='').write(s)
print('tabs', s.count('\t'), 'ff', s.count('\x0c'), '959' in s)
