r"""R-ERA script 3: environment, brace, and dollar balance of the edited TeX files."""
import re
import sys
from collections import Counter

BS = chr(92)
for p in sys.argv[1:]:
    s = open(p, encoding='utf-8').read()
    b = Counter(re.findall(BS * 2 + r'begin\{(\w+\*?)\}', s))
    e = Counter(re.findall(BS * 2 + r'end\{(\w+\*?)\}', s))
    t = s.replace(BS + '{', '').replace(BS + '}', '')
    bad = [x for x in (BS + ';', BS + ',', BS + '!', '---') if x in s]
    print(p.split('/')[-1], 'envs balanced', b == e, dict(b), 'brace', t.count('{') - t.count('}'),
          'dollars even', s.count('$') % 2 == 0, 'banned spacing', bad)
