import re
import sys

files = ['chapters/03_dynamics.tex', 'supplements/s03_variation.tex', 'chapters/04_prediction.tex',
         'appendices/conditional_memory.tex', 'appendices/interacting_memory.tex', 'companion/c03_dynamics.tex',
         'companion/c04_communication.tex', 'companion/c04_interacting_memory.tex']
for f in files:
    s = open(f, encoding='utf-8', newline='').read()
    b = re.findall(r'\\begin\{([a-z*]+)\}', s)
    e = re.findall(r'\\end\{([a-z*]+)\}', s)
    ok = sorted(b) == sorted(e)
    # dollar parity per paragraph (ignore \$)
    bad = []
    for i, para in enumerate(re.split(r'\n\s*\n', s)):
        n = len(re.findall(r'(?<!\\)\$', para))
        if n % 2:
            bad.append(i)
    braces = s.count('{') - s.count('}') - s.count('\\{') + s.count('\\}')
    crlf = s.count('\r\n'); lf = s.count('\n')
    print(f, 'env ok' if ok else 'ENV MISMATCH', 'odd$paras', bad[:5], 'brace', braces, 'crlf', crlf, 'lf', lf)
