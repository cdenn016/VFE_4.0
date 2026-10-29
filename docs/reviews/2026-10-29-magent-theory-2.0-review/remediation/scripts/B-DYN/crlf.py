import sys

for f in sys.argv[1:]:
    s = open(f, encoding='utf-8', newline='').read()
    s = s.replace('\r\n', '\n').replace('\n', '\r\n')
    open(f, 'w', encoding='utf-8', newline='').write(s)
    print(f, s.count('\r\n'))
