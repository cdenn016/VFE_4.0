import sys
p, start_marker, end_marker, newfile = sys.argv[1:5]
s = open(p, encoding='utf-8').read()
a = s.index(start_marker)
b = s.index(end_marker, a)
new = open(newfile, encoding='utf-8').read()
s = s[:a] + new + s[b:]
open(p, 'w', encoding='utf-8', newline='').write(s)
print('ok', a, b)
