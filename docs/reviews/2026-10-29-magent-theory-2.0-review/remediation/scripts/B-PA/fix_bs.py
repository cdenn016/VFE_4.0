"""Text-edit helper: repair a backspace character introduced by a shell escape."""
import sys
p = sys.argv[1]
s = open(p, 'rb').read()
i = s.find(b'\x08')
print('found at', i, s[i-5:i+10])
s = s.replace(b'$\x08ar u$', b'$\\bar u$')
open(p, 'wb').write(s)
print('remaining', s.count(b'\x08'))
