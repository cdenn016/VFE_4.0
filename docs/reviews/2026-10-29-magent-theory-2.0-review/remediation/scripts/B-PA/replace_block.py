"""Text-edit helper: replace the block of a CRLF file between two markers with
the contents of a text file (written with LF), keeping CRLF line endings.
Usage: python replace_block.py FILE START_MARKER END_MARKER NEWTEXT_FILE"""
import sys
p, start, end, newf = sys.argv[1:5]
s = open(p, newline='').read()
a = s.index(start)
b = s.index(end, a)
new = open(newf, encoding='utf-8').read().replace('\r\n', '\n').replace('\n', '\r\n')
s = s[:a] + new + s[b:]
open(p, 'w', newline='').write(s)
print('replaced', b - a, 'chars with', len(new))
