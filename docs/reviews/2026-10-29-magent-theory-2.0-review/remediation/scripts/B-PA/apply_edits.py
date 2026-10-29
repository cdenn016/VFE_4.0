"""Text-edit helper. Applies exact old->new replacements to a CRLF file.
Edit file format: blocks of
@@@OLD
<old text>
@@@NEW
<new text>
@@@END
Each old text must occur exactly once. All or nothing.
Usage: python apply_edits.py TARGET EDITFILE"""
import sys
target, editf = sys.argv[1:3]
s = open(target, newline='').read()
raw = open(editf, encoding='utf-8').read().replace('\r\n', '\n')
blocks = raw.split('@@@OLD\n')[1:]
for k, b in enumerate(blocks):
    old, rest = b.split('\n@@@NEW\n', 1)
    new = rest.split('\n@@@END', 1)[0]
    old = old.replace('\n', '\r\n')
    new = new.replace('\n', '\r\n')
    n = s.count(old)
    if n != 1:
        sys.exit(f'edit {k}: found {n} occurrences of: {old[:80]!r}')
    s = s.replace(old, new)
open(target, 'w', newline='').write(s)
print('applied', len(blocks), 'edits to', target)
