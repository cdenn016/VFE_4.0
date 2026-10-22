r"""R-ERA script 4: punctuate display equations as parts of sentences.

For each equation environment, the last math line (skipping label lines and the
closing of inner aligned/split blocks) gets a period if the following prose starts
with an uppercase letter or a sectioning/heading command, and a comma otherwise.
Lines that already end in punctuation are left alone. Prints every change.
"""
import sys

BS = chr(92)


def process(path: str) -> None:
    lines = open(path, encoding='utf-8').read().split('\n')
    i = 0
    changes = []
    while i < len(lines):
        if lines[i].strip().startswith(BS + 'end{equation}'):
            # find following prose
            k = i + 1
            while k < len(lines) and lines[k].strip() == '':
                k += 1
            nxt = lines[k].lstrip() if k < len(lines) else ''
            stop = (nxt[:1].isupper() or nxt.startswith(BS) and not nxt.startswith(BS + 'cref')
                    or nxt == '')
            if nxt.startswith(BS + 'Cref'):
                stop = True
            mark = '.' if stop else ','
            # find last math line
            j = i - 1
            while j > 0 and (lines[j].strip().startswith(BS + 'label') or lines[j].strip().startswith(BS + 'end{')
                             or lines[j].strip() == ''):
                j -= 1
            s = lines[j].rstrip()
            if s and s[-1] not in '.,;:':
                if s.endswith(BS + BS):
                    pass
                else:
                    lines[j] = s + mark
                    changes.append((j + 1, mark, s[-40:], nxt[:30]))
        i += 1
    open(path, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
    for c in changes:
        print(path.split('/')[-1], c)


for p in sys.argv[1:]:
    process(p)
