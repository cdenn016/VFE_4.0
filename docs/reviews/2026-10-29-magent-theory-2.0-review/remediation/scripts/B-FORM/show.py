import re,glob,sys
ids=sys.argv[1:]
for f in sorted(glob.glob('../../../lenses/*.md')):
    txt=open(f,encoding='utf-8').read()
    for b in re.split(r'\n(?=### )',txt):
        h=b.split('\n',1)[0]
        for i in ids:
            if h.startswith('### '+i+':') or h.startswith('### '+i+' '):
                sys.stdout.write(b.strip()+'\n\n')
