import re,glob,sys
ids=sys.argv[1:]
pat=re.compile(r'10_boundary|11_conditional|12_open|c10_boundary|c11_formation|c12_open|ch(apter)? ?1[012]\b')
for f in sorted(glob.glob('../../../lenses/*.md')):
    txt=open(f,encoding='utf-8').read()
    for b in re.split(r'\n(?=### )',txt):
        h=b.split('\n',1)[0]
        if any(h.startswith('### '+i+':') for i in ids):
            print(h)
            for fld in ['Location','Fix']:
                m=re.search(r'\*\*'+fld+r':\*\*(.*?)(?=\n\*\*[A-Z][a-z]+:\*\*|\Z)',b,re.S)
                if m: print(fld+':',m.group(1).strip()[:1800])
            print()
