import re,glob,sys
pat=re.compile(r'10_boundary|11_conditional|12_open|c10_boundary|c11_formation|c12_open')
for f in sorted(glob.glob('../../../lenses/*.md')):
    txt=open(f,encoding='utf-8').read()
    blocks=re.split(r'\n(?=### )',txt)
    for b in blocks:
        if not b.startswith('### '): continue
        head=b.split('\n',1)[0]
        loc=re.search(r'\*\*Location:\*\*(.*)',b)
        sev=re.search(r'\*\*Severity:\*\*(.*)',b)
        cat=re.search(r'\*\*Category:\*\*(.*)',b)
        if loc and pat.search(loc.group(1)):
            print(head[4:110],'|',(cat.group(1).strip() if cat else ''),'|',(sev.group(1).strip() if sev else ''))
