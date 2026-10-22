import re,sys,glob,os
bib=sys.argv[1]
existing=open(bib,encoding='utf-8').read()
have=set(re.findall(r'@\w+\{([^,\s]+),',existing))
entries={}
for log in sorted(glob.glob('logs/*.md')):
    t=open(log,encoding='utf-8').read()
    i=t.find('BIB ENTRIES')
    if i<0: continue
    sec=t[i:]
    j=re.search(r'\n#+\s*NEW LABELS|\nNEW LABELS',sec)
    if j: sec=sec[:j.start()]
    # extract balanced @type{...}
    pos=0
    while True:
        m=re.search(r'@(\w+)\s*\{\s*([^,\s]+)\s*,',sec[pos:])
        if not m: break
        st=pos+m.start(); depth=0; k=pos+m.end()-1
        k=sec.index('{',st)
        for e in range(k,len(sec)):
            if sec[e]=='{': depth+=1
            elif sec[e]=='}':
                depth-=1
                if depth==0: break
        block=sec[st:e+1]; key=m.group(2)
        entries.setdefault(key,(block,os.path.basename(log)))
        pos=e+1
new={k:v for k,v in entries.items() if k not in have}
print('found',len(entries),'new',len(new),'already present',sorted(set(entries)&have))
if '--write' in sys.argv:
    with open(bib,'a',encoding='utf-8') as f:
        for k,(b,src) in sorted(new.items()):
            f.write('\n'+b.strip()+'\n')
for k,(b,src) in sorted(new.items()): print(k,src)
