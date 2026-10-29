p = 'appendices/interacting_memory.tex'
s = open(p, encoding='utf-8', newline='').read()
a = s.index('\theoremheading{Repeated affine Gaussian closure}')
b = s.index('\section{Failure of finite-mixture closure}')
blk = s[a:b]
blk = blk.replace('S_t^{-1}', '(\Sigma^D_t)^{-1}').replace('S_{t+1}^{-1}', '(\Sigma^D_{t+1})^{-1}').replace('S_{t+1}\left', '\Sigma^D_{t+1}\left').replace('\mathcal N(m_t,S_t)$ with $S_t\succ0', '\mathcal N(m_t,\Sigma^D_t)$ with $\Sigma^D_t\succ0')
open(p, 'w', encoding='utf-8', newline='').write(s[:a] + blk + s[b:])
print(blk.count('S_t'), blk.count('Sigma^D'))
