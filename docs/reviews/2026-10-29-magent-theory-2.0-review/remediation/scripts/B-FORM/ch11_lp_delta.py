import numpy as np
from ch11_lp_null import run
I=np.eye(2); u=(1.2807,1.5057,1.8665,2.3057)
for de in [0.09,0.095,0.10,0.105,0.11]:
    hf,mf=run(u,I,0.0,1.0,1.0,de,'full'); hl,ml=run(u,I,0.0,1.0,1.0,de,'lp')
    print('delta',de,[b for _,b in hf],'full margin %.4f'%min(mf),[b for _,b in hl],'lp margin %.4f'%min(ml))
for kap in [0.05,0.1,0.5,1.0]:
    hl,ml=run(u,I,kap,1.0,1.0,0.1,'lp'); print('kappa',kap,hl)
for T in [0.3,3.0,10.0]:
    hl,ml=run(u,I,0.0,1.0,T,0.1,'lp'); print('T',T,hl)
