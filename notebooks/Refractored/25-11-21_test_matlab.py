
#%%
import matlab.engine
import numpy as np


eng = matlab.engine.start_matlab()

#%%
filename="/home/au643300/DataHandling/data/raw/field.0493.u"
#vel,xF,yF,zF,Lx,Ly,Lz,t,Re,flowtype,dstar,pou,rlam,spanv,kxvec,kzvec,vscal
a = eng.readdns(filename,4)

#%%
a[0].size