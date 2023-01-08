import numpy as np
import faiss
from FM_sparse import FM_sparse

def Chain2data(Chain, XnYnZn, Kernel_Grv, Kernel_Mag):
    
    k0 = int(Chain[0])
    xc = Chain[1:1+k0]
    yc = Chain[1+k0:1+2*k0]
    zc = Chain[1+2*k0:1+3*k0]
    rhoc = Chain[1+3*k0:1+4*k0]

    xc = xc.astype('float32')
    yc = yc.astype('float32')
    zc = zc.astype('float32')
    rhoc = rhoc.astype('float32')


    TrainPoints = np.column_stack((xc,yc,zc)).copy()
    index = faiss.IndexFlatL2(3)
    index.add(TrainPoints)
    D, I = index.search(XnYnZn, 1)     # actual search    
    DensityModel = rhoc[I[:,0]].copy()
    DensityModel = DensityModel.astype('float32')
    dg = FM_sparse(DensityModel, Kernel_Grv)

    SusModel = DensityModel / 50.0
    SusModel[DensityModel<0.2] = 0
    dT = FM_sparse(SusModel, Kernel_Mag)
    
    return dg, dT, DensityModel