import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegressionCV

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import metrics
import faiss
from Model_Generator import Model_Generator
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import matplotlib        as mpl
from matplotlib import rc
from scipy.interpolate import griddata

import math
from Gravity_3D_Kernel_Expanded_MD import Gravity_3D_Kernel_Expanded_MD
from Magnetic_3D_Kernel_Expanded_MD import Magnetic_3D_Kernel_Expanded_MD
from Kernel_compressor import Kernel_compressor
from scipy.sparse import csr_matrix
from FM_sparse import FM_sparse
import scipy.sparse

from mpi4py import MPI
from jointdata2features import jointdata2features
from functools import reduce
import sys
from imblearn.under_sampling import RandomUnderSampler 
from scipy.interpolate import CubicSpline
from datetime import datetime
import pickle
import logging
import warnings
warnings.filterwarnings('ignore')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
status = MPI.Status()
Nthreads = comm.Get_size() # No. MCMC chains or MPI threads

##### object of random under sampling for balancing data
# rus = RandomUnderSampler()


def NaN_remover(features):
    nan_tf = ~np.isnan(features).any(axis=1)
    return nan_tf


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def data_balancing(labels, CX):
    
    tf_b = np.full(labels.shape, False)    
    
    unq, cnt = np.unique(labels, return_counts=True)
    cnt = (cnt / CX) + 1
    cnt = cnt.astype('int')
  
    for i, ilabel in enumerate(unq):
                
        tf = labels == ilabel
                    
        true_idx = np.argwhere(tf)

        random_idx = np.random.default_rng().choice(len(true_idx), size=cnt[i], replace=False)

        random_index = true_idx[random_idx]

        tf_b[random_index] = True
            
            
    return tf_b

### combining two rf models into one
def combine_rfs(rf_a, rf_b):
    rf_a.estimators_ += rf_b.estimators_
    rf_a.n_estimators = len(rf_a.estimators_)
    return rf_a


def list_flatten(ListofList):
    return [item for sublist in ListofList for item in sublist]


def data_misfit(dg_1, dg_2, sigma):
    misfit = np.sum(((dg_1 - dg_2) / sigma)**2)
    return misfit


def IsInBox(dg_1, dg_2):
    dg_max = dg_1.max()
    dg_min = dg_1.min()
    tf = (dg_2 <= dg_max) & (dg_2 >= dg_min)
    tf_sum = np.sum(tf)    
    return tf_sum/dg_2.size

##################################################################################

inbox_threshold = 0.95


fpath = ""

fpath_loaddesk = os.getcwd()+'/loaddesk'

fpath_training_dataset = os.getcwd()+'/training_dataset'

inbox_threshold_folder = os.path.join(fpath_training_dataset, str(int(inbox_threshold*100)))


if rank == 0:
    
    if not os.path.exists(fpath_loaddesk):
        os.mkdir(fpath_loaddesk)
        
    if not os.path.exists(fpath_training_dataset):
        os.mkdir(fpath_training_dataset)
     
    if not os.path.exists(inbox_threshold_folder):
        os.mkdir(inbox_threshold_folder)

comm.Barrier()


log_filename = os.path.join(fpath_training_dataset, 'output.log')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(filename)s | %(message)s',
                    handlers = [logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)])
    
    

Ndatapoints = 32         # Number of total data in 1D array shape
CX = 64                 # must be dividable by downsample rate of x 
CY = 64                 # must be dividable by downsample rate of x 
CZ = 64                 # must be dividable by downsample rate of z

XnYnZn = np.zeros(CX*CY*CZ)
dg_obs = np.zeros(Ndatapoints*Ndatapoints,dtype=float)
dT_obs = np.zeros(Ndatapoints*Ndatapoints,dtype=float)
Kernel_Grv = np.zeros((Ndatapoints*Ndatapoints,CX*CY*CZ)).astype('float32')
Kernel_Mag = np.zeros((Ndatapoints*Ndatapoints,CX*CY*CZ)).astype('float32')
Gkernelsp = csr_matrix(Kernel_Grv)
Mkernelsp = csr_matrix(Kernel_Mag)

Gravity_Data = np.loadtxt('GRV_2D_Data.txt').astype('float32')
Magnetic_Data = np.loadtxt('RTP_2D_Data.txt').astype('float32')

xs = np.linspace(np.amin(Gravity_Data[:,0]),np.amax(Gravity_Data[:,0]),Ndatapoints)
ys = np.flip(np.linspace(np.amin(Gravity_Data[:,1]),np.amax(Gravity_Data[:,1]), Ndatapoints),0)
# ys = np.linspace(Gravity_Data[0,1],Gravity_Data[-1,1], Ndatapoints)
XS,YS = np.meshgrid(xs,ys)
GRV_obs = griddata(Gravity_Data[:,0:2], Gravity_Data[:,2], (XS, YS), method='nearest')
RTP_obs = griddata(Magnetic_Data[:,0:2], Magnetic_Data[:,2], (XS, YS), method='nearest')

dg_obs = GRV_obs.flatten('C')
dT_obs = RTP_obs.flatten('C')


sigma_dg_obs = np.std(dg_obs)
sigma_dT_obs = np.std(dT_obs)


# model space
Z0 = 0                
ZEND = 10000          
Pad_Length = 5000

xmodel = np.linspace(np.amin(XS)-Pad_Length,np.amax(XS)+Pad_Length,CX)
ymodel = np.flip(np.linspace(np.amin(YS)-Pad_Length,np.amax(YS)+Pad_Length,CY))
zmodel = np.linspace(Z0,ZEND,CZ)
Y, Z, X = np.meshgrid(ymodel, zmodel, xmodel)
dx = abs(X[0,0,1]-X[0,0,0])
dy = abs(Y[0,1,0]-Y[0,0,0])
dz = abs(Z[1,0,0]-Z[0,0,0])

XYZ = np.column_stack((X.flatten(),Y.flatten(),Z.flatten())).astype('float32')

x_min=np.min(X)-dx/2
x_max=np.max(X)+dx/2
y_min=np.min(Y)-dy/2
y_max=np.max(Y)+dy/2
z_min=np.min(Z)-dz/2
z_max=np.max(Z)+dz/2

Xn_3D = np.divide(X-x_min,x_max-x_min)
Yn_3D = np.divide(Y-y_min,y_max-y_min)
Zn_3D = np.divide(Z-z_min,z_max-z_min)

Xn = Xn_3D.flatten()
Yn = Yn_3D.flatten()
Zn = Zn_3D.flatten()
XnYnZn = np.column_stack((Xn,Yn,Zn)).astype('float32')

Inc = 90 * (3.14159265359/180)
Dec = 0 * (3.14159265359/180)
Fe = 43314  #(nT)

if rank == 0:
    
    Gkernelsp_str = fpath_loaddesk +'//'+'Gkernelsp.npz'
    Mkernelsp_str = fpath_loaddesk +'//'+'Mkernelsp.npz'

    if os.path.exists(Gkernelsp_str) and os.path.exists(Mkernelsp_str):
        
        logging.info("sparse kernels exist")
        
        Gkernelsp = scipy.sparse.load_npz(Gkernelsp_str)
        Mkernelsp = scipy.sparse.load_npz(Mkernelsp_str)

    else:
        
        logging.info("sparse kernels do not exist and will be computed")
        
        Kernel_Grv_str = fpath_loaddesk +'//'+'Kernel_Grv'+'.npy'
        Kernel_Mag_str = fpath_loaddesk +'//'+'Kernel_Mag'+'.npy'
        
        if os.path.exists(Kernel_Grv_str) and os.path.exists(Kernel_Mag_str):
            
            logging.info("Actual kernels exist and sparse kernels will be computed")
            
            Kernel_Grv = np.load(Kernel_Grv_str) # use this, if the latest result exists 
            Kernel_Mag = np.load(Kernel_Mag_str) # use this, if the latest result exists
    
            
        else:
            
            logging.info("Actual kernels do not exist and will be computed")

            for iz in np.arange(CZ):
                c1 = iz*CX*CY
                c2 = (iz+1)*CX*CY
                Kernel_Grv[:,c1:c2] = Gravity_3D_Kernel_Expanded_MD(X[iz,:,:],Y[iz,:,:],Z[iz,:,:],XS,YS,dx,dy,dz)
                Kernel_Mag[:,c1:c2] = Magnetic_3D_Kernel_Expanded_MD(X[iz,:,:],Y[iz,:,:],Z[iz,:,:],XS,YS,dx,dy,dz,Inc,Dec,Fe)

            Kernel_Grv = Kernel_Grv*1e8
            Kernel_Grv = Kernel_Grv.astype('float32')
            Kernel_Mag = Kernel_Mag.astype('float32')

            np.save(Kernel_Grv_str, Kernel_Grv)
            np.save(Kernel_Mag_str, Kernel_Mag)


        [Gkernelsp, Mkernelsp] = Kernel_compressor(Ndatapoints,CX,CY,CZ,Kernel_Grv,Kernel_Mag,fpath_loaddesk)

        scipy.sparse.save_npz(Gkernelsp_str, Gkernelsp)
        scipy.sparse.save_npz(Mkernelsp_str, Mkernelsp)



Kernel_Grv = comm.bcast(Gkernelsp, root=0)
Kernel_Mag = comm.bcast(Mkernelsp, root=0)
comm.Barrier()

### list of rho can have single or multiple values such as rho_salt = [-0.4, -0.2]

rho_sed = [0.0]
rho_salt = [-0.13]
rho_base = [0.4]

XSn = np.divide(XS-x_min,x_max-x_min)
YSn = np.divide(YS-y_min,y_max-y_min)

XSn = XSn.flatten().astype('float32')
YSn = YSn.flatten().astype('float32')


NKEEP = 50          # dump a binary file to desk every NKEEP records
Nmodels = 1000000
Kmin = 6
Kmax = 50

ChainKeep = np.zeros((NKEEP, 1 + Kmax*4)).astype('float32')
Chain = np.zeros(1 + Kmax*4).astype('float32')


if rank != 0:
    for imodel in range(Nmodels):

        Chain = Model_Generator(XnYnZn, rho_sed, rho_salt, rho_base,
                                              Kmin, Kmax, Kernel_Grv, dg_obs,
                                              inbox_threshold)

        ## Sending model to Master 
        comm.Send(Chain, dest=0, tag=rank)

        
ikeep = 0
isave = 0

if rank == 0:
    
    for i in range((Nthreads-1)*Nmodels):
                
        Chain[:] = 0.
        comm.Recv(Chain, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        Chain = Chain.copy()
        ChainKeep[ikeep, :] = Chain.copy()
        ikeep += 1
        
#         k0 = int(Chain[0])
#         xc = Chain[1:1+k0]
#         yc = Chain[1+k0:1+2*k0]
#         zc = Chain[1+2*k0:1+3*k0]
#         rhoc = Chain[1+3*k0:1+4*k0]
        
#         xc = xc.astype('float32')
#         yc = yc.astype('float32')
#         zc = zc.astype('float32')
#         rhoc = rhoc.astype('float32')


#         TrainPoints = np.column_stack((xc,yc,zc)).copy()
#         index = faiss.IndexFlatL2(3)
#         index.add(TrainPoints)
#         D, I = index.search(XnYnZn, 1)     # actual search    
#         DensityModel = rhoc[I[:,0]].copy()
#         DensityModel = DensityModel.astype('float32')

#         dg = FM_sparse(DensityModel, Kernel_Grv)

#         isinbox_c = IsInBox(dg_obs, dg)

    



        if ikeep == NKEEP:

            daytime = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")

            npy_name = 'training_' +  daytime + '.npy'

            npy_path = os.path.join(inbox_threshold_folder, npy_name)

            np.save(npy_path, ChainKeep)
            ikeep = 0
            ChainKeep[:, :] = 0.
            
            isave += 1
            
            logging.info("Number of saved files {}".format(isave))




logging.info('rank', rank, 'The End')

MPI.Finalize
        
        
        