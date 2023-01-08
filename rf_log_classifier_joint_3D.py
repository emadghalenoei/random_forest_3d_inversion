import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegressionCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

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
from PLOT import plot_datagrid, plot_field_data_model_slice, plot_valid_data_model_slice
from Chain2data import Chain2data


from mpi4py import MPI
from jointdata2features import jointdata2features
from functools import reduce
import sys
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


# def Chain2data(Chain, XnYnZn, Kernel_Grv, Kernel_Mag):
    
#     k0 = int(Chain[0])
#     xc = Chain[1:1+k0]
#     yc = Chain[1+k0:1+2*k0]
#     zc = Chain[1+2*k0:1+3*k0]
#     rhoc = Chain[1+3*k0:1+4*k0]

#     xc = xc.astype('float32')
#     yc = yc.astype('float32')
#     zc = zc.astype('float32')
#     rhoc = rhoc.astype('float32')


#     TrainPoints = np.column_stack((xc,yc,zc)).copy()
#     index = faiss.IndexFlatL2(3)
#     index.add(TrainPoints)
#     D, I = index.search(XnYnZn, 1)     # actual search    
#     DensityModel = rhoc[I[:,0]].copy()
#     DensityModel = DensityModel.astype('float32')
#     dg = FM_sparse(DensityModel, Kernel_Grv)

#     SusModel = DensityModel / 50.0
#     SusModel[DensityModel<0.2] = 0
#     dT = FM_sparse(SusModel, Kernel_Mag)
    
#     return dg, dT, DensityModel

##################################################################################

inbox_threshold = 0.95

fpath = ""

fpath_loaddesk = os.getcwd()+'/loaddesk'

fpath_training_dataset = os.getcwd()+'/training_dataset'

inbox_threshold_folder = os.path.join(fpath_training_dataset, str(int(inbox_threshold*100)))



if rank == 0:
    daytime = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    fpath = os.getcwd()+'//' + daytime
    if os.path.exists(fpath) and os.path.isdir(fpath):
        shutil.rmtree(fpath)
    os.mkdir(fpath)
    
    fpath_Models = fpath+'/Models'
    fpath_Figs = fpath+'/Figs'
    fpath_Arrays = fpath+'/Arrays'

    
    os.mkdir(fpath_Models)
    os.mkdir(fpath_Figs)
    os.mkdir(fpath_Arrays)

    if not os.path.exists(fpath_loaddesk):
        os.mkdir(fpath_loaddesk)
        

    
    
fpath = comm.bcast(fpath, root=0)
comm.Barrier()


fpath_Models = fpath+'/Models'
fpath_Figs = fpath+'/Figs'
fpath_Arrays = fpath+'/Arrays'


    
log_filename = os.path.join(fpath, 'output.log')

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

if rank == 0:
    np.save(fpath_Arrays +'//'+'dg_obs.npy', dg_obs)
    np.save(fpath_Arrays +'//'+'dT_obs.npy', dT_obs)


sigma_dg_obs = np.std(dg_obs)
sigma_dT_obs = np.std(dT_obs)


# model space
Z0 = 0                
ZEND = 10000          
Pad_Length = 5000

xmodel = np.linspace(np.amin(XS)-Pad_Length,np.amax(XS)+Pad_Length,CX)
ymodel = np.flip(np.linspace(np.amin(YS)-Pad_Length,np.amax(YS)+Pad_Length,CY))
zmodel = np.linspace(Z0,ZEND,CZ)

if rank == 0:
    np.save(fpath_Arrays +'//'+'xmodel.npy', xmodel)
    np.save(fpath_Arrays +'//'+'ymodel.npy', ymodel)
    np.save(fpath_Arrays +'//'+'zmodel.npy', zmodel)

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

# label_sed = int(rho_sed[0] * 100)
# label_salt = int(rho_salt[0] * 100)
# label_base = int(rho_base[0] * 100)

XSn = np.divide(XS-x_min,x_max-x_min)
YSn = np.divide(YS-y_min,y_max-y_min)

XSn = XSn.flatten().astype('float32')
YSn = YSn.flatten().astype('float32')

k_list = [3, 9, 15, 21, 27, 33, 39, 45, 51] ### do not insert 1
if rank == 0:
    logging.info("k_list: {}".format(k_list))

### use knn to cover 1/5 of data coverage
### in 2d inversion, used 6 knn over 30 data
### in 3d inversion, use 6x6 = 36 knn over 32x32 data
### 3nn in 2d is equal to 9nn in 3dd

#3nn in 2d      . . . x o x . . .
# 9nn in 3d     . x x x .
#               . x o x .
#               . X X X .
    
D = np.zeros((XnYnZn.shape[0], max(k_list))).astype('float32')
I = np.zeros((XnYnZn.shape[0], max(k_list))).astype('int')

if rank == 0:
    index = faiss.IndexFlatL2(3)
    index.add(np.column_stack((XSn, YSn, 0.0*XSn)))
    
    D, I = index.search(XnYnZn, max(k_list))

D = comm.bcast(D, root=0)
I = comm.bcast(I, root=0)
comm.Barrier()
        
n_estimators = 100

labels_training_list = []
features_training_list = []

labels_testing_list = []
features_testing_list = []

npy_list = os.listdir(inbox_threshold_folder)
npyfile = os.path.join(inbox_threshold_folder, npy_list[0])
Chainkeep = np.load(npyfile)


Nmodels = 4000  ### half of them are for training rf and another half are for log reg
# Nmodels = min([2000, len(npy_list)*Chainkeep.shape[0]]) ### each worker sends Nmodels to master, default is 200

imodel = 0

rus = RandomUnderSampler(random_state=32)
ros = RandomOverSampler(random_state=33)

for npy in npy_list:
    
    if imodel >= Nmodels:
        break
    
    if npy.endswith('.npy'):
                
        npyfile = os.path.join(inbox_threshold_folder, npy)

        Chainkeep = np.load(npyfile)
        
        if rank == 0:
            logging.info("rank {}, imodel {} / {}".format(rank, imodel, Nmodels))

        for ichain in range(Chainkeep.shape[0]):

            Chain = Chainkeep[ichain, :].copy()
            
            dg, dT, DensityModel = Chain2data(Chain, XnYnZn, Kernel_Grv, Kernel_Mag)
            
            ''' adding noise to data '''
            noise_g_level = 0.04
            sigma_g_original = noise_g_level*max(abs(dg))
            uncorr_noise_g = sigma_g_original*np.random.randn(len(dg))
            dg = dg + uncorr_noise_g

            noise_T_level = 0.06
            sigma_T_original = noise_T_level*max(abs(dT))
            uncorr_noise_T = sigma_T_original*np.random.randn(len(dT))
            dT = dT + uncorr_noise_T

            labels = (DensityModel*100).astype('int') ### mapping labels from float to int for rf classification

            tfb = data_balancing(labels, CX)
            labels = labels[tfb]
            
            features = jointdata2features([dg, dT], D[tfb], I[tfb], k_list)
              
            ''' balancing labels with imblearn'''
            # oversampling with ros has memory issue, we use undersampling with rus
            
            keys, values = np.unique(labels, return_counts=True)
            sampling_strategy_dict = {}
            for ikey, ival in zip(keys, values): 
                if ikey == 0:
                    ival = int(0.7 * ival) # we get only 70% of sediments otherwise the final prediction is biased to sediment 
                sampling_strategy_dict[ikey] = ival

            
            
#             n_salt = np.sum(labels<0)
#             n_base = np.sum(labels>0)
#             n_sed = int(np.sum(labels==0) * 0.7) # we get only 70% of sediments otherwise the final prediction is biased to sediment 
#             sampling_strategy = {label_salt: n_salt, label_sed: n_sed, label_base: n_base}
            
            rus = RandomUnderSampler(sampling_strategy=sampling_strategy_dict)
            features, labels = rus.fit_resample(features, labels)


            tf1 = labels <= 0
            tf2 = labels >= 0
            tf3 = (labels < 0) | (labels > 0)

            if rank == 0:

                if ichain % 2 == 0:

                    labels_training_list.append(labels)
                    features_training_list.append(features)

                else:

                    labels_testing_list.append(labels)
                    features_testing_list.append(features)


            elif rank == 1:

                if ichain % 2 == 0:

                    labels_training_list.append(labels[tf1])
                    features_training_list.append(features[tf1])

                else:

                    labels_testing_list.append(labels)
                    features_testing_list.append(features)


            elif rank == 2:

                if ichain % 2 == 0:

                    labels_training_list.append(labels[tf2])
                    features_training_list.append(features[tf2])

                else:

                    labels_testing_list.append(labels)
                    features_testing_list.append(features)


            elif rank == 3:

                if ichain % 2 == 0:

                    labels_training_list.append(labels[tf3])
                    features_training_list.append(features[tf3])

                else:

                    labels_testing_list.append(labels)
                    features_testing_list.append(features) 

            imodel += 1

        ##### End of collecting training samples #########################

labels_training = np.concatenate(labels_training_list)
features_training = np.concatenate(features_training_list)

labels_testing = np.concatenate(labels_testing_list)
features_testing = np.concatenate(features_testing_list)

npyfile = os.path.join(inbox_threshold_folder, npy_list[0])
Chainkeep = np.load(npyfile)
Chain = Chainkeep[0, :]     
dg, dT, DensityModel = Chain2data(Chain, XnYnZn, Kernel_Grv, Kernel_Mag)

noise_g_level = 0.04
sigma_g_original = noise_g_level*max(abs(dg))
uncorr_noise_g = sigma_g_original*np.random.randn(len(dg))

noise_T_level = 0.06
sigma_T_original = noise_T_level*max(abs(dT))
uncorr_noise_T = sigma_T_original*np.random.randn(len(dT))

uncorr_noise_g = comm.bcast(uncorr_noise_g, root=0)
uncorr_noise_T = comm.bcast(uncorr_noise_T, root=0)
comm.Barrier()

dg_validating = dg + uncorr_noise_g
dT_validating = dT + uncorr_noise_T


labels_validating = (DensityModel*100).astype('int') ### mapping labels from float to int for rf classification
features_validating = jointdata2features([dg_validating, dT_validating], D, I, k_list)

# labels_validating = labels_testing_list[-1]
# features_validating = features_testing_list[-1]
        
logging.info("starting rf fitting from rank: {}".format(rank))
    
rf0 = RandomForestClassifier(n_jobs = -1, n_estimators = n_estimators)

rf0.fit(features_training, labels_training)

logging.info("ending rf fitting from rank: {}".format(rank))


pred_testing0 = rf0.predict(features_testing)
clsrpt = classification_report(labels_testing, pred_testing0)
logging.info("="*30)
logging.info("classification_report from rank {}".format(rank))
logging.info("{}classification_report{}".format('\n', clsrpt))
logging.info("="*30)

prob_testing0 = rf0.predict_proba(features_testing)

prob_validating0 = rf0.predict_proba(features_validating)

features_field = jointdata2features([dg_obs, dT_obs], D, I, k_list)
prob_field0 = rf0.predict_proba(features_field)


classifier_name = "rf_" + str(rank) + ".pkl"
classifier_path = os.path.join(fpath_Models, classifier_name)
pickle.dump(rf0, open(classifier_path, 'wb'), pickle.HIGHEST_PROTOCOL)


if rank != 0:

    comm.send(prob_testing0, dest=0, tag=3)
    comm.send(prob_validating0, dest=0, tag=4)
    comm.send(prob_field0, dest=0, tag=5)

        


if rank == 0:
    
    rf_rcv = 0
    
    while rf_rcv != 9:
        
        comm.Probe(status=status)
        
        if status.tag == 3 and status.source == 1:   
            
            prob_testing1 = comm.recv(source = status.source, tag = 3)
            rf_rcv += 1
                
        if status.tag == 3 and status.source == 2:
            
            prob_testing2 = comm.recv(source = status.source, tag = 3)
            rf_rcv += 1
            
        if status.tag == 3 and status.source == 3:
            
            prob_testing3 = comm.recv(source = status.source, tag = 3)
            rf_rcv += 1

        if status.tag == 4 and status.source == 1:
            
            prob_validating1 = comm.recv(source = status.source, tag = 4)
            rf_rcv += 1
                
        if status.tag == 4 and status.source == 2:
            
            prob_validating2 = comm.recv(source = status.source, tag = 4)
            rf_rcv += 1
            
        if status.tag == 4 and status.source == 3:
            
            prob_validating3 = comm.recv(source = status.source, tag = 4)
            rf_rcv += 1      
            
        if status.tag == 5 and status.source == 1:
            
            prob_field1 = comm.recv(source = status.source, tag = 5)
            rf_rcv += 1
                
        if status.tag == 5 and status.source == 2:
            
            prob_field2 = comm.recv(source = status.source, tag = 5)
            rf_rcv += 1
            
        if status.tag == 5 and status.source == 3:
            
            prob_field3 = comm.recv(source = status.source, tag = 5)
            rf_rcv += 1              
            

    logging.info("starting log reg fitting from rank: {}".format(rank))

    

#     prob_testing = np.column_stack((prob_testing0, prob_testing1, prob_testing2, prob_testing3))
#     prob_validating = np.column_stack((prob_validating0, prob_validating1, prob_validating2, prob_validating3))
#     prob_field = np.column_stack((prob_field0, prob_field1, prob_field2, prob_field3))

    prob_testing = prob_testing0.copy()
    prob_validating = prob_validating0.copy()
    prob_field = prob_field0.copy()

    log_reg = LogisticRegressionCV(cv=10, random_state=0, n_jobs=-1, max_iter=1000)
    
    
    log_reg.fit(prob_testing, labels_testing)
    
    classifier_path = os.path.join(fpath_Models, "log_reg.pkl")
    pickle.dump(log_reg, open(classifier_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    
    logging.info("ending log reg fitting from rank: {}".format(rank))
    
            
    ########################################################################     
    
    #### prediction for validating data


    new_prediction = np.zeros((CZ,CY, CX))
    
    DensityModel_validating = labels_validating / 100.0
    
#     dg_validating = FM_sparse(DensityModel_validating, Kernel_Grv)
    
    SusModel_validating = DensityModel_validating / 50.0
    SusModel_validating[DensityModel_validating<0.2] = 0
    
#     dT_validating = FM_sparse(SusModel_validating, Kernel_Mag)

    labels_validating_pred = log_reg.predict(prob_validating)
    
    density_prediction = labels_validating_pred / 100.0
    
    dg_new_prediction = FM_sparse(density_prediction, Kernel_Grv)
    
    sus_prediction = density_prediction / 50.0
    sus_prediction[density_prediction<0.2] = 0
    
    dT_new_prediction = FM_sparse(sus_prediction, Kernel_Mag)
    
    
    np.save(fpath_Arrays +'//'+'dg_validating.npy', dg_validating)
    np.save(fpath_Arrays +'//'+'dg_new_prediction.npy', dg_new_prediction)
    np.save(fpath_Arrays +'//'+'DensityModel_validating.npy', DensityModel_validating)
    np.save(fpath_Arrays +'//'+'density_prediction.npy', density_prediction)
    
    np.save(fpath_Arrays +'//'+'dT_validating.npy', dT_validating)
    np.save(fpath_Arrays +'//'+'dT_new_prediction.npy', dT_new_prediction)
    np.save(fpath_Arrays +'//'+'SusModel_validating.npy', SusModel_validating)
    np.save(fpath_Arrays +'//'+'sus_prediction.npy', sus_prediction)

    logging.info("gravity data misfit for validation data: {}".format(data_misfit(dg_validating,
                                                                                  dg_new_prediction,
                                                                                  sigma_g_original)))
    
    logging.info("magnetic data misfit for validation data: {}".format(data_misfit(dT_validating,
                                                                                   dT_new_prediction,
                                                                                   sigma_T_original)))
    
    
    clsrpt = classification_report(labels_validating, labels_validating_pred)
    logging.info("="*30)
    logging.info("classification_report for log reg model from rank {}".format(rank))
    logging.info("{}classification_report for log reg model{}".format('\n', clsrpt))
    logging.info("="*30)
    
    labels_validating_prob = log_reg.predict_proba(prob_validating)
    uncertainty_validation = np.max(labels_validating_prob, axis=1)
    np.save(fpath_Arrays +'//'+'uncertainty_validation.npy', uncertainty_validation)


    

    ########### Apply on field data ############################
   
    field_density_prediction = log_reg.predict(prob_field)
    
    field_density_prediction = field_density_prediction / 100.0
    

    
    dg_fieldpred = FM_sparse(field_density_prediction, Kernel_Grv)
    

    
    field_sus_prediction = field_density_prediction / 50.0
    field_sus_prediction[field_density_prediction<0.2] = 0
    
    dT_fieldpred = FM_sparse(field_sus_prediction, Kernel_Mag)
    
    np.save(fpath_Arrays +'//'+'field_density_prediction.npy', field_density_prediction)
    np.save(fpath_Arrays +'//'+'field_sus_prediction.npy', field_sus_prediction)
    np.save(fpath_Arrays +'//'+'dg_fieldpred.npy', dg_fieldpred)
    np.save(fpath_Arrays +'//'+'dT_fieldpred.npy', dT_fieldpred)

    
    logging.info("gravity data misfit for feild data: {}".format(data_misfit(dg_obs, dg_fieldpred , sigma_dg_obs)))
    logging.info("magnetic data misfit for feild data: {}".format(data_misfit(dT_obs, dT_fieldpred , sigma_dT_obs)))

    field_density_prob = log_reg.predict_proba(prob_field)
    uncertainty_field = np.max(field_density_prob, axis=1)
    np.save(fpath_Arrays +'//'+'uncertainty_field.npy', uncertainty_field)


    ################################################################
    
    
    plot_field_data_model_slice(dg_obs, dg_fieldpred, dT_obs, dT_fieldpred, Ndatapoints,
                                xs, field_density_prediction, uncertainty_field,
                                xmodel, ymodel, zmodel, xticks=[650, 675, 700],
                                output_path=os.path.join(fpath_Figs, 'Field_model.pdf'))


    plot_valid_data_model_slice(dg_validating, dg_new_prediction, dT_validating, dT_new_prediction, Ndatapoints,
                                xs-xmodel.min(), DensityModel_validating,
                                density_prediction, uncertainty_validation,
                                xmodel-xmodel.min(), ymodel, zmodel, xticks=[10, 30, 50],
                                output_path=os.path.join(fpath_Figs, 'Validating_model.pdf'))
                               
                                
    ###############################################################################################################
    
    
    plot_datagrid(dg_obs, dg_fieldpred,  dT_obs, dT_fieldpred, xs, ys, Ndatapoints,
                  xticks=[650, 675, 700], yticks=[2690,2710,2730],
                  output_path=os.path.join(fpath_Figs, 'FieldData_Fit.pdf'))

    plot_datagrid(dg_validating, dg_new_prediction, dT_validating, dT_new_prediction, xs-xs.min(), ys-ys.min(), Ndatapoints,
                  xticks=[10, 30, 50], yticks=[0, 10, 20, 30, 40],
                  output_path=os.path.join(fpath_Figs, 'ValidData_Fit.pdf'))







    logging.info('The End')

MPI.Finalize
        
        
        