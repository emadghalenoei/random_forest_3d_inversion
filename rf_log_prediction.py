import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegressionCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


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
from PLOT import plot_datagrid, plot_field_data_model_slice, plot_valid_data_model_slice, plot_datainbox, plot_3dmodel, plot_prob_calib
# from rf_log_classifier_joint_3D import data_misfit, Chain2data
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


def prob_calib(label, pos_label, prob_validating, labels_validating):


    p1 = prob_validating[:, pos_label]


    pred_probs_space = np.linspace(p1.min(), p1.max(), 10)

    true_prob_list = []
    mean_prob_list = []

    ###

    for i in range(len(pred_probs_space)-1):

        prob_cond = np.where((p1 > pred_probs_space[i]) & (p1 < pred_probs_space[i+1]))[0]

        true_prob = np.sum(labels_validating[prob_cond] == label) / len(labels_validating[prob_cond])

        mean_prob = np.mean(p1[prob_cond])

        true_prob_list.append(true_prob)

        mean_prob_list.append(mean_prob)

    return true_prob_list, mean_prob_list



def data_misfit(dg_1, dg_2, sigma):
    misfit = np.sum(((dg_1 - dg_2) / sigma)**2)
    return misfit

#######################################################################

folder_name = '2022_12_26-10_49_24_AM'

fpath_loaddesk = os.getcwd()+'/loaddesk'

fpath_Arrays = os.getcwd()+'//'+folder_name+'//'+'Arrays'
fpath_Figs = os.getcwd()+'//'+folder_name+'//'+'Figs'
fpath_Models = os.getcwd()+'//'+folder_name+'//'+'Models'
fpath_Predictions = os.getcwd()+'//'+folder_name+'//'+'Predictions'
# fpath_Predictions_Figs = os.getcwd()+'//'+folder_name+'//'+'Predictions'+'//'+'Figs'

npyfile_index = 15
fpath_Predictions_npy = os.getcwd()+'//'+folder_name+'//'+'Predictions' +'//'+'npy'+str(npyfile_index)
fpath_Predictions_npy_Figs = os.getcwd()+'//'+folder_name+'//'+'Predictions' +'//'+'npy'+str(npyfile_index) + '//' + 'Figs'

if rank == 0:
    if not os.path.exists(fpath_Predictions):
        os.mkdir(fpath_Predictions)
    
    if not os.path.exists(fpath_Predictions_npy):
        os.mkdir(fpath_Predictions_npy)
        
    if not os.path.exists(fpath_Predictions_npy_Figs):
        os.mkdir(fpath_Predictions_npy_Figs)   


comm.Barrier()

    
log_filename = os.path.join(fpath_Predictions_npy, 'output.log')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(filename)s | %(message)s',
                    handlers = [logging.FileHandler(log_filename, mode='w'), logging.StreamHandler(sys.stdout)])
    
    
    
    
inbox_threshold = 0.95
fpath_training_dataset = os.getcwd()+'/training_dataset'
inbox_threshold_folder = os.path.join(fpath_training_dataset, str(int(inbox_threshold*100)))



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



XSn = np.divide(XS-x_min,x_max-x_min)
YSn = np.divide(YS-y_min,y_max-y_min)

XSn = XSn.flatten().astype('float32')
YSn = YSn.flatten().astype('float32')

k_list = [3, 9, 15, 21, 27, 33, 39, 45, 51] ### do not insert 1
if rank == 0:
    logging.info("k_list: {}".format(k_list))
    
    
    
    
D = np.zeros((XnYnZn.shape[0], max(k_list))).astype('float32')
I = np.zeros((XnYnZn.shape[0], max(k_list))).astype('int')

if rank == 0:
    index = faiss.IndexFlatL2(3)
    index.add(np.column_stack((XSn, YSn, 0.0*XSn)))
    
    D, I = index.search(XnYnZn, max(k_list))

D = comm.bcast(D, root=0)
I = comm.bcast(I, root=0)
comm.Barrier()



npy_list = os.listdir(inbox_threshold_folder)
npyfile = os.path.join(inbox_threshold_folder, npy_list[npyfile_index])
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

features_field = jointdata2features([dg_obs, dT_obs], D, I, k_list)


classifier_name = "rf_" + str(rank) + ".pkl"
classifier_path = os.path.join(fpath_Models, classifier_name)
rf0 = pickle.load(open(classifier_path, 'rb'))


pred_validating0 = rf0.predict(features_validating)

clsrpt = classification_report(labels_validating, pred_validating0)
logging.info("="*30)
logging.info("classification_report from rank {}".format(rank))
logging.info("{}classification_report{}".format('\n', clsrpt))
logging.info("="*30)



prob_validating0 = rf0.predict_proba(features_validating)

prob_field0 = rf0.predict_proba(features_field)


if rank != 0:

    comm.send(prob_validating0, dest=0, tag=4)
    comm.send(prob_field0, dest=0, tag=5)


    
if rank == 0:
    
    rf_rcv = 0
    
    while rf_rcv != 6:
        
        comm.Probe(status=status)
        
        
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
            
            
            
#     prob_validating = np.column_stack((prob_validating0, prob_validating1, prob_validating2, prob_validating3))
#     prob_field = np.column_stack((prob_field0, prob_field1, prob_field2, prob_field3))
    
    prob_validating = prob_validating0.copy()
    prob_field = prob_field0.copy()

    
    classifier_path = os.path.join(fpath_Models, "log_reg.pkl")
    log_reg = pickle.load(open(classifier_path, 'rb'))
    
    ###############################################################################
    
    new_prediction = np.zeros((CZ, CY, CX))
    
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
    
    
    np.save(fpath_Predictions_npy +'//'+'dg_validating.npy', dg_validating)
    np.save(fpath_Predictions_npy +'//'+'dg_new_prediction.npy', dg_new_prediction)
    np.save(fpath_Predictions_npy +'//'+'DensityModel_validating.npy', DensityModel_validating)
    np.save(fpath_Predictions_npy +'//'+'density_prediction.npy', density_prediction)
    
    np.save(fpath_Predictions_npy +'//'+'dT_validating.npy', dT_validating)
    np.save(fpath_Predictions_npy +'//'+'dT_new_prediction.npy', dT_new_prediction)
    np.save(fpath_Predictions_npy +'//'+'SusModel_validating.npy', SusModel_validating)
    np.save(fpath_Predictions_npy +'//'+'sus_prediction.npy', sus_prediction)
    
    clsrpt = classification_report(labels_validating, labels_validating_pred)
    logging.info("="*30)
    logging.info("classification_report for log reg model from rank {}".format(rank))
    logging.info("{}classification_report for log reg model{}".format('\n', clsrpt))
    logging.info("="*30)
    
    labels_validating_prob = log_reg.predict_proba(prob_validating)
    uncertainty_validation = np.max(labels_validating_prob, axis=1)
    np.save(fpath_Predictions_npy +'//'+'uncertainty_validation.npy', uncertainty_validation)
    
    ####################################################
    iy_data_valid = 11
    ymodel_valid = ymodel-ymodel.min()
    ys_valid =  ys-ymodel.min()
    iy_valid = np.argmin(abs(ymodel_valid - ys_valid[iy_data_valid]),0)
    logging.info("ys_valid[iy_data_valid] {}".format(ys_valid[iy_data_valid]))
    logging.info("ymodel_valid[iy_valid] {}".format(ymodel_valid[iy_valid]))

    ####################################################
    
    plot_valid_data_model_slice(dg_validating, dg_new_prediction, dT_validating, dT_new_prediction,
                                Ndatapoints, xs-xmodel.min(), ys-ymodel.min(), DensityModel_validating,
                                density_prediction, uncertainty_validation,
                                xmodel-xmodel.min(), ymodel-ymodel.min(), zmodel,  iy_data_valid, iy_valid, xticks=[10, 30, 50], 
                                output_path=os.path.join(fpath_Predictions_npy_Figs, 'Validating_model.pdf'))
    
    
    
    logging.info("gravity data misfit for validation data: {}".format(data_misfit(dg_validating,
                                                                                  dg_new_prediction,
                                                                                  sigma_g_original)))
    
    logging.info("magnetic data misfit for validation data: {}".format(data_misfit(dT_validating,
                                                                                   dT_new_prediction,
                                                                                   sigma_T_original)))
    
    
    
    
    ########### Apply on field data ############################
   
    field_density_prediction = log_reg.predict(prob_field)
    
    field_density_prediction = field_density_prediction / 100.0

    dg_fieldpred = FM_sparse(field_density_prediction, Kernel_Grv)
    
    field_sus_prediction = field_density_prediction / 50.0
    field_sus_prediction[field_density_prediction<0.2] = 0
    
    dT_fieldpred = FM_sparse(field_sus_prediction, Kernel_Mag)
    
    np.save(fpath_Predictions_npy +'//'+'field_density_prediction.npy', field_density_prediction)
    np.save(fpath_Predictions_npy +'//'+'field_sus_prediction.npy', field_sus_prediction)
    np.save(fpath_Predictions_npy +'//'+'dg_fieldpred.npy', dg_fieldpred)
    np.save(fpath_Predictions_npy +'//'+'dT_fieldpred.npy', dT_fieldpred)


    field_density_prob = log_reg.predict_proba(prob_field)
    uncertainty_field = np.max(field_density_prob, axis=1)
    np.save(fpath_Predictions_npy +'//'+'uncertainty_field.npy', uncertainty_field)


    ################################################################
    iy_data = 9
#     iy = 44
    iy = np.argmin(abs(ymodel - ys[iy_data]),0)
    logging.info("ys[iy_data] {}".format(ys[iy_data]))
    logging.info("ymodel[iy] {}".format(ymodel[iy]))
    logging.info("iy {}".format(iy))
                                
                                
    plot_field_data_model_slice(dg_obs, dg_fieldpred, dT_obs, dT_fieldpred, Ndatapoints,
                                xs, ys, field_density_prediction, uncertainty_field,
                                xmodel, ymodel, zmodel, iy_data, iy, xticks=[650, 675, 700],
                                output_path=os.path.join(fpath_Predictions_npy_Figs, 'Field_model.pdf'))
    
    
    

    ###############################################################################################################
    plot_datagrid(dg_obs, dg_fieldpred,  dT_obs, dT_fieldpred, xs, ys, Ndatapoints,
              iy_data, xticks=[650, 675, 700], yticks=[2690,2710,2730],
              output_path=os.path.join(fpath_Predictions_npy_Figs, 'FieldData_Fit.pdf'))

    plot_datagrid(dg_validating, dg_new_prediction, dT_validating, dT_new_prediction, xs-xs.min(), ys-ys.min(), Ndatapoints,
              iy_data_valid, xticks=[10, 30, 50], yticks=[0, 10, 20, 30, 40],
              output_path=os.path.join(fpath_Predictions_npy_Figs, 'ValidData_Fit.pdf'))
    
    
    ###############################################################################################################
    '''just noticed rf0 works really good and even better than log reg'''
    
    DensityModel_validating_rf0 = pred_validating0 / 100.0
        
    SusModel_validating_rf0 = DensityModel_validating_rf0 / 50.0
    SusModel_validating_rf0[DensityModel_validating_rf0<0.2] = 0
        
    
    dg_validating_rf0 = FM_sparse(DensityModel_validating_rf0, Kernel_Grv)
    
    dT_validating_rf0 = FM_sparse(SusModel_validating_rf0, Kernel_Mag)
    
    
    np.save(fpath_Predictions_npy +'//'+'dg_validating_rf0.npy', dg_validating_rf0)
    np.save(fpath_Predictions_npy +'//'+'DensityModel_validating_rf0.npy', DensityModel_validating_rf0)
    
    np.save(fpath_Predictions_npy +'//'+'dT_validating_rf0.npy', dT_validating_rf0)
    np.save(fpath_Predictions_npy +'//'+'SusModel_validating_rf0.npy', SusModel_validating_rf0)
    
    
#     labels_validating_prob_rf0 = rf0.predict_proba(features_validating)
    
    uncertainty_validation_rf0= np.max(prob_validating0, axis=1)
    np.save(fpath_Predictions_npy +'//'+'uncertainty_validation_rf0.npy', uncertainty_validation_rf0)
    
    plot_valid_data_model_slice(dg_validating, dg_validating_rf0, dT_validating, dT_validating_rf0,
                                Ndatapoints, xs-xmodel.min(),  ys-ymodel.min(), DensityModel_validating,
                                DensityModel_validating_rf0, uncertainty_validation_rf0,
                                xmodel-xmodel.min(), ymodel-ymodel.min(), zmodel, iy_data_valid, iy_valid, xticks=[10, 30, 50],
                                output_path=os.path.join(fpath_Predictions_npy_Figs, 'Validating_model_rf0.pdf'))
    
    
    
    logging.info("gravity data misfit for validation data (rf0): {}".format(data_misfit(dg_validating,
                                                                                  dg_validating_rf0,
                                                                                  sigma_g_original)))
    
    logging.info("magnetic data misfit for validation data (rf0): {}".format(data_misfit(dT_validating,
                                                                                   dT_validating_rf0,
                                                                                   sigma_T_original)))
    
    plot_datagrid(dg_validating, dg_validating_rf0, dT_validating, dT_validating_rf0, xs-xs.min(), ys-ys.min(), Ndatapoints, 
                  iy_data_valid, xticks=[10, 30, 50], yticks=[0, 10, 20, 30, 40],
                  output_path=os.path.join(fpath_Predictions_npy_Figs, 'ValidData_Fit_rf0.pdf'))
    
    
    
    
    ####################################################################################################
    '''rf0 over field data'''
    pred_field0 = rf0.predict(features_field)

    field_density_prediction_rf0 = pred_field0 / 100.0

    dg_fieldpred_rf0 = FM_sparse(field_density_prediction_rf0, Kernel_Grv)
    
    field_sus_prediction_rf0 = field_density_prediction_rf0 / 50.0
    field_sus_prediction_rf0[field_density_prediction_rf0<0.2] = 0
    
    dT_fieldpred_rf0 = FM_sparse(field_sus_prediction_rf0, Kernel_Mag)
    
    uncertainty_field_rf0 = np.max(prob_field0, axis=1)
    
    np.save(fpath_Predictions_npy +'//'+'field_density_prediction_rf0.npy', field_density_prediction_rf0)
    np.save(fpath_Predictions_npy +'//'+'field_sus_prediction_rf0.npy', field_sus_prediction_rf0)
    np.save(fpath_Predictions_npy +'//'+'dg_fieldpred_rf0.npy', dg_fieldpred_rf0)
    np.save(fpath_Predictions_npy +'//'+'dT_fieldpred_rf0.npy', dT_fieldpred_rf0)
    np.save(fpath_Predictions_npy +'//'+'uncertainty_field_rf0.npy', uncertainty_field_rf0)

    ################################################################
                                
                                
    plot_field_data_model_slice(dg_obs, dg_fieldpred_rf0, dT_obs, dT_fieldpred_rf0, Ndatapoints,
                                xs, ys, field_density_prediction_rf0, uncertainty_field_rf0,
                                xmodel, ymodel, zmodel, iy_data, iy, xticks=[650, 675, 700],
                                output_path=os.path.join(fpath_Predictions_npy_Figs, 'Field_model_rf0.pdf'))
    
    plot_datagrid(dg_obs, dg_fieldpred_rf0,  dT_obs, dT_fieldpred_rf0, xs, ys, Ndatapoints,
                  iy_data, xticks=[650, 675, 700], yticks=[2690,2710,2730],
                  output_path=os.path.join(fpath_Predictions_npy_Figs, 'FieldData_Fit_rf0.pdf'))
    
    ########################################################################
    
    PMD_g = np.load(fpath_loaddesk+'//'+'PMD_g.npy')

    plot_3dmodel(xmodel, ymodel, zmodel, field_density_prediction, uncertainty_field, PMD_g,
                 iy, xticks=[650, 675, 700], yticks=[2690,2730],
                 output_path=os.path.join(fpath_Predictions_npy_Figs, '3Dmodel_one_view.pdf'))
    
    plot_3dmodel(xmodel, ymodel, zmodel, field_density_prediction_rf0, uncertainty_field_rf0, PMD_g,
                 iy, xticks=[650, 675, 700], yticks=[2690,2730],
                 output_path=os.path.join(fpath_Predictions_npy_Figs, '3Dmodel_one_view_rf0.pdf'))
    
    plot_datainbox(npyfile, XnYnZn, Kernel_Grv, Kernel_Mag, Ndatapoints, xs,
                   output_path=os.path.join(fpath_Predictions_npy_Figs, 'datainbox.pdf'))
    
    
    ####################################################################################
    '''probability calibration'''
    
#     logging.info("log loss for rf0 is {}".format(log_loss(labels_validating, prob_validating0))) 
#     logging.info("log loss for log_reg is {}".format(log_loss(labels_validating, labels_validating_prob))) 
    
    pos_label = np.argwhere(rf0.classes_ == -13)[0]
          
    true_prob_list_rf0, mean_prob_list_rf0 = prob_calib(-13, pos_label, prob_validating0, labels_validating)
       
    pos_label = np.argwhere(log_reg.classes_ == -13)[0]

    true_prob_list_logreg, mean_prob_list_logreg = prob_calib(-13, pos_label, labels_validating_prob, labels_validating)
    
    plot_prob_calib(mean_prob_list_rf0, true_prob_list_rf0, mean_prob_list_logreg, true_prob_list_logreg,
                    output_path=os.path.join(fpath_Predictions_npy_Figs, 'prob_calib.pdf'))


    



    logging.info('The End')

MPI.Finalize
        
        




