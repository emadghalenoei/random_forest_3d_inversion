from datetime import datetime
import os
import sys
import logging
import yaml
import glob
import numpy as np
from Chain2data import Chain2data
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix
import scipy
from Gravity_3D_Kernel_Expanded_MD import Gravity_3D_Kernel_Expanded_MD
from Magnetic_3D_Kernel_Expanded_MD import Magnetic_3D_Kernel_Expanded_MD
from tqdm import tqdm
from Kernel_compressor import Kernel_compressor
from imblearn.under_sampling import RandomUnderSampler
from jointdata2features import jointdata2features
import faiss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import joblib
import random
from FM_sparse import FM_sparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import PLOT



def run(input_config, constants_folder, output_folder):
    # load input configuration for the script
    Ndatapoints = input_config['Ndatapoints']  # Number of total data in 1D array shape
    CX = input_config['CX']  # must be dividable by downsample rate of x
    CY = input_config['CY']  # must be dividable by downsample rate of x
    CZ = input_config['CZ']  # must be dividable by downsample rate of z
    gravity_data_std = input_config['gravity_data_std']  # mGal
    rtp_data_std = input_config['rtp_data_std']  # nT
    gravity_data_path = input_config['gravity_data_path']
    rtp_data_path = input_config['rtp_data_path']
    # model space
    Z0 = input_config['Z0']
    ZEND = input_config['ZEND']
    Pad_Length = input_config['Pad_Length']

    rho_sed = input_config['rho_sed']
    rho_salt = input_config['rho_salt']
    rho_base = input_config['rho_base']

    Inc = input_config['Inc']
    Dec = input_config['Dec']
    Fe = input_config['Fe']

    NKEEP = input_config['NKEEP']  # dump a binary file to desk every NKEEP records
    Nmodels = input_config['Nmodels']
    Kmin = input_config['Kmin']
    Kmax = input_config['Kmax']

    training_data = input_config['training_data']

    max_num_training_model = input_config['max_num_training_model']

    #####################################################################

    Kernel_Grv = np.zeros((Ndatapoints * Ndatapoints, CX * CY * CZ)).astype('float32')
    Kernel_Mag = np.zeros((Ndatapoints * Ndatapoints, CX * CY * CZ)).astype('float32')

    Gravity_Data = np.loadtxt(gravity_data_path).astype('float32')
    Magnetic_Data = np.loadtxt(rtp_data_path).astype('float32')

    xs = np.linspace(np.amin(Gravity_Data[:, 0]), np.amax(Gravity_Data[:, 0]), Ndatapoints)
    ys = np.flip(np.linspace(np.amin(Gravity_Data[:, 1]), np.amax(Gravity_Data[:, 1]), Ndatapoints), 0)

    XS, YS = np.meshgrid(xs, ys)
    GRV_obs = griddata(Gravity_Data[:, 0:2], Gravity_Data[:, 2], (XS, YS), method='nearest')
    RTP_obs = griddata(Magnetic_Data[:, 0:2], Magnetic_Data[:, 2], (XS, YS), method='nearest')

    dg_obs = GRV_obs.flatten('C')
    dT_obs = RTP_obs.flatten('C')

    xmodel = np.linspace(np.amin(XS) - Pad_Length, np.amax(XS) + Pad_Length, CX)
    ymodel = np.flip(np.linspace(np.amin(YS) - Pad_Length, np.amax(YS) + Pad_Length, CY))
    zmodel = np.linspace(Z0, ZEND, CZ)
    Y, Z, X = np.meshgrid(ymodel, zmodel, xmodel)
    dx = abs(X[0, 0, 1] - X[0, 0, 0])
    dy = abs(Y[0, 1, 0] - Y[0, 0, 0])
    dz = abs(Z[1, 0, 0] - Z[0, 0, 0])

    # XYZ = np.column_stack((X.flatten(), Y.flatten(), Z.flatten())).astype('float32')

    x_min = np.min(X) - dx / 2
    x_max = np.max(X) + dx / 2
    y_min = np.min(Y) - dy / 2
    y_max = np.max(Y) + dy / 2
    z_min = np.min(Z) - dz / 2
    z_max = np.max(Z) + dz / 2

    Xn_3D = np.divide(X - x_min, x_max - x_min)
    Yn_3D = np.divide(Y - y_min, y_max - y_min)
    Zn_3D = np.divide(Z - z_min, z_max - z_min)

    Xn = Xn_3D.flatten()
    Yn = Yn_3D.flatten()
    Zn = Zn_3D.flatten()
    XnYnZn = np.column_stack((Xn, Yn, Zn)).astype('float32')

    del Xn, Yn, Zn, Xn_3D, Yn_3D, Zn_3D

    Gkernelsp_path = os.path.join(constants_folder, 'Gkernelsp.npz')
    Mkernelsp_path = os.path.join(constants_folder, 'Mkernelsp.npz')

    if os.path.exists(Gkernelsp_path) and os.path.exists(Mkernelsp_path):
        logging.info("sparse kernels exist")

        Gkernelsp = scipy.sparse.load_npz(Gkernelsp_path)
        Mkernelsp = scipy.sparse.load_npz(Mkernelsp_path)

    else:

        for iz in tqdm(np.arange(CZ), desc='kernel'):
            c1 = iz * CX * CY
            c2 = (iz + 1) * CX * CY
            Kernel_Grv[:, c1:c2] = Gravity_3D_Kernel_Expanded_MD(X[iz, :, :], Y[iz, :, :], Z[iz, :, :], XS, YS, dx, dy,
                                                                 dz)
            Kernel_Mag[:, c1:c2] = Magnetic_3D_Kernel_Expanded_MD(X[iz, :, :], Y[iz, :, :], Z[iz, :, :], XS, YS, dx, dy,
                                                                  dz,
                                                                  Inc, Dec, Fe)

        Kernel_Grv = Kernel_Grv * 1e8
        Kernel_Grv = Kernel_Grv.astype('float32')
        Kernel_Mag = Kernel_Mag.astype('float32')

        [Gkernelsp, Mkernelsp] = Kernel_compressor(Ndatapoints, CX, CY, CZ, Kernel_Grv, Kernel_Mag)

        scipy.sparse.save_npz(Gkernelsp_path, Gkernelsp)
        scipy.sparse.save_npz(Mkernelsp_path, Mkernelsp)

    del Kernel_Grv, Kernel_Mag
    del X, Y, Z

    XSn = np.divide(XS - x_min, x_max - x_min)
    YSn = np.divide(YS - y_min, y_max - y_min)

    XSn = XSn.flatten().astype('float32')
    YSn = YSn.flatten().astype('float32')

    # k_list = [3, 9, 15, 21, 27, 33, 39, 45, 51]  ### do not insert 1
    k_list = input_config['k_list']

    index = faiss.IndexFlatL2(3)
    index.add(np.column_stack((XSn, YSn, 0.0 * XSn)))

    D, I = index.search(XnYnZn, max(k_list))

    #####################################################################

    clf = joblib.load(os.path.join(constants_folder, "rf_inversion.joblib"))

    features_field = jointdata2features([dg_obs, dT_obs], D, I, k_list)

    DensityModel_pred = clf.predict(features_field)
    DensityModel_prob = clf.predict_proba(features_field)
    DensityModel_prob = np.max(DensityModel_prob, axis=1)

    DensityModel_pred = DensityModel_pred / 100.0
    SusModel_pred = DensityModel_pred / 50.0
    SusModel_pred[DensityModel_pred < 0.2] = 0

    dg_prediction = FM_sparse(DensityModel_pred, Gkernelsp)
    dT_prediction = FM_sparse(SusModel_pred, Mkernelsp)


    output_field = os.path.join(output_folder, 'field_results')
    os.makedirs(output_field, exist_ok=True)

    np.save(os.path.join(output_field, 'dg_field_prediction.npy'), dg_prediction)
    np.save(os.path.join(output_field, 'dT_field_prediction.npy'), dT_prediction)
    np.save(os.path.join(output_field, 'DensityModel_field.npy'), DensityModel_pred)
    np.save(os.path.join(output_field, 'SusModel_field.npy'), SusModel_pred)
    np.save(os.path.join(output_field, 'DensityModel_prob.npy'), DensityModel_prob)


    ####################################################################################
    max_index = np.argmax(dT_obs.reshape((Ndatapoints, Ndatapoints)))  # index in flattened array
    iy_data, ix_data = np.unravel_index(max_index, dT_obs.reshape((Ndatapoints, Ndatapoints)).shape)

    iy = np.argmin(abs(ymodel - ys[iy_data]), 0)
    logging.info("ys[iy_data] {}".format(ys[iy_data]))
    logging.info("ymodel[iy] {}".format(ymodel[iy]))
    logging.info("iy {}".format(iy))

    ####################################################

    PLOT.plot_field_data_model_slice(dg_obs, dg_prediction, dT_obs, dT_prediction, Ndatapoints,
                                     xs, ys, DensityModel_pred, DensityModel_prob,
                                     xmodel, ymodel, zmodel, iy_data, iy, xticks=[650, 675, 700],
                                     output_path=os.path.join(output_field, 'Field_model_rf.pdf'))

    PLOT.plot_datagrid(dg_obs, dg_prediction, dT_obs, dT_prediction, xs, ys, Ndatapoints,
                       iy_data, xticks=[650, 675, 700], yticks=[2690, 2710, 2730],
                       output_path=os.path.join(output_field, 'FieldData_Fit_rf.pdf'))

    ########################################################################

    PMD_g = np.load(os.path.join(constants_folder, 'PMD_g.npy'))

    PLOT.plot_3dmodel(xmodel, ymodel, zmodel, DensityModel_pred, DensityModel_prob, PMD_g,
                      iy, xticks=[650, 675, 700], yticks=[2690, 2730],
                      output_path=os.path.join(output_field, '3Dmodel_one_view.pdf'))

    ########################################################################
    basement_path = os.path.join(constants_folder, 'basement_ghasha.txt')
    salt_path = os.path.join(constants_folder, 'saltdome_gasha.txt')

    PLOT.plot_density_4slices(DensityModel_pred, xmodel, ymodel, zmodel,
                              basement_path, salt_path,
                              iy_list=[28, 21], ix_list=[21, 30],
                              vmin=-0.4, vmax=0.4,
                              cmap='seismic',
                              output_path=os.path.join(output_field, 'compare_regularized.pdf'))


    ###############################################################################################################


if __name__ == "__main__":

    start_time = datetime.now()
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    log_filename = start_time.strftime("%Y_%m_%d_%I_%M_%S_%p") + '.log'
    log_filename = os.path.splitext(os.path.basename(__file__))[0] + '_' + log_filename
    log_path = os.path.join(log_dir, log_filename)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set global logging level
    # Remove any existing handlers (important!)
    logger.handlers.clear()
    # Create formatter
    formatter = logging.Formatter('%(asctime)s | %(filename)s | %(message)s')
    # Create and add file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Create and add stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    constants_folder  = os.path.join(os.getcwd(), 'constants')
    os.makedirs(constants_folder, exist_ok=True)

    output_folder = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_folder, exist_ok=True)


    with open(os.path.join(constants_folder, 'input_config.yaml'), "r") as file:
        input_config = yaml.safe_load(file)

    run(input_config, constants_folder, output_folder)