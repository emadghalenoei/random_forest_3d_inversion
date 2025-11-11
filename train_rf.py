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



def split_npy_files(path, max_num_training_model, train_ratio=0.9, train_yaml='train.yaml', test_yaml='test.yaml', seed=42):
    """
    Get all .npy files from a given path, split into training and test,
    and save just the file names (no full path) in YAML files.

    Parameters:
        path (str): Directory path containing .npy files.
        train_ratio (float): Fraction of files for training (default 0.9).
        train_yaml (str): Output YAML filename for training files.
        test_yaml (str): Output YAML filename for test files.
        seed (int): Random seed for reproducibility.
    """
    # Collect all .npy files
    files = glob.glob(os.path.join(path, '*.npy'))
    files.sort()  # optional: keep sorted order

    # Keep only base names
    files = [os.path.basename(f) for f in files]

    # Shuffle with fixed seed
    random.seed(seed)
    random.shuffle(files)

    files = files[:max_num_training_model]


    # Split
    n_train = int(len(files) * train_ratio)
    train_files = files[:n_train]
    test_files = files[n_train:]

    # Save to YAML
    with open(os.path.join(path, train_yaml), 'w') as f:
        yaml.safe_dump(train_files, f)

    with open(os.path.join(path, test_yaml), 'w') as f:
        yaml.safe_dump(test_files, f)

    logging.info(f"Total files: {len(files)}, Train: {len(train_files)}, Test: {len(test_files)}")
    logging.info(f"Train YAML saved to {train_yaml}, Test YAML saved to {test_yaml}")



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


def run(input_config, constants_folder):

    # load input configuration for the script
    Ndatapoints = input_config['Ndatapoints']  # Number of total data in 1D array shape
    CX = input_config['CX']  # must be dividable by downsample rate of x
    CY = input_config['CY']  # must be dividable by downsample rate of x
    CZ = input_config['CZ']  # must be dividable by downsample rate of z
    gravity_data_std = input_config['gravity_data_std']  # mGal
    rtp_data_std = input_config['rtp_data_std'] # nT
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

    k_list = input_config['k_list']

    index = faiss.IndexFlatL2(3)
    index.add(np.column_stack((XSn, YSn, 0.0 * XSn)))

    D, I = index.search(XnYnZn, max(k_list))
    #####################################################################

    # npy_list = glob.glob(os.path.join(training_data, '*.npy'))

    # npy_list = npy_list[:max_num_training_model]

    labels_list = []
    features_list = []

    split_npy_files(training_data, max_num_training_model,
                    train_ratio=0.9, train_yaml='train.yaml', test_yaml='test.yaml', seed=42)

    with open(os.path.join(training_data, 'train.yaml'), 'r') as f:
        train_files = yaml.safe_load(f)

    for npy_file in tqdm(train_files, desc='processing train data'):

        npy_file = os.path.join(training_data, npy_file)

        Chainkeep = np.load(npy_file)

        for ichain in range(Chainkeep.shape[0]):

            Chain = Chainkeep[ichain, :].copy()

            dg, dT, DensityModel = Chain2data(Chain, XnYnZn, Gkernelsp, Mkernelsp)

            ''' adding noise to data '''
            uncorr_noise_g = gravity_data_std * np.random.randn(len(dg))
            dg = dg + uncorr_noise_g

            uncorr_noise_T = rtp_data_std * np.random.randn(len(dT))
            dT = dT + uncorr_noise_T

            labels = (DensityModel * 100).astype('int')  ### mapping labels from float to int for rf classification

            tfb = data_balancing(labels, CX)
            labels = labels[tfb]

            features = jointdata2features([dg, dT], D[tfb], I[tfb], k_list)

            ''' balancing labels with imblearn'''
            # oversampling with ros has memory issue, we use undersampling with rus

            keys, values = np.unique(labels, return_counts=True)
            sampling_strategy_dict = {}
            for ikey, ival in zip(keys, values):
                if ikey == 0:
                    ival = int(
                        0.7 * ival)  # we get only 70% of sediments otherwise the final prediction is biased to sediment
                sampling_strategy_dict[ikey] = ival

            rus = RandomUnderSampler(sampling_strategy=sampling_strategy_dict)
            features, labels = rus.fit_resample(features, labels)

            labels_list.append(labels)
            features_list.append(features)

    ##############################################

    labels = np.concatenate(labels_list).astype('float32')
    features = np.concatenate(features_list).astype('float32')

    del labels_list, features_list
    del D, I

    # --- Classifier (direction) ---
    clf = RandomForestClassifier(random_state=42, n_jobs = -1)
    tscv = TimeSeriesSplit(n_splits=5)
    accs, f1s = [], []

    for tr, va in tscv.split(features):
        clf.fit(features[tr], labels[tr])
        preds = clf.predict(features[va])
        accs.append(accuracy_score(labels[va], preds))
        f1s.append(f1_score(labels[va], preds, average='weighted'))

    acc_mean = float(np.mean(accs).round(3))
    f1_mean = float(np.mean(f1s).round(3))

    logging.info("TimeSeriesCV Accuracy: {}".format(acc_mean))
    logging.info("TimeSeriesCV F1: {}".format(f1_mean))
    logging.info("fitting all data to rf")
    clf.fit(features, labels)

    # Save classifier
    clf_path = os.path.join(constants_folder, "rf_inversion.joblib")
    joblib.dump(clf, clf_path)
    logging.info("clf saved to {}".format(clf_path))


    #################################################################################




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

    # output_folder = os.path.join(os.getcwd(), 'rf')
    # os.makedirs(output_folder, exist_ok=True)


    with open(os.path.join(constants_folder, 'input_config.yaml'), "r") as file:
        input_config = yaml.safe_load(file)

    run(input_config, constants_folder)