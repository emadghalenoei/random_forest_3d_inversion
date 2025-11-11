import numpy as np
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix
from datetime import datetime
import os
import logging
import sys
import scipy
from Gravity_3D_Kernel_Expanded_MD import Gravity_3D_Kernel_Expanded_MD
from Magnetic_3D_Kernel_Expanded_MD import Magnetic_3D_Kernel_Expanded_MD
from Kernel_compressor import Kernel_compressor
from Model_Generator import Model_Generator
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml



def run(input_config, constants_folder):

    # load input configuration for the script
    Ndatapoints = input_config['Ndatapoints']  # Number of total data in 1D array shape
    CX = input_config['CX']  # must be dividable by downsample rate of x
    CY = input_config['CY']  # must be dividable by downsample rate of x
    CZ = input_config['CZ']  # must be dividable by downsample rate of z
    gravity_data_std = input_config['gravity_data_std'] # mGal
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

    Gkernelsp_path = os.path.join(constants_folder, 'Gkernelsp.npz')
    Mkernelsp_path = os.path.join(constants_folder, 'Mkernelsp.npz')

    if os.path.exists(Gkernelsp_path) and os.path.exists(Mkernelsp_path):
        logging.info("sparse kernels exist")

        Gkernelsp = scipy.sparse.load_npz(Gkernelsp_path)
        # Mkernelsp = scipy.sparse.load_npz(Mkernelsp_path)

    else:

        for iz in tqdm(np.arange(CZ), desc='kernel'):
            c1 = iz * CX * CY
            c2 = (iz + 1) * CX * CY
            Kernel_Grv[:, c1:c2] = Gravity_3D_Kernel_Expanded_MD(X[iz, :, :], Y[iz, :, :], Z[iz, :, :], XS, YS, dx, dy, dz)
            Kernel_Mag[:, c1:c2] = Magnetic_3D_Kernel_Expanded_MD(X[iz, :, :], Y[iz, :, :], Z[iz, :, :], XS, YS, dx, dy, dz,
                                                                  Inc, Dec, Fe)

        Kernel_Grv = Kernel_Grv * 1e8
        Kernel_Grv = Kernel_Grv.astype('float32')
        Kernel_Mag = Kernel_Mag.astype('float32')

        # np.save(Gkernelsp_path, Kernel_Grv)
        # np.save(Mkernelsp_path, Kernel_Mag)

        [Gkernelsp, Mkernelsp] = Kernel_compressor(Ndatapoints, CX, CY, CZ, Kernel_Grv, Kernel_Mag)

        scipy.sparse.save_npz(Gkernelsp_path, Gkernelsp)
        scipy.sparse.save_npz(Mkernelsp_path, Mkernelsp)


    del Kernel_Grv, Kernel_Mag




    # XSn = np.divide(XS - x_min, x_max - x_min)
    # YSn = np.divide(YS - y_min, y_max - y_min)

    # XSn = XSn.flatten().astype('float32')
    # YSn = YSn.flatten().astype('float32')


    os.makedirs(training_data, exist_ok=True)

    def run_model():
        """Wrapper for Model_Generator call (no arguments needed if captured from scope)."""
        return Model_Generator(XnYnZn, rho_sed, rho_salt, rho_base, Kmin, Kmax, Gkernelsp, dg_obs, gravity_data_std)

    ChainKeep = np.zeros((NKEEP, 1 + Kmax * 4), dtype=np.float32)
    ikeep = 0
    isave = 0

    # ---- THREAD EXECUTION ----
    max_workers = os.cpu_count() - 3 or 1
    logging.info(f"Using {max_workers} threads.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_model) for _ in range(Nmodels)]

        for future in tqdm(as_completed(futures), total=len(futures), desc='Nmodels'):
            try:
                result = future.result()
            except Exception as e:
                logging.error(f"Model_Generator failed: {e}")
                continue

            ChainKeep[ikeep, :] = result
            ikeep += 1

            # ---- Periodic save ----
            if ikeep == NKEEP:
                daytime = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
                npy_name = f"training_{daytime}.npy"
                npy_path = os.path.join(training_data, npy_name)

                np.save(npy_path, ChainKeep)
                ikeep = 0
                ChainKeep.fill(0.0)
                isave += 1
                logging.info(f"Saved file #{isave}: {npy_name}")

    # Save remainder (if any)
    if ikeep > 0:
        daytime = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
        npy_name = f"training_{daytime}_final.npy"
        npy_path = os.path.join(training_data, npy_name)
        np.save(npy_path, ChainKeep[:ikeep])
        logging.info(f"Saved final batch with {ikeep} models.")



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

    input_config_path = os.path.join(constants_folder, "input_config.yaml")

    if os.path.exists(input_config_path):
        with open(input_config_path, "r") as file:
            input_config = yaml.safe_load(file)

    else:

        input_config = {}
        input_config['gravity_data_std'] = 0.2 # std of gravity data in mGal
        input_config['rtp_data_std'] = 1.0 # std of gravity data in nT
        input_config['gravity_data_path'] = 'field_data/GRV_2D_Data.txt'
        input_config['rtp_data_path'] = 'field_data/RTP_2D_Data.txt'
        input_config['Ndatapoints'] = 32  # Number of total data in 1D array shape

        # model space
        input_config['CX'] = 64 # must be dividable by downsample rate of x
        input_config['CY'] = 64 # must be dividable by downsample rate of y
        input_config['CZ'] = 64 # must be dividable by downsample rate of z
        input_config['Z0'] = 0
        input_config['ZEND'] = 10000
        input_config['Pad_Length'] = 5000 # to minimize the edge effects

        input_config['Inc'] = 90 * (3.14159265359 / 180)
        input_config['Dec'] = 0 * (3.14159265359 / 180)
        input_config['Fe'] = 43314  # (nT)

        input_config['rho_sed'] = [0.0]
        input_config['rho_salt'] = [-0.13]
        input_config['rho_base'] = [0.4]

        input_config['NKEEP'] = 50  # dump a binary file to desk every NKEEP records
        input_config['Nmodels'] = 1000000  # total number of models to be sampled for training

        input_config['Kmin'] = 6 # min number of Voronoi node in the model space
        input_config['Kmax'] = 50 # max number of Voronoi node in the model space

        input_config['training_data'] = 'training_data'

        input_config['max_num_training_model'] = 10000

        input_config['k_list'] = [3, 9, 15, 21, 27, 33, 39, 45, 51]  ### do not insert 1

        with open(os.path.join(constants_folder, "input_config.yaml"), "w") as f:
            yaml.safe_dump(input_config, f)


    run(input_config, constants_folder)