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
from sklearn.metrics import classification_report
import pprint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from scipy.stats import chi2
from math import ceil

def calibration_metrics(y_true, y_prob, n_bins=10, bootstrap_ci=False, n_boot=1000, random_state=None):
    """
    Compute calibration/fitness metrics for a binary probability predictor.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Binary labels (0/1).
    y_prob : array-like, shape (n_samples,)
        Predicted probability for the positive class.
    n_bins : int
        Number of bins for calibration (uniform probability bins).
    bootstrap_ci : bool
        If True, compute bootstrap CIs for ECE and Brier (slow).
    n_boot : int
        Number of bootstrap samples if bootstrap_ci True.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    results : dict
        Keys: brier, brier_skill, logloss, ece, mce, hl_stat, hl_pvalue, cal_curve (dict with prob_pred, prob_true, bin_counts)
        If bootstrap_ci True, includes ece_ci and brier_ci (tuples).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = y_true.size
    if n == 0:
        raise ValueError("Empty inputs")

    # Basic metrics
    brier = brier_score_loss(y_true, y_prob)            # lower is better
    # Brier skill score vs. climatology (base rate)
    p_ref = y_true.mean()
    brier_ref = np.mean((y_true - p_ref)**2)
    brier_skill = 1.0 - brier / brier_ref if brier_ref > 0 else np.nan
    logloss = log_loss(y_true, y_prob, labels=[0,1])

    # Calibration curve (sklearn)
    prob_true_bins, prob_pred_bins = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

    # Compute ECE and MCE (uniform-prob bins)
    # We'll compute per-bin accuracy (acc_j), mean predicted prob (conf_j), bin size n_j
    bins = np.linspace(0.0, 1.0, n_bins+1)
    bin_idx = np.digitize(y_prob, bins) - 1            # 0..n_bins-1
    # clamp (edge case when y_prob == 1.0)
    bin_idx = np.clip(bin_idx, 0, n_bins-1)

    ece = 0.0
    mce = 0.0
    bin_stats = []
    for j in range(n_bins):
        mask = (bin_idx == j)
        n_j = mask.sum()
        if n_j == 0:
            # no samples in bin
            acc_j = np.nan
            conf_j = np.nan
            delta = 0.0
        else:
            acc_j = y_true[mask].mean()
            conf_j = y_prob[mask].mean()
            delta = abs(acc_j - conf_j)
            ece += (n_j / n) * delta
            mce = max(mce, delta)
        bin_stats.append({'n': int(n_j), 'acc': acc_j, 'conf': conf_j})

    # Hosmer-Lemeshow-like chi-square test (binned)
    # HL statistic: sum ((O_j - E_j)^2 / (Var_j)), Var_j = n_j * p_j * (1 - p_j)
    # where O_j = observed positives in bin, E_j = sum(prob) in bin = n_j * conf_j
    hl_stat = 0.0
    dof = 0
    for j in range(n_bins):
        n_j = bin_stats[j]['n']
        conf_j = bin_stats[j]['conf']
        if n_j == 0 or np.isnan(conf_j):
            continue
        O_j = int(round(bin_stats[j]['acc'] * n_j))
        E_j = conf_j * n_j
        Var_j = n_j * conf_j * (1.0 - conf_j)
        # numerical guard
        if Var_j <= 0:
            continue
        hl_stat += (O_j - E_j)**2 / Var_j
        dof += 1
    # degrees of freedom for HL often taken as (n_bins - 2) or (dof - 2) based on non-empty bins
    df = max(1, dof - 2)
    hl_pvalue = 1.0 - chi2.cdf(hl_stat, df)

    results = {
        'n_samples': n,
        'base_rate': float(p_ref),
        'brier': float(brier),
        'brier_ref': float(brier_ref),
        'brier_skill': float(brier_skill),
        'log_loss': float(logloss),
        'ece': float(ece),
        'mce': float(mce),
        'hl_stat': float(hl_stat),
        'hl_df': int(df),
        'hl_pvalue': float(hl_pvalue),
        'cal_curve': {'prob_pred': prob_pred_bins, 'prob_true': prob_true_bins, 'bin_stats': bin_stats, 'bins': bins}
    }

    # Optional bootstrap for CI (ECE and Brier)
    if bootstrap_ci:
        rng = np.random.RandomState(random_state)
        ece_boot = []
        brier_boot = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            y_t_s = y_true[idx]
            y_p_s = y_prob[idx]
            # compute brier
            brier_boot.append(brier_score_loss(y_t_s, y_p_s))
            # compute ece
            bin_idx_s = np.clip(np.digitize(y_p_s, bins) - 1, 0, n_bins-1)
            n_s = y_t_s.size
            ece_s = 0.0
            for j in range(n_bins):
                mask_s = (bin_idx_s == j)
                n_js = mask_s.sum()
                if n_js == 0:
                    continue
                acc_js = y_t_s[mask_s].mean()
                conf_js = y_p_s[mask_s].mean()
                ece_s += (n_js / n_s) * abs(acc_js - conf_js)
            ece_boot.append(ece_s)
        # percentile CI
        ece_boot = np.array(ece_boot)
        brier_boot = np.array(brier_boot)
        results['ece_ci'] = (float(np.percentile(ece_boot, 2.5)), float(np.percentile(ece_boot, 97.5)))
        results['brier_ci'] = (float(np.percentile(brier_boot, 2.5)), float(np.percentile(brier_boot, 97.5)))

    return results


def save_classification_report_png(y_true, y_pred, output_path="classification_report.png",
                                   title="Classification Report"):
    # Generate classification report as a dictionary
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Drop "support" for visualization clarity
    if "support" in report_df.columns:
        report_df = report_df.drop(columns=["support"])

    # Round numbers
    report_df = np.round(report_df, 2)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))

    # Show as image
    im = ax.imshow(report_df.values, cmap='Blues', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(report_df.columns)))
    ax.set_yticks(np.arange(len(report_df.index)))
    ax.set_xticklabels(report_df.columns, fontsize=9)
    ax.set_yticklabels(report_df.index, fontsize=9)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell with its value
    for i in range(len(report_df.index)):
        for j in range(len(report_df.columns)):
            value = report_df.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=8)

    # Title and colorbar
    ax.set_title(title, fontsize=12, pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logging.info(f"Classification report saved as: {output_path}")


# ---- Confusion matrix plotting ----
def plot_confusion_matrix(gt_labels, pred_labels, output_file):
    all_labels = sorted(set(gt_labels) | set(pred_labels))
    cm = confusion_matrix(gt_labels, pred_labels, labels=all_labels)

    cm_normalized = cm.astype(np.float64)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm_normalized, row_sums, out=np.zeros_like(cm_normalized), where=row_sums != 0)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=all_labels)
    disp.plot(cmap="Blues", values_format=".2f")
    # plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

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

    with open(os.path.join(training_data, 'test.yaml'), 'r') as f:
        test_files = yaml.safe_load(f)

    # Choose one random index
    # idx = random.randrange(len(test_files))
    idx = 2
    npy_file = os.path.join(training_data, test_files[idx])
    Chainkeep = np.load(npy_file)
    # idx_chain = random.randrange(Chainkeep.shape[0])
    idx_chain = 0
    logging.info("idx: {}, idx_chain: {}, file name: {},".format(idx, idx_chain, test_files[idx]))

    Chain = Chainkeep[idx_chain, :].copy()

    dg_test, dT_test, DensityModel_test = Chain2data(Chain, XnYnZn, Gkernelsp, Mkernelsp)

    SusModel_test = DensityModel_test / 50.0
    SusModel_test[DensityModel_test < 0.2] = 0

    ''' adding noise to data '''
    uncorr_noise_g = gravity_data_std * np.random.randn(len(dg_test))
    dg_test = dg_test + uncorr_noise_g

    uncorr_noise_T = rtp_data_std * np.random.randn(len(dT_test))
    dT_test = dT_test + uncorr_noise_T

    features_test = jointdata2features([dg_test, dT_test], D, I, k_list)

    DensityModel_pred = clf.predict(features_test)
    DensityModel_prob_arr = clf.predict_proba(features_test)
    DensityModel_prob = np.max(DensityModel_prob_arr, axis=1)

    DensityModel_pred = DensityModel_pred / 100.0
    SusModel_pred = DensityModel_pred / 50.0
    SusModel_pred[DensityModel_pred < 0.2] = 0

    dg_prediction = FM_sparse(DensityModel_pred, Gkernelsp)
    dT_prediction = FM_sparse(SusModel_pred, Mkernelsp)


    output_test = os.path.join(output_folder, test_files[idx], str(idx_chain))
    os.makedirs(output_test, exist_ok=True)

    np.save(os.path.join(output_test, 'dg_test.npy'), dg_test)
    np.save(os.path.join(output_test, 'dT_test.npy'), dT_test)
    np.save(os.path.join(output_test, 'DensityModel_test.npy'), DensityModel_test)
    np.save(os.path.join(output_test, 'SusModel_test.npy'), SusModel_test)


    np.save(os.path.join(output_test, 'dg_prediction.npy'), dg_prediction)
    np.save(os.path.join(output_test, 'dT_prediction.npy'), dT_prediction)
    np.save(os.path.join(output_test, 'DensityModel_pred.npy'), DensityModel_pred)
    np.save(os.path.join(output_test, 'SusModel_pred.npy'), SusModel_pred)
    np.save(os.path.join(output_test, 'DensityModel_prob.npy'), DensityModel_prob)

    labels_test_truth = (DensityModel_test * 100).astype('int')  ### mapping labels from float to int for rf classification
    labels_test_pred = (DensityModel_pred * 100).astype('int')  ### mapping labels from float to int for rf classification

    plot_confusion_matrix(labels_test_truth, labels_test_pred, os.path.join(output_test, 'confusion_matrix_test.png'))

    ####################################################################################
    save_classification_report_png(labels_test_truth, labels_test_pred, os.path.join(output_test, "classification_report.png"))

    ###################################################################################
    max_index = np.argmax(dT_test.reshape((Ndatapoints, Ndatapoints)))  # index in flattened array
    iy_data_valid, ix_data_valid = np.unravel_index(max_index, dT_test.reshape((Ndatapoints, Ndatapoints)).shape)

    # iy_data_valid = y_index
    ymodel_valid = ymodel - ymodel.min()
    ys_valid = ys - ymodel.min()
    iy_valid = np.argmin(abs(ymodel_valid - ys_valid[iy_data_valid]), 0)
    logging.info("ys_valid[iy_data_valid] {}".format(ys_valid[iy_data_valid]))
    logging.info("ymodel_valid[iy_valid] {}".format(ymodel_valid[iy_valid]))

    ####################################################

    PLOT.plot_valid_data_model_slice(dg_test, dg_prediction, dT_test, dT_prediction,
                                     Ndatapoints, xs - xmodel.min(), ys - ymodel.min(), DensityModel_test,
                                     DensityModel_pred, DensityModel_prob,
                                     xmodel - xmodel.min(), ymodel - ymodel.min(), zmodel, iy_data_valid, iy_valid,
                                     xticks=[10, 30, 50],
                                     output_path=os.path.join(output_test, 'test_model.pdf'))

    ###############################################################################################################

    PLOT.plot_datagrid(dg_test, dg_prediction, dT_test, dT_prediction, xs - xs.min(), ys - ys.min(),
                  Ndatapoints, iy_data_valid, xticks=[10, 30, 50], yticks=[0, 10, 20, 30, 40],
                  output_path=os.path.join(output_test, 'test_fit.pdf'))

    PLOT.plot_datainbox(dg_obs, npy_file, XnYnZn, Gkernelsp, Mkernelsp, Ndatapoints, xs,
                        output_path=os.path.join(output_test, 'datainbox.pdf'))

    ###############################################################################################################
    # plot the reliability diagram (a.k.a. probability calibration curve) for a Random Forest classifier
    # Define target labels
    target_labels = {'Salt': -13, 'Sediment': 0, 'Basement': 40}

    y_true_dict = {}
    y_prob_dict = {}

    for label_name, label_value in target_labels.items():
        # Build binary ground truth
        y_true_binary = np.zeros(len(labels_test_truth))
        y_true_binary[labels_test_truth == label_value] = 1
        y_true_dict[label_name] = y_true_binary

        # Get predicted probabilities for this class
        class_index = np.where(clf.classes_ == label_value)[0][0]
        y_prob_dict[label_name] = DensityModel_prob_arr[:, class_index]

    # Plot
    PLOT.plot_reliability_diagram_multi(
        y_true_dict,
        y_prob_dict,
        n_bins=10,
        output_path=os.path.join(output_test, 'reliability_multi.pdf')
    )

    res = calibration_metrics(y_true_dict['Salt'], y_prob_dict['Salt'], n_bins=10, bootstrap_ci=True, n_boot=500, random_state=0)
    pprint.pprint(res)

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