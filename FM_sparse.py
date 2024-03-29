"""
@author: emadg
"""
import numpy as np
#import scipy.sparse
import pywt

def FM_sparse(DensityModel,Gkernelsp):
    
    wname = 'db4'
    wv = pywt.Wavelet(wname)
    Nlevel = 2
#     thrg = 0.001
#     thrT = 0.7
    Wmode = 'periodization'
    
    CX = int(np.cbrt(DensityModel.size))
    CToT = CX*CX*CX

    DensityModel_3D = DensityModel.reshape((CX,CX,CX))
    Model_coeff  = pywt.wavedecn(DensityModel_3D, wv, mode=Wmode, level=Nlevel)
    Model_3D_coeff = pywt.coeffs_to_array(Model_coeff)[0]
    Model_3D_coeff_row = Model_3D_coeff.reshape((CToT,1))
    data_g_wave = Gkernelsp @ Model_3D_coeff_row

    data_g = np.squeeze(data_g_wave)
 
    return data_g
    
    