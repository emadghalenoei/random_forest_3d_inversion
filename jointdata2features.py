import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys

def jointdata2features(datalist, D, I, k_list):
    
    
    II = I[:, :9] ### equal to k=1
#     print("IIII ",I.shape)
#     sys.stdout.flush()

    features = np.column_stack((datalist[0][II], datalist[1][II])).astype('float32')
#     print("1 ",features.shape)
#     sys.stdout.flush()

#     features = ind_model.copy()
    
    ind_row = np.arange(D.shape[0])
    
    for idata in range(len(datalist)):
        
#         if idata == 1: continue ### use just gravity not magnetic for now
        
        data = datalist[idata]
        
                
#         features = np.column_stack((features, data[II])).astype('float32')

#         features = data[II].copy()
 
        for k in k_list:

            if k == 1: continue ### k=1 is only one value and mean or std , etc. are the same

            II = I[:, :k]
            DD = D[:, :k]

            data_NN = data[II].astype('float32')        
            data_NN_mean = np.nanmean(data_NN, axis = 1).astype('float32')
            data_NN_std = np.nanstd(data_NN, axis = 1).astype('float32')
            data_NN_min = np.nanmin(data_NN, axis = 1).astype('float32')
            data_NN_max = np.nanmax(data_NN, axis = 1).astype('float32')

            data_NN_argmin = np.argmin(data_NN, axis = 1)
            data_NN_argmax = np.argmax(data_NN, axis = 1)
            D_datamin  = D[ind_row, data_NN_argmin].astype('float32')
            D_datamax  = D[ind_row, data_NN_argmax].astype('float32')

            features = np.column_stack((features, data_NN_mean, data_NN_std, data_NN_min, data_NN_max, D_datamin, D_datamax))
#             print("2 ", features.shape)
#             sys.stdout.flush()

        

    features = features.astype('float32')
    
    return features

    
    
#     dg_1NN = dg[I1].astype('float32')
#     dg_5NN = dg[I5].astype('float32')
#     dg_10NN = dg[I10].astype('float32')
#     dg_15NN = dg[I15].astype('float32')
#     dg_20NN = dg[I20].astype('float32')
#     dg_25NN = dg[I25].astype('float32')
    
    
    
# #     dg_1NN_mean = np.nanmean(dg_1NN, axis = 1).astype('float32')
#     dg_5NN_mean = np.nanmean(dg_5NN, axis = 1).astype('float32')
#     dg_10NN_mean = np.nanmean(dg_10NN, axis = 1).astype('float32')
#     dg_15NN_mean = np.nanmean(dg_15NN, axis = 1).astype('float32')
#     dg_20NN_mean = np.nanmean(dg_20NN, axis = 1).astype('float32')
#     dg_25NN_mean = np.nanmean(dg_25NN, axis = 1).astype('float32')

# #     dg_1NN_std = np.nanstd(dg_1NN, axis = 1).astype('float32')
#     dg_5NN_std = np.nanstd(dg_5NN, axis = 1).astype('float32')
#     dg_10NN_std = np.nanstd(dg_10NN, axis = 1).astype('float32')
#     dg_15NN_std = np.nanstd(dg_15NN, axis = 1).astype('float32')
#     dg_20NN_std = np.nanstd(dg_20NN, axis = 1).astype('float32')
#     dg_25NN_std = np.nanstd(dg_25NN, axis = 1).astype('float32')

# #     dg_1NN_min = np.nanmin(dg_1NN, axis = 1).astype('float32')
#     dg_5NN_min = np.nanmin(dg_5NN, axis = 1).astype('float32')
#     dg_10NN_min = np.nanmin(dg_10NN, axis = 1).astype('float32')
#     dg_15NN_min = np.nanmin(dg_15NN, axis = 1).astype('float32')
#     dg_20NN_min = np.nanmin(dg_20NN, axis = 1).astype('float32')
#     dg_25NN_min = np.nanmin(dg_25NN, axis = 1).astype('float32')

# #     dg_1NN_max = np.nanmax(dg_1NN, axis = 1).astype('float32')
#     dg_5NN_max = np.nanmax(dg_5NN, axis = 1).astype('float32')
#     dg_10NN_max = np.nanmax(dg_10NN, axis = 1).astype('float32')
#     dg_15NN_max = np.nanmax(dg_15NN, axis = 1).astype('float32')
#     dg_20NN_max = np.nanmax(dg_20NN, axis = 1).astype('float32')
#     dg_25NN_max = np.nanmax(dg_25NN, axis = 1).astype('float32')
    
# #     dg_5NN_grd = np.gradient(dg_5NN, axis = 1).astype('float32')

    
    
    
# #     dg_5NN_sum = np.nansum(dg_5NN, axis=1)    ###  sum
# #     dg_10NN_sum = np.nansum(dg_10NN, axis=1)   ###  sum
# #     dg_15NN_sum = np.nansum(dg_15NN, axis=1)   ###  sum
# #     dg_20NN_sum = np.nansum(dg_20NN, axis=1)   ###  sum
# #     dg_25NN_sum = np.nansum(dg_25NN, axis=1)   ###  sum


    
# #     dg_5NN_median = np.nanmedian(dg_5NN, axis = 1).astype('float32')
# #     dg_10NN_median = np.nanmedian(dg_10NN, axis = 1).astype('float32')
# #     dg_15NN_median = np.nanmedian(dg_15NN, axis = 1).astype('float32')
# #     dg_20NN_median = np.nanmedian(dg_20NN, axis = 1).astype('float32')
# #     dg_25NN_median = np.nanmedian(dg_25NN, axis = 1).astype('float32')

# #     dg_5NN_wsum = np.nansum(dg_5NN/D5, axis=1)    ### weighted sum
# #     dg_10NN_wsum = np.nansum(dg_10NN/D10, axis=1)   ### weighted sum
# #     dg_15NN_wsum = np.nansum(dg_15NN/D15, axis=1)   ### weighted sum
# #     dg_20NN_wsum = np.nansum(dg_20NN/D20, axis=1)   ### weighted sum
# #     dg_25NN_wsum = np.nansum(dg_25NN/D25, axis=1)   ### weighted sum

   
    
#     dg_5NN_ptp = np.ptp(dg_5NN, axis = 1).astype('float32') ### Range of values (maximum - minimum) along an axis
#     dg_10NN_ptp = np.ptp(dg_10NN, axis = 1).astype('float32')
#     dg_15NN_ptp = np.ptp(dg_15NN, axis = 1).astype('float32')
#     dg_20NN_ptp = np.ptp(dg_20NN, axis = 1).astype('float32')
#     dg_25NN_ptp = np.ptp(dg_25NN, axis = 1).astype('float32')

    
# #     dg_5NN_avg = np.average(dg_5NN, weights=1/D5, axis = 1).astype('float32')
# #     dg_10NN_avg = np.average(dg_10NN, weights=1/D10, axis = 1).astype('float32')
    
# #     dg_5NN_q25 = np.nanquantile(dg_5NN, 0.25, axis=1).astype('float32')
# #     dg_10NN_q25 = np.nanquantile(dg_10NN, 0.25, axis=1).astype('float32')
    
# #     dg_5NN_q50 = np.nanquantile(dg_5NN, 0.50, axis=1).astype('float32')    ### equal to median
# #     dg_10NN_q50 = np.nanquantile(dg_10NN, 0.50, axis=1).astype('float32')  ### equal to median
    
# #     dg_5NN_q75 = np.nanquantile(dg_5NN, 0.75, axis=1).astype('float32')
# #     dg_10NN_q75 = np.nanquantile(dg_10NN, 0.75, axis=1).astype('float32')

    
# #     dg_1NN_argmin = np.argmin(dg_1NN, axis = 1)
#     dg_5NN_argmin = np.argmin(dg_5NN, axis = 1)
#     dg_10NN_argmin = np.argmin(dg_10NN, axis = 1)
#     dg_15NN_argmin = np.argmin(dg_15NN, axis = 1)
#     dg_20NN_argmin = np.argmin(dg_20NN, axis = 1)
#     dg_25NN_argmin = np.argmin(dg_25NN, axis = 1)
    
# #     dg_1NN_argmax = np.argmax(dg_1NN, axis = 1)
#     dg_5NN_argmax = np.argmax(dg_5NN, axis = 1)
#     dg_10NN_argmax = np.argmax(dg_10NN, axis = 1)
#     dg_15NN_argmax = np.argmax(dg_15NN, axis = 1)
#     dg_20NN_argmax = np.argmax(dg_20NN, axis = 1)
#     dg_25NN_argmax = np.argmax(dg_25NN, axis = 1)
    
#     ind_row = np.arange(D1.shape[0])

# #     D1_dgmin  = D1[ind_row, dg_1NN_argmin].astype('float32')
#     D5_dgmin  = D5[ind_row, dg_5NN_argmin].astype('float32')
#     D10_dgmin = D10[ind_row, dg_10NN_argmin].astype('float32')
#     D15_dgmin = D15[ind_row, dg_15NN_argmin].astype('float32')
#     D20_dgmin = D20[ind_row, dg_20NN_argmin].astype('float32')
#     D25_dgmin = D25[ind_row, dg_25NN_argmin].astype('float32')
    
# #     D1_dgmax = D1[ind_row, dg_1NN_argmax].astype('float32')
#     D5_dgmax  = D5[ind_row, dg_5NN_argmax].astype('float32')
#     D10_dgmax = D10[ind_row, dg_10NN_argmax].astype('float32')
#     D15_dgmax = D15[ind_row, dg_15NN_argmax].astype('float32')
#     D20_dgmax = D20[ind_row, dg_20NN_argmax].astype('float32')
#     D25_dgmax = D25[ind_row, dg_25NN_argmax].astype('float32')
    
    
# #     features = np.column_stack((ind_model, dg_5NN,
# #                                 dg_5NN_mean, dg_10NN_mean, dg_15NN_mean,
# #                                 dg_5NN_std, dg_10NN_std, dg_15NN_std,
# #                                 dg_5NN_min, dg_10NN_min, dg_15NN_min,
# #                                 dg_5NN_max, dg_10NN_max, dg_15NN_max,
# #                                 D5_dgmin, D10_dgmin, D15_dgmin,
# #                                 D5_dgmax, D10_dgmax, D15_dgmax,
# #                                 dg_5NN_ptp, dg_10NN_ptp, dg_15NN_ptp))
    

#     features = np.column_stack((ind_model, dg_5NN,
#                                 dg_5NN_mean, dg_10NN_mean, dg_15NN_mean, dg_20NN_mean, dg_25NN_mean,
#                                 dg_5NN_std, dg_10NN_std, dg_15NN_std, dg_20NN_std, dg_25NN_std,
#                                 dg_5NN_min, dg_10NN_min, dg_15NN_min, dg_20NN_min, dg_25NN_min,
#                                 dg_5NN_max, dg_10NN_max, dg_15NN_max, dg_20NN_max, dg_25NN_max,
#                                 D5_dgmin, D10_dgmin, D15_dgmin, D20_dgmin, D25_dgmin,
#                                 D5_dgmax, D10_dgmax, D15_dgmax, D20_dgmax, D25_dgmax,
#                                 dg_5NN_ptp, dg_10NN_ptp, dg_15NN_ptp, dg_20NN_ptp, dg_25NN_ptp))
    
    
# #     features = MinMaxScaler().fit_transform(features)
