import logging

import numpy as np
import faiss
from FM_sparse import FM_sparse
import sys


def IsInBox(dg_1, dg_2):
    dg_1_max = dg_1.max()
    dg_1_min = dg_1.min()

    dg_2_max = dg_2.max()
    dg_2_min = dg_2.min()

    # tf = (abs(dg_1_max - dg_2_max) <= std) & (abs(dg_1_min - dg_2_min) <= std)

    dg_max_dif = abs(dg_1_max - dg_2_max)
    dg_min_dif = abs(dg_1_min - dg_2_min)

    return dg_max_dif, dg_min_dif

def IsInBox_minmax(dg_1, dg_2):
    dg_max = dg_1.max()
    dg_min = dg_1.min()
    tf = (dg_2 <= dg_max) & (dg_2 >= dg_min)
    tf_sum = np.sum(tf)
    return tf_sum/dg_2.size

def xyz2inbox(xc,yc,zc,rhoc, XnYnZn, Kernel_Grv, dg_obs):

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

    # isinbox_c = IsInBox_minmax(dg_obs, dg)

    dg_max_dif, dg_min_dif = IsInBox(dg_obs, dg)

    return dg_max_dif, dg_min_dif


def Model_Generator(XnYnZn, rho_sed, rho_salt, rho_base, Kmin, Kmax, Kernel_Grv, dg_obs, dg_field_std=0.2):
    
    Chain = np.zeros(1+Kmax*4).astype('float32')

    rho_salt_min = min(rho_salt)
    rho_salt_max = max(rho_salt)
    
    
    rho_base_min = min(rho_base)
    rho_base_max = max(rho_base)
    
    global_flag = True
    
    while global_flag:
    
        flag = True

        while flag:

            Nnode = np.random.randint(Kmin, Kmax)

            xc = np.random.rand(Nnode).astype('float32')
            yc = np.random.rand(Nnode).astype('float32')
            zc = np.random.rand(Nnode).astype('float32')


            X1c = 0.2
            X2c = 0.8
            Y1c = 0.2
            Y2c = 0.8
            Z1c = 0.5 #0.65
            Z2c = 0.85

            logic_sed = (xc<X1c) | (xc>X2c) | (yc<Y1c) | (yc>Y2c) | (zc<Z1c)
            logic_salt = (xc>=X1c) & (xc<=X2c) & (yc>=Y1c) & (yc<=Y2c) & (zc>=Z1c) & (zc<=Z2c)
            logic_base = (xc>=X1c) & (xc<=X2c) & (yc>=Y1c) & (yc<=Y2c) & (zc>Z2c)

            if np.any(logic_sed) and np.any(logic_salt) and np.any(logic_base):
                flag = False #break the while loop


        rhoc = logic_sed * np.random.choice(rho_sed, Nnode) + logic_salt * np.random.choice(rho_salt, Nnode) + logic_base * np.random.choice(rho_base, Nnode)


    #     r = np.random.rand(Nnode).astype('float32')

    #     rhoc = logic_sed * rho_sed + logic_salt*(rho_salt_min+r*(rho_salt_max-rho_salt_min))+(logic_base)*(rho_base_min+r*(rho_base_max-rho_base_min))



        dg_max_dif_c, dg_min_dif_c = xyz2inbox(xc,yc,zc,rhoc, XnYnZn, Kernel_Grv, dg_obs)
        
        for itry in range(10):
            
            if global_flag == False:
                break

            for inode in range(Nnode):

#                 if rank == 0:
#                     print(rank, inode, isinbox_c)
#                     sys.stdout.flush()


                if (dg_max_dif_c <= dg_field_std) & (dg_min_dif_c <= dg_field_std): # >= inbox_threshold:
                    global_flag = False
                    break

                flag = True

                while flag:



                    xp = xc.copy()
                    yp = yc.copy()
                    zp = zc.copy()
                    rhop = rhoc.copy()

                    xp[inode] = np.random.normal(loc=xp[inode], scale=0.2)
                    if xp[inode] < 0 or xp[inode] > 1: xp[inode] = xc[inode]

                    yp[inode] = np.random.normal(loc=yp[inode], scale=0.2)
                    if yp[inode] < 0 or yp[inode] > 1: yp[inode] = yc[inode]

                    zp[inode] = np.random.normal(loc=zp[inode], scale=0.2)
                    if zp[inode] < 0 or zp[inode] > 1: zp[inode] = zc[inode]


                    logic_sed = (xp<X1c) | (xp>X2c) | (yp<Y1c) | (yp>Y2c) | (zp<Z1c)
                    logic_salt = (xp>=X1c) & (xp<=X2c) & (yp>=Y1c) & (yp<=Y2c) & (zp>=Z1c) & (zp<=Z2c)
                    logic_base = (xp>=X1c) & (xp<=X2c) & (yp>=Y1c) & (yp<=Y2c) & (zp>Z2c)

                    if np.any(logic_sed) and np.any(logic_salt) and np.any(logic_base):
                        flag = False #break the while loop


                rhop = logic_sed * np.random.choice(rho_sed, Nnode) + logic_salt * np.random.choice(rho_salt, Nnode) + logic_base * np.random.choice(rho_base, Nnode)


            #     r = np.random.rand(Nnode).astype('float32')

            #     rhoc = logic_sed * rho_sed + logic_salt*(rho_salt_min+r*(rho_salt_max-rho_salt_min))+(logic_base)*(rho_base_min+r*(rho_base_max-rho_base_min))



                xp = xp.astype('float32')
                yp = yp.astype('float32')
                zp = zp.astype('float32')
                rhop = rhop.astype('float32')

                dg_max_dif_p, dg_min_dif_p = xyz2inbox(xp,yp,zp,rhop, XnYnZn, Kernel_Grv, dg_obs)

                if dg_max_dif_p <= dg_max_dif_c and dg_min_dif_p <= dg_min_dif_c:
                    xc = xp.copy()
                    yc = yp.copy()
                    zc = zp.copy()
                    rhoc = rhop.copy()
                    # isinbox_c = isinbox_p
                    # isinbox_final_c = isinbox_final_p
                    dg_max_dif_c = dg_max_dif_p
                    dg_min_dif_c = dg_min_dif_p


            
            
    Chain[0] = Nnode
    Chain[1:1+np.size(xc)*4] = np.concatenate((xc,yc,zc,rhoc))
    # logging.info('accepted model with dg_max_diff: {} and dg_min_diff: {}'.format(dg_diff_c[0], dg_diff_c[1]))
        
    return Chain







