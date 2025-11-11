import os
import sys
import matplotlib.pyplot as plt
import matplotlib        as mpl
from matplotlib import rc
import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from Chain2data import Chain2data
from matplotlib.patches import Rectangle
# import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier

##############################################################3
# folder_name = '2022_12_24-05_48_53_PM'
#
# fpath_Arrays = os.getcwd()+'//'+folder_name+'//'+'Arrays'
# fpath_Figs = os.getcwd()+'//'+folder_name+'//'+'Figs'
# fpath_loaddesk = os.getcwd()+'/loaddesk'

##############################################################

def plot_density_4slices(field_density_prediction, xmodel, ymodel, zmodel,
                          basement_path, salt_path,
                          iy_list, ix_list,
                          vmin=-0.4, vmax=0.4,
                          cmap='seismic',
                          output_path=None):
    """
    Plot 4 slices of a 3D density model with basement and salt overlays.

    iy_list: list of 2 indices for E-Z slices
    ix_list: list of 2 indices for N-Z slices
    """
    # Load basement and salt
    basement = np.loadtxt(basement_path)
    salt = np.loadtxt(salt_path)
    X_hoj, Y_hoj, H_basement = basement[:,0], basement[:,1], -basement[:,2]  # positive depth
    X_salt, Y_salt, H_salt = salt[:,0], salt[:,1], -salt[:,2]  # positive depth

    CX, CY, CZ = len(xmodel), len(ymodel), len(zmodel)
    field_density_prediction_3D = field_density_prediction.reshape((CZ, CY, CX))

    fig, axs = plt.subplots(2,2, figsize=(12,10))
    axs = axs.flatten()
    slice_labels = ['(a)', '(b)', '(c)', '(d)']

    fsize = 15  # unified font size for everything

    # Tick values
    xticks = [650, 675, 700]  # km for E-Z
    zticks = [2.5, 5, 7.5, 10]
    yticks = [2690, 2710, 2730]  # km for N-Z

    xticklabels = list(map(str, xticks))
    yticklabels = list(map(str, yticks))
    zticklabels = list(map(str, zticks))

    # ------------------------
    # Subplots (a),(c) E-Z slices
    # ------------------------
    for i, iy in enumerate(iy_list):
        ax = axs[i*2]  # (a) = 0, (c) = 2
        im = ax.imshow(
            field_density_prediction_3D[:, iy, :],
            interpolation='none',
            vmin=vmin, vmax=vmax,
            extent=(xmodel[0]/1000, xmodel[-1]/1000, zmodel[-1]/1000, zmodel[0]/1000),
            aspect='auto',
            cmap=cmap
        )

        Y_PMD = ymodel[iy]
        nbrs = NearestNeighbors(n_neighbors=1).fit(Y_hoj.reshape(-1,1))
        _, iy_hoj = nbrs.kneighbors([[Y_PMD]])
        iy_hoj = iy_hoj[0,0]

        mask_basement = np.isclose(Y_hoj, Y_hoj[iy_hoj], atol=50)
        mask_salt = np.isclose(Y_salt, Y_hoj[iy_hoj], atol=50)

        # Plot lines (converted to km)
        ax.plot(X_salt[mask_salt]/1000, H_salt[mask_salt]/1000, 'k--', linewidth=1.8)
        ax.plot(X_hoj[mask_basement]/1000, H_basement[mask_basement]/1000, 'k-', linewidth=1.8)

        # Subplot label
        ax.text(0.03, 0.94, slice_labels[i*2], transform=ax.transAxes,
                fontsize=fsize, fontweight='normal', va='top', ha='left', color='k')

        ax.set_xlabel("Easting (km)", fontsize=fsize, fontweight="normal")
        ax.set_ylabel("Depth (km)", fontsize=fsize, fontweight="normal")
        ax.tick_params(axis='both', labelsize=fsize)
        ax.grid(True, linestyle=':', alpha=0.4)

    # ------------------------
    # Subplots (b),(d) N-Z slices
    # ------------------------
    for i, ix in enumerate(ix_list):
        ax = axs[i*2+1]  # (b)=1, (d)=3
        im = ax.imshow(
            field_density_prediction_3D[:, :, ix],
            interpolation='none',
            vmin=vmin, vmax=vmax,
            extent=(ymodel[-1]/1000, ymodel[0]/1000, zmodel[-1]/1000, zmodel[0]/1000),
            aspect='auto',
            cmap=cmap
        )

        X_PMD = xmodel[ix]
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_hoj.reshape(-1,1))
        _, ix_hoj = nbrs.kneighbors([[X_PMD]])
        ix_hoj = ix_hoj[0,0]

        mask_basement = np.isclose(X_hoj, X_hoj[ix_hoj], atol=50)
        mask_salt = np.isclose(X_salt, X_hoj[ix_hoj], atol=50)

        ax.plot(Y_salt[mask_salt]/1000, H_salt[mask_salt]/1000, 'k--', linewidth=1.8)
        ax.plot(Y_hoj[mask_basement]/1000, H_basement[mask_basement]/1000, 'k-', linewidth=1.8)

        ax.text(0.03, 0.94, slice_labels[i*2+1], transform=ax.transAxes,
                fontsize=fsize, fontweight='normal', va='top', ha='left', color='k')

        ax.set_xlabel("Northing (km)", fontsize=fsize, fontweight="normal")
        ax.set_ylabel("Depth (km)", fontsize=fsize, fontweight="normal")
        ax.tick_params(axis='both', labelsize=fsize)
        ax.grid(True, linestyle=':', alpha=0.4)

    # Tick layout & labels
    plt.setp(axs[0], xticks=xticks, yticks=zticks)
    plt.setp(axs[1], xticks=yticks, yticks=zticks)
    plt.setp(axs[2], xticks=xticks, yticks=zticks)
    plt.setp(axs[3], xticks=yticks, yticks=zticks)

    axs[0].set_xticklabels([]); axs[1].set_xticklabels([])
    axs[1].set_yticklabels([]); axs[3].set_yticklabels([])

    axs[0].set_xlabel("")
    axs[1].set_xlabel("")

    axs[1].set_ylabel("")
    axs[3].set_ylabel("")

    # Layout adjustments
    plt.subplots_adjust(wspace=0.05, hspace=0.1, left=0.08, right=0.87, top=0.92, bottom=0.08)

    # Adjust positions slightly
    dx0 = 0.025
    for ax in axs:
        pos = ax.get_position()
        pos.x0 -= dx0
        pos.x1 -= dx0
        ax.set_position(pos)

    # Add colorbar
    cbar_pos_density = fig.add_axes([0.89, 0.35, 0.02, 0.3])
    cbar_density = plt.colorbar(im, cax=cbar_pos_density, orientation='vertical',
                                ticklocation='right', ticks=[-0.4, -0.2, 0, 0.2, 0.4])
    cbar_density.set_label(r'$\Delta \rho \ (\mathrm{g.cm^{-3}})$', fontsize=fsize, weight='normal')
    cbar_density.ax.tick_params(labelsize=fsize)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")

    # plt.show()



def plot_valid_data_model_slice(dg, dg_new_prediction, dT, dT_new_prediction, Ndatapoints, xs, ys, DensityModel_validating,
                               density_prediction, uncertainty, xmodel, ymodel, zmodel, iy_data, iy, xticks, output_path):
    
    xticklabels = list(map(str, xticks))

    
    fig, axs = plt.subplots(5,1, sharex=False, sharey=False ,gridspec_kw={'wspace': 0.05, 'hspace': 0.05},figsize=(8, 5))
    
    fsize = 10

    
    plt.rc('font', weight='normal')
    plt.rc('xtick', labelsize=fsize)
    plt.rc('ytick', labelsize=fsize)

    pos0 = axs[0].get_position() # get the original position
    pos1 = axs[1].get_position() # get the original position 
    pos2 = axs[2].get_position() # get the original position
    pos3 = axs[3].get_position() # get the original position
    pos4 = axs[4].get_position() # get the original position


    dx0 = 0.025
    pos0.x0 -= dx0  # or use: pos00 = [pos00.x0 + 0.1, pos00.y0 ,  pos00.width, pos00.height] 
    pos1.x0 -= dx0
    pos2.x0 -= dx0
    pos3.x0 -= dx0
    pos4.x0 -= dx0

    dx1 = 0.025
    pos0.x1 -= dx1  # or use: pos00 = [pos00.x0 + 0.1, pos00.y0 ,  pos00.width, pos00.height] 
    pos1.x1 -= dx1
    pos2.x1 -= dx1
    pos3.x1 -= dx1
    pos4.x1 -= dx1

    axs[0].set_position(pos0) # set a new position
    axs[1].set_position(pos1) # set a new position
    axs[2].set_position(pos2) # set a new position
    axs[3].set_position(pos3) # set a new position
    axs[4].set_position(pos4) # set a new position

    
    WhiteBlueGreenYellowRed = np.loadtxt(os.getcwd()+'//'+'WhiteBlueGreenYellowRed.txt')
    WBGR = np.ones((WhiteBlueGreenYellowRed.shape[0],4))
    WBGR[:,:-1] = WhiteBlueGreenYellowRed.copy()
    WBGR = mpl.colors.ListedColormap(WBGR, name='WBGR', N=WBGR.shape[0])

    
    
    x1 = xmodel[0]/1000
    x2 = xmodel[-1]/1000
    y1 = ymodel[0]/1000
    y2 = ymodel[-1]/1000
    z1 = zmodel[0]/1000
    z2 = zmodel[-1]/1000
    
    dg_Grid = dg.reshape((Ndatapoints, Ndatapoints))
    dg_new_prediction_Grid = dg_new_prediction.reshape((Ndatapoints, Ndatapoints))
    dT_Grid = dT.reshape((Ndatapoints, Ndatapoints))
    dT_new_prediction_Grid = dT_new_prediction.reshape((Ndatapoints, Ndatapoints))
    
#     iy_data = 16
#     iy = np.argmin(abs(ymodel - ys[iy_data]),0)
    
#     print("plot_valid_data_model_slice: ", ys[iy_data])
#     print("plot_valid_data_model_slice: ", ymodel[iy])

#     logging.info("validation ys {}".format(ys[iy_data]))
#     logging.info("validation ymodel {}".format(ymodel[iy]))
    

    axs[0].plot(xs/1000, dg_Grid[iy_data,:], 'k.',linewidth=2)
    axs[0].plot(xs/1000, dg_new_prediction_Grid[iy_data,:], '-',linewidth=2, color='gray')
    
    axs[1].plot(xs/1000, dT_Grid[iy_data,:], 'k.',linewidth=2)
    axs[1].plot(xs/1000, dT_new_prediction_Grid[iy_data,:], '-',linewidth=2, color='gray')
    
    CX = len(xmodel)
    CY = len(ymodel)
    CZ = len(zmodel)


    
    xspace = np.linspace(x1, x2, num=CX)
    yspace = np.flip(np.linspace(y1, y2, num=CY))
    zspace = np.linspace(z1, z2, num=CZ)
    
    DensityModel_validating_3D = DensityModel_validating.reshape((CZ,CY,CX))
    density_prediction_3D = density_prediction.reshape((CZ,CY,CX))
    uncertainty_3D = uncertainty.reshape((CZ,CY,CX))
                                                                 


    im00 = axs[2].imshow(DensityModel_validating_3D[:,iy,:], interpolation='none',
                   vmin = -0.4, vmax = 0.4, extent=(x1,x2,z2,z1), aspect='auto', cmap='seismic')

    im01 = axs[3].imshow(density_prediction_3D[:,iy,:], interpolation='none',
                   vmin = -0.4, vmax = 0.4, extent=(x1,x2,z2,z1), aspect='auto', cmap='seismic')
    
    im02 = axs[4].imshow(uncertainty_3D[:,iy,:], interpolation='none',
                   vmin = 0.0, vmax = 1.0, extent=(x1,x2,z2,z1), aspect='auto', cmap=WBGR.reversed())    
    
    
    plt.setp(axs[0], xticks=xticks)
    plt.setp(axs[1], xticks=xticks)
    plt.setp(axs[2], xticks=xticks)
    plt.setp(axs[3], xticks=xticks)
    plt.setp(axs[4], xticks=xticks, xticklabels=xticklabels)
    
    axs[0].set_xticklabels([])
    axs[1].set_xticklabels([])
    axs[2].set_xticklabels([])
    axs[3].set_xticklabels([])
    

    plt.setp(axs[2], yticks=[2.5, 5, 7.5, 10], yticklabels=['2.5', '5', '7.5', '10'])
    plt.setp(axs[3], yticks=[2.5, 5, 7.5, 10], yticklabels=['2.5', '5', '7.5', '10'])
    plt.setp(axs[4], yticks=[2.5, 5, 7.5, 10], yticklabels=['2.5', '5', '7.5', '10'])

    
    axs[0].set_ylabel('Gravity \n (mGal)',fontweight="normal", fontsize = fsize)
    axs[1].set_ylabel('Magnetic \n (nT)',fontweight="normal", fontsize = fsize)
    axs[2].set_ylabel('Depth \n (km)',fontweight="normal", fontsize = fsize)
    axs[3].set_ylabel('Depth \n (km)',fontweight="normal", fontsize = fsize)
    axs[4].set_ylabel('Depth \n (km)',fontweight="normal", fontsize = fsize)
    axs[4].set_xlabel('Easting (km)',fontweight="normal", fontsize = fsize)
    
    axs[0].text(x1+1,0,'(a)', fontweight="normal", fontsize = fsize)
    axs[1].text(x1+1,50,'(b)', fontweight="normal", fontsize = fsize)
    axs[2].text(x1+1,2,'(c)', fontweight="normal", fontsize = fsize)
    axs[3].text(x1+1,2,'(d)', fontweight="normal", fontsize = fsize)
    axs[4].text(x1+1,2,'(e)', fontweight="normal", fontsize = fsize)
    
    axs[0].set_xlim([x1, x2])
    axs[1].set_xlim([x1, x2])
    axs[2].set_xlim([x1, x2])
    axs[3].set_xlim([x1, x2])
    axs[4].set_xlim([x1, x2])
    
    
    cbar_pos_density = fig.add_axes([0.89, 0.35, 0.02, 0.2])
    cbar_density = plt.colorbar(im00, ax=axs[2] ,shrink=0.3, cax = cbar_pos_density,
                        orientation='vertical', ticklocation = 'right', ticks = [-0.4, -0.2,0, 0.2, 0.4])
    cbar_density.ax.tick_params(labelsize=fsize)
    cbar_density.set_label(label = r'$\Delta \rho \ (\mathregular{g.cm^{-3}})$', weight='normal')

    cbar_pos_unc = fig.add_axes([0.89, 0.1, 0.02, 0.2]) 
    cbar_unc = plt.colorbar(im02, ax=axs[4] ,shrink=0.3,  cax = cbar_pos_unc,
                        orientation='vertical', ticklocation = 'right', ticks = [0,0.4,0.8])
    cbar_unc.ax.tick_params(labelsize=fsize)
    cbar_unc.set_label(label='probability', weight='normal')
    cbar_unc.ax.set_yticklabels(['0', '0.4', '0.8']) 



    fpath = os.getcwd()
    figname = 'Validating_model'
    fignum = ''
#     fig.savefig(fpath_Figs+'/'+figname+str(fignum)+'.pdf')
    fig.savefig(output_path)

    plt.close(fig)    # close the figure window

    
    
    
    ############################################################

def plot_field_data_model_slice(dg_obs, dg_fieldpred, dT_obs, dT_fieldpred, Ndatapoints,
                                xs, ys, field_density_prediction, uncertainty_field,
                                xmodel, ymodel, zmodel, iy_data, iy, xticks, output_path):
    
    xticklabels = list(map(str, xticks))
    
    WhiteBlueGreenYellowRed = np.loadtxt(os.getcwd()+'//'+'WhiteBlueGreenYellowRed.txt')
    WBGR = np.ones((WhiteBlueGreenYellowRed.shape[0],4))
    WBGR[:,:-1] = WhiteBlueGreenYellowRed.copy()
    WBGR = mpl.colors.ListedColormap(WBGR, name='WBGR', N=WBGR.shape[0])
    
    
    fig, axs = plt.subplots(4,1, sharex=False, sharey=False ,gridspec_kw={'wspace': 0.05, 'hspace': 0.05},figsize=(8, 5))    
    # fig = plt.figure()
    plt.rc('font', weight='normal')
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    
    fsize = 10

    
    plt.rc('font', weight='normal')
    plt.rc('xtick', labelsize=fsize)
    plt.rc('ytick', labelsize=fsize)

    pos0 = axs[0].get_position() # get the original position
    pos1 = axs[1].get_position() # get the original position 
    pos2 = axs[2].get_position() # get the original position
    pos3 = axs[3].get_position() # get the original position


    dx0 = 0.025
    pos0.x0 -= dx0  # or use: pos00 = [pos00.x0 + 0.1, pos00.y0 ,  pos00.width, pos00.height] 
    pos1.x0 -= dx0
    pos2.x0 -= dx0
    pos3.x0 -= dx0

    dx1 = 0.025
    pos0.x1 -= dx1  # or use: pos00 = [pos00.x0 + 0.1, pos00.y0 ,  pos00.width, pos00.height] 
    pos1.x1 -= dx1
    pos2.x1 -= dx1
    pos3.x1 -= dx1

    axs[0].set_position(pos0) # set a new position
    axs[1].set_position(pos1) # set a new position
    axs[2].set_position(pos2) # set a new position
    axs[3].set_position(pos3) # set a new position
    
    
#     iy_data = 8
#     iy = 44
    
#     print("plot_field_data_model_slice: ", ys[iy_data])
#     print("plot_field_data_model_slice: ", ymodel[iy])

#     logging.info("field ys {}".format(ys[iy_data]))
#     logging.info("field ymodel {}".format(ymodel[iy]))
    
    x1 = xmodel[0]/1000
    x2 = xmodel[-1]/1000
    y1 = ymodel[0]/1000
    y2 = ymodel[-1]/1000
    z1 = zmodel[0]/1000
    z2 = zmodel[-1]/1000
    
    CX = len(xmodel)
    CY = len(ymodel)
    CZ = len(zmodel)
    
    
    dg_obs_Grid = dg_obs.reshape((Ndatapoints, Ndatapoints))
    dg_fieldpred_Grid = dg_fieldpred.reshape((Ndatapoints, Ndatapoints))
    dT_obs_Grid = dT_obs.reshape((Ndatapoints, Ndatapoints))
    dT_fieldpred_Grid = dT_fieldpred.reshape((Ndatapoints, Ndatapoints))


    axs[0].plot(xs/1000, dg_obs_Grid[iy_data, :], 'k.',linewidth=2)
    axs[0].plot(xs/1000, dg_fieldpred_Grid[iy_data, :], '-',linewidth=2, color='gray')
    axs[0].set_ylim(-5, 9)

    axs[1].plot(xs/1000, dT_obs_Grid[iy_data, :], 'k.',linewidth=2)
    axs[1].plot(xs/1000, dT_fieldpred_Grid[iy_data, :], '-',linewidth=2, color='gray') 
    
    field_density_prediction_3D = field_density_prediction.reshape((CZ,CY,CX))
    uncertainty_field_3D = uncertainty_field.reshape((CZ,CY,CX))
    
    im02 = axs[2].imshow(field_density_prediction_3D[:,iy,:], interpolation='none',
           vmin = -0.4, vmax = 0.4, extent=(x1,x2,z2,z1), aspect='auto', cmap='seismic') # cmap='seismic'

    im03 = axs[3].imshow(uncertainty_field_3D[:,iy,:], interpolation='none',
                         extent=(x1,x2,z2,z1), aspect='auto', cmap=WBGR.reversed()) # cmap='seismic'
    
    plt.setp(axs[0], xticks=xticks)
    plt.setp(axs[1], xticks=xticks)
    plt.setp(axs[2], xticks=xticks)
    plt.setp(axs[3], xticks=xticks, xticklabels=xticklabels)
    
    axs[0].set_xticklabels([])
    axs[1].set_xticklabels([])
    axs[2].set_xticklabels([])
    
    plt.setp(axs[2], yticks=[2.5, 5, 7.5, 10], yticklabels=['2.5', '5', '7.5', '10'])
    plt.setp(axs[3], yticks=[2.5, 5, 7.5, 10], yticklabels=['2.5', '5', '7.5', '10'])

    
    axs[0].set_ylabel('Gravity \n (mGal)',fontweight="normal", fontsize = fsize)
    axs[1].set_ylabel('Magnetic \n (nT)',fontweight="normal", fontsize = fsize)
    axs[2].set_ylabel('Depth \n (km)',fontweight="normal", fontsize = fsize)
    axs[3].set_ylabel('Depth \n (km)',fontweight="normal", fontsize = fsize)
    axs[3].set_xlabel('Easting (km)',fontweight="normal", fontsize = fsize)
    
    axs[0].text(x1+1,0,'(a)', fontweight="normal", fontsize = fsize)
    axs[1].text(x1+1,50,'(b)', fontweight="normal", fontsize = fsize)
    axs[2].text(x1+1,2,'(c)', fontweight="normal", fontsize = fsize)
    axs[3].text(x1+1,2,'(d)', fontweight="normal", fontsize = fsize)
    
    axs[0].set_xlim([x1, x2])
    axs[1].set_xlim([x1, x2])
    axs[2].set_xlim([x1, x2])
    axs[3].set_xlim([x1, x2])
    
    cbar_pos_density = fig.add_axes([0.89, 0.35, 0.02, 0.2])
    cbar_density = plt.colorbar(im02, ax=axs[2] ,shrink=0.3, cax = cbar_pos_density,
                        orientation='vertical', ticklocation = 'right', ticks = [-0.4, -0.2,0, 0.2, 0.4])
    cbar_density.ax.tick_params(labelsize=fsize)
    cbar_density.set_label(label = r'$\Delta \rho \ (\mathregular{g.cm^{-3}})$', weight='normal')

    cbar_pos_unc = fig.add_axes([0.89, 0.1, 0.02, 0.2]) 
    cbar_unc = plt.colorbar(im03, ax=axs[3] ,shrink=0.3,  cax = cbar_pos_unc,
                        orientation='vertical', ticklocation = 'right', ticks = [0,0.4,0.8])
    cbar_unc.ax.tick_params(labelsize=fsize)
    cbar_unc.set_label(label='probability', weight='normal')
    cbar_unc.ax.set_yticklabels(['0', '0.4', '0.8']) 


    # plt.show()

    # fpath = os.getcwd()
    figname = 'Field_model'
    fignum = ''

#     fig.savefig(fpath_Figs+'/'+figname+str(fignum)+'.pdf')
    fig.savefig(output_path)
    plt.close(fig)    # close the figure window
    
    
#########################################################################################    

def plot_datagrid(dg_obs, dg_fieldpred, dT_obs, dT_fieldpred, xs, ys, Ndatapoints, iy_data, xticks, yticks, output_path):
    
    xticklabels = list(map(str, xticks))
    yticklabels = list(map(str, yticks))
    
    coastal = np.loadtxt(os.getcwd()+'//'+'coastal.txt')
    SirBaniYassaltisland = np.loadtxt(os.getcwd()+'//'+'SirBaniYassaltisland.txt')
    Ghasha_polygon = np.loadtxt(os.getcwd()+'//'+'Ghasha_polygon.txt')

    coastal = coastal/1000.
    SirBaniYassaltisland = SirBaniYassaltisland/1000.
    Ghasha_polygon = Ghasha_polygon/1000.
    
    
    fs = 10
    dgmin = np.amin(dg_obs)
    dgmax = np.amax(dg_obs)
    dTmin = np.amin(dT_obs)
    dTmax = np.amax(dT_obs)

    xs1 = np.min(xs)/1000
    xs2 = np.max(xs)/1000

    ys1 = np.min(ys)/1000
    ys2 = np.max(ys)/1000

    fig, axs = plt.subplots(2,3, sharex=False, sharey=False ,gridspec_kw={'wspace':0.0 , 'hspace': 0.0},figsize=(7, 4))
    
    plt.rc('font', weight='normal')
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    
    dx0 = -0.0
    dx1 = -0.0
    dy0 = +0.1
    dy1 = +0.1
    
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            
            posij = axs[i,j].get_position()
            posij.x0 += dx0
            posij.x1 += dx1
            posij.y0 += dy0
            posij.y1 += dy1
            axs[i,j].set_position(posij) # set a new position


            

    res_g = dg_obs - dg_fieldpred
    res_T = dg_obs - dg_fieldpred




    im00 = axs[0,0].imshow(dg_obs.reshape((Ndatapoints,Ndatapoints)),interpolation='none', vmin=dgmin,vmax=dgmax ,
                           extent=(xs1,xs2,ys1,ys2), aspect=abs((xs2-xs1)/(ys2-ys1)), cmap='jet')

    im01 = axs[0,1].imshow(dg_fieldpred.reshape((Ndatapoints,Ndatapoints)),interpolation='none', vmin=dgmin,vmax=dgmax ,
                           extent=(xs1,xs2,ys1,ys2), aspect=abs((xs2-xs1)/(ys2-ys1)), cmap='jet')

    im02 = axs[0,2].imshow(res_g.reshape((Ndatapoints,Ndatapoints))*0.1,interpolation='none', vmin=dgmin,vmax=dgmax ,
                           extent=(xs1,xs2,ys1,ys2), aspect=abs((xs2-xs1)/(ys2-ys1)), cmap='jet')
    
    
    im10 = axs[1,0].imshow(dT_obs.reshape((Ndatapoints,Ndatapoints)),interpolation='none', vmin=dTmin,vmax=dTmax ,
                           extent=(xs1,xs2,ys1,ys2), aspect=abs((xs2-xs1)/(ys2-ys1)), cmap='hsv')

    im11 = axs[1,1].imshow(dT_fieldpred.reshape((Ndatapoints,Ndatapoints)),interpolation='none', vmin=dTmin,vmax=dTmax ,
                           extent=(xs1,xs2,ys1,ys2), aspect=abs((xs2-xs1)/(ys2-ys1)), cmap='hsv')

    im12 = axs[1,2].imshow(res_T.reshape((Ndatapoints,Ndatapoints)),interpolation='none', vmin=dTmin,vmax=dTmax ,
                           extent=(xs1,xs2,ys1,ys2), aspect=abs((xs2-xs1)/(ys2-ys1)), cmap='hsv')
    

    
    plt.setp(axs[0,0], xticks=xticks)
    plt.setp(axs[0,1], xticks=xticks)
    plt.setp(axs[0,2], xticks=xticks)

    plt.setp(axs[1,0], xticks=xticks, xticklabels=xticklabels)
    plt.setp(axs[1,1], xticks=xticks, xticklabels=xticklabels)
    plt.setp(axs[1,2], xticks=xticks, xticklabels=xticklabels)


    plt.setp(axs[0,0], yticks=yticks, yticklabels=yticklabels)
    plt.setp(axs[0,1], yticks=yticks)
    plt.setp(axs[0,2], yticks=yticks)

    plt.setp(axs[1,0], yticks=yticks, yticklabels=yticklabels)
    plt.setp(axs[1,1], yticks=yticks)
    plt.setp(axs[1,2], yticks=yticks)
    
    
    axs[1,0].set_xlabel('Easting (km)',fontweight="normal", fontsize = fs)
    axs[1,1].set_xlabel('Easting (km)',fontweight="normal", fontsize = fs)
    axs[1,2].set_xlabel('Easting (km)',fontweight="normal", fontsize = fs)

    axs[0,0].set_ylabel('Northing (km)',fontweight="normal", fontsize = fs)
    axs[1,0].set_ylabel('Northing (km)',fontweight="normal", fontsize = fs)


    axs[0,0].set_xticklabels([])
    axs[0,1].set_xticklabels([])
    axs[0,2].set_xticklabels([])

    axs[0,1].set_yticklabels([])
    axs[0,2].set_yticklabels([])
    
    axs[1,1].set_yticklabels([])
    axs[1,2].set_yticklabels([])

    axs[0,0].text(xs1+1,ys2-5,'(a)', fontweight="normal", fontsize = fs)
    axs[0,1].text(xs1+1,ys2-5,'(b)', fontweight="normal", fontsize = fs)
    axs[0,2].text(xs1+1,ys2-5,'(c)', fontweight="normal", fontsize = fs)
    axs[1,0].text(xs1+1,ys2-5,'(d)', fontweight="normal", fontsize = fs)
    axs[1,1].text(xs1+1,ys2-5,'(e)', fontweight="normal", fontsize = fs)
    axs[1,2].text(xs1+1,ys2-5,'(f)', fontweight="normal", fontsize = fs)
    
        
    if xs.min() != 0.0: # field
        
#         iy_data = 8
        yh = ys[iy_data]/1000
        
#         print("plot_datagrid for field: ", ys[iy_data])
        
#         logging.info("field ys {}".format(ys[iy_data]))
        
        axs[0,0].axhline(y=yh, color='black', ls='--', lw=2)
        axs[0,1].axhline(y=yh, color='black', ls='--', lw=2)
        axs[0,2].axhline(y=yh, color='black', ls='--', lw=2)

        axs[1,0].axhline(y=yh, color='black', ls='--', lw=2)
        axs[1,1].axhline(y=yh, color='black', ls='--', lw=2)
        axs[1,2].axhline(y=yh, color='black', ls='--', lw=2)

    
    
        axs[0,0].plot(Ghasha_polygon[:,0],Ghasha_polygon[:,1],'k-',linewidth=2.5) 
        axs[0,1].plot(Ghasha_polygon[:,0],Ghasha_polygon[:,1],'k-',linewidth=2.5) 
        axs[1,0].plot(Ghasha_polygon[:,0],Ghasha_polygon[:,1],'k-',linewidth=2.5) 
        axs[1,1].plot(Ghasha_polygon[:,0],Ghasha_polygon[:,1],'k-',linewidth=2.5) 

        axs[0,0].plot(SirBaniYassaltisland[:,0],SirBaniYassaltisland[:,1],'w-',linewidth=2.5, color = 'gray') 
        axs[0,1].plot(SirBaniYassaltisland[:,0],SirBaniYassaltisland[:,1],'w-',linewidth=2.5, color = 'gray') 
        axs[1,0].plot(SirBaniYassaltisland[:,0],SirBaniYassaltisland[:,1],'w-',linewidth=2.5, color = 'gray') 
        axs[1,1].plot(SirBaniYassaltisland[:,0],SirBaniYassaltisland[:,1],'w-',linewidth=2.5, color = 'gray') 

        # axs[0,0].plot(coastal[:,0],coastal[:,1],'w.',markersize=2.5, color = 'gray')

        axs[0,0].text(683,2720,'Ghasha', fontweight="normal", fontsize = fs)
        axs[0,1].text(683,2720,'Ghasha', fontweight="normal", fontsize = fs)
        axs[1,0].text(683,2720,'Ghasha', fontweight="normal", fontsize = fs)
        axs[1,1].text(683,2720,'Ghasha', fontweight="normal", fontsize = fs)


        axs[0,0].text(650,2686,'Sir Bani Yas Is.', fontweight="normal", fontsize = fs-3)
        axs[0,1].text(650,2686,'Sir Bani Yas Is.', fontweight="normal", fontsize = fs-3)
        axs[1,0].text(650,2686,'Sir Bani Yas Is.', fontweight="normal", fontsize = fs-3)
        axs[1,1].text(650,2686,'Sir Bani Yas Is.', fontweight="normal", fontsize = fs-3)
        
        mag_colorbar_ticks = [0, 50, 100]

        
    else:
        
#         iy_data = 16
        yh = ys[iy_data]/1000
        
#         print("plot_datagrid for validation: ", ys[iy_data])

        
#         logging.info("validation ys {}".format(ys[iy_data]))
        
        axs[0,0].axhline(y=yh, color='black', ls='--', lw=2)
        axs[0,1].axhline(y=yh, color='black', ls='--', lw=2)
        axs[0,2].axhline(y=yh, color='black', ls='--', lw=2)

        axs[1,0].axhline(y=yh, color='black', ls='--', lw=2)
        axs[1,1].axhline(y=yh, color='black', ls='--', lw=2)
        axs[1,2].axhline(y=yh, color='black', ls='--', lw=2)
        
        mag_colorbar_ticks = [0, 50, 100, 150]


    cbar_pos_density = fig.add_axes([0.9, 0.65, 0.01, 0.25]) 
    cbar_density = plt.colorbar(im00, ax=axs[0,0] ,shrink=0.3, cax = cbar_pos_density,
                        orientation='vertical', ticklocation = 'right', ticks = [-2,0,2,4])
    cbar_density.ax.tick_params(labelsize=12)
    cbar_density.ax.set_title('mGal',fontsize = 8)
    
    cbar_pos_sus = fig.add_axes([0.9, 0.25, 0.01, 0.25]) 
    cbar_sus = plt.colorbar(im10, ax=axs[1,0] ,shrink=0.3, cax = cbar_pos_sus,
                        orientation='vertical', ticklocation = 'right', ticks = mag_colorbar_ticks)
    cbar_sus.ax.tick_params(labelsize=12)
    cbar_sus.ax.set_title('nT',fontsize = 8)


    # plt.show()
#     figname = 'Data_Fit'
    fig.savefig(output_path)
    plt.close(fig)    # close the figure window

###############################################################################################################
def plot_3dmodel(xmodel, ymodel, zmodel, field_density_prediction, uncertainty_field, PMD_g, iy, xticks, yticks, output_path):
    
    xticklabels = list(map(str, xticks))
    yticklabels = list(map(str, yticks))
    
    fs = 10
    
    xmodel = xmodel/1000
    ymodel = ymodel/1000
    zmodel = zmodel/1000

    
    Y, Z, X = np.meshgrid(ymodel, zmodel, xmodel)

    X_flat = X.flatten().copy()
    Y_flat = Y.flatten().copy()
    Z_flat = Z.flatten().copy()
    PMD_g_flat = PMD_g.flatten().copy()

    ### Thresholding
    ind_rf = np.where(field_density_prediction != 0 )[0]

    X_flat_ind_rf = X_flat[ind_rf]
    Y_flat_ind_rf = Y_flat[ind_rf]
    Z_flat_ind_rf = Z_flat[ind_rf]
    field_density_prediction_ind_rf = field_density_prediction[ind_rf]
    uncertainty_field_ind_rf = uncertainty_field[ind_rf]
    
    
    ind_pmd = np.where(abs(PMD_g_flat) >= 0.05)[0]

    X_flat_ind_pmd = X_flat[ind_pmd]
    Y_flat_ind_pmd = Y_flat[ind_pmd]
    Z_flat_ind_pmd = Z_flat[ind_pmd]
    PMD_g_flat_ind_pmd = PMD_g_flat[ind_pmd]


#     fig, axe = plt.subplots(1, 1, figsize=(6, 6))
#     axs = Axes3D(fig)

#     fig = plt.figure()
#     axs = fig.add_subplot(projection='3d')

    fig = plt.figure(figsize=(7, 6))
    
    
    #########################################################
    ### this is a fake subplot to just add colorbar and then will be invisible
    axs = fig.add_subplot(1, 1, 1, projection='3d')
    
    im00 = axs.scatter(X_flat_ind_rf, Y_flat_ind_rf, Z_flat_ind_rf, c=field_density_prediction_ind_rf,
                       marker='s', s=10,
                       vmin=-0.2, vmax=0.4,
                       alpha=0.4, cmap = 'jet')
    
    
#     fig.colorbar(im00, shrink=0.5, aspect=10)
    
    axs.axis('off')
    axs.set_visible(False)

    
    cbar_pos_density = fig.add_axes([0.3, 0.9, 0.4, 0.02])
    cbar_density = fig.colorbar(im00, ax=axs ,shrink=0.3, cax=cbar_pos_density,
                                orientation='horizontal', ticklocation = 'top', ticks = [-0.4, -0.2,0, 0.2, 0.4])
    cbar_density.ax.tick_params(labelsize=fs)
    cbar_density.set_label(label = r'$\Delta \rho \ (\mathregular{g.cm^{-3}})$', weight='normal')
    
    fig.delaxes(axs)

    
    
    ##################################################################################
    axs = fig.add_subplot(2, 2, 1, projection='3d')
    
#     pos = axs.get_position()
#     pos.y0 += -0.3
#     axs.set_position(pos) # set a new position
   

    im00 = axs.scatter(X_flat_ind_rf, Y_flat_ind_rf, Z_flat_ind_rf, c=field_density_prediction_ind_rf*uncertainty_field_ind_rf,
                       marker='s', s=10,
                       vmin=-0.2, vmax=0.4,
                       alpha=0.4, cmap = 'jet')
#     axs.invert_zaxis()
    # axe.invert_yaxis()
    # axe.invert_xaxis()
    
    axs.set_xlabel('Easting (km)',fontweight="normal", fontsize = fs)
    axs.set_ylabel('Northing (km)',fontweight="normal", fontsize = fs)
    axs.set_zlabel('Depth (km)',fontweight="normal", fontsize = fs)
    
    plt.setp(axs, xticks=xticks, xticklabels=xticklabels)
    plt.setp(axs, yticks=yticks, yticklabels=yticklabels)
    plt.setp(axs, zticks=[4, 6, 8, 10], zticklabels=['4', '6', '8', '10'])
    
    axs.set_xlim([630, 730])
    axs.set_ylim([2670, 2750])
    axs.set_zlim([3, 10])
    
    axs.invert_zaxis()


    axs.view_init(elev=10., azim=260.)
        
    # axs.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # axs.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # axs.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    axs.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    axs.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    axs.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))

    axs.text(631, 2730, 3.5,'(a)', fontweight="normal", fontsize = fs)



###############################################################################
    axs = fig.add_subplot(2, 2, 2, projection='3d')

    
    
    im00 = axs.scatter(X_flat_ind_pmd, Y_flat_ind_pmd, Z_flat_ind_pmd, c=PMD_g_flat_ind_pmd,
                       marker='s', s=10,
                       vmin=-0.2, vmax=0.4,
                       alpha=0.4, cmap = 'jet')
    
    # axe.invert_yaxis()
    # axe.invert_xaxis()
    
    axs.set_xlabel('Easting (km)',fontweight="normal", fontsize = fs)
#     axs.set_ylabel('Northing (km)',fontweight="normal", fontsize = fs)
#     axs.set_zlabel('Depth (km)',fontweight="normal", fontsize = fs)
    
    plt.setp(axs, xticks=xticks, xticklabels=xticklabels)
    plt.setp(axs, yticks=yticks, yticklabels=yticklabels)
    plt.setp(axs, zticks=[4, 6, 8, 10], zticklabels=['4', '6', '8', '10'])
    
        
    axs.set_xlim([630, 730])
    axs.set_ylim([2670, 2750])
    axs.set_zlim([3, 10])
    
    axs.invert_zaxis()

    
    axs.view_init(elev=10., azim=260.)
    
    axs.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    axs.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    axs.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    
    axs.text(631, 2730, 3.5,'(b)', fontweight="normal", fontsize = fs)
    
    
    ############################################################################################
    axs = fig.add_subplot(2, 2, 3)
    
#     iy = 44
#     print("plot_3dmodel for field: ", ymodel[iy])

#     logging.info("3d model ymodel {}".format(ymodel[iy]))
    
    field_density_prediction_3D = field_density_prediction.reshape((len(xmodel),len(ymodel),len(zmodel)))
    uncertainty_field_3D = uncertainty_field.reshape((len(xmodel),len(ymodel),len(zmodel)))
    
    X_profile = X[:,iy,:]
    Z_profile = Z[:,iy,:]
    field_density_prediction_3D_profile = field_density_prediction_3D[:,iy,:]
    uncertainty_field_3D_profile = uncertainty_field_3D[:,iy,:]

    X_profile_flat = X_profile.flatten()
    Z_profile_flat = Z_profile.flatten()
    field_density_prediction_3D_profile_flat = field_density_prediction_3D_profile.flatten()
    uncertainty_field_3D_profile_flat = uncertainty_field_3D_profile.flatten()

    ind_nonzero = np.where(field_density_prediction_3D_profile_flat != 0.0)[0]


    im00 = axs.scatter(X_profile_flat[ind_nonzero], Z_profile_flat[ind_nonzero],
                       c = field_density_prediction_3D_profile_flat[ind_nonzero]*uncertainty_field_3D_profile_flat[ind_nonzero],
                        vmin=-0.2, vmax=0.4, marker="s", cmap='jet')
    
    axs.set_xlim([640, 720])
    axs.set_ylim([3, 10])
    axs.invert_yaxis()
    
    axs.set_xlabel('Easting (km)',fontweight="normal", fontsize = fs)
    axs.set_ylabel('Depth (km)',fontweight="normal", fontsize = fs)
    
    axs.text(641, 3.5,'(c)', fontweight="normal", fontsize = fs)

    
    ############################################################################################
    axs = fig.add_subplot(2, 2, 4)
    
#     iy = 44
#     print("plot_3dmodel for field: ", ymodel[iy])

#     logging.info("3d model ymodel {}".format(ymodel[iy]))

    
#     X_profile = X[:,iy,:]
#     Z_profile = Z[:,iy,:]
    PMD_g_profile = PMD_g[:,iy,:]
    
    X_profile_flat = X_profile.flatten()
    Z_profile_flat = Z_profile.flatten()
    PMD_g_profile_flat = PMD_g_profile.flatten()
    
    ind_nonzero = np.where(abs(PMD_g_profile_flat) >= 0.05)[0]


    im00 = axs.scatter(X_profile_flat[ind_nonzero], Z_profile_flat[ind_nonzero],
                       c = PMD_g_profile_flat[ind_nonzero],
                        vmin=-0.2, vmax=0.4, marker="s", cmap='jet')
    
    axs.set_xlim([640, 720])
    axs.set_ylim([3, 10])
    axs.invert_yaxis()
    
    axs.set_xlabel('Easting (km)',fontweight="normal", fontsize = fs)
#     axs.set_ylabel('Depth (km)',fontweight="normal", fontsize = fs)

    axs.text(641, 3.5,'(d)', fontweight="normal", fontsize = fs)


    # plt.show()
    fig.savefig(output_path)
    plt.close(fig) 

    
    
####################################################################

def plot_datainbox(dg_obs, npyfile, XnYnZn, Kernel_Grv, Kernel_Mag, Ndatapoints, xs, output_path):
    Chainkeep = np.load(npyfile)
    dg_list = []
    iy_data = 8

    for ichain in range(Chainkeep.shape[0]):
        Chain = Chainkeep[ichain, :]     
        dg, dT, DensityModel = Chain2data(Chain, XnYnZn, Kernel_Grv, Kernel_Mag)
        dg_2d = dg.reshape((Ndatapoints,Ndatapoints))
        dg_list.append(dg_2d[iy_data,:])



    fig, axs = plt.subplots(1,1 , sharex=False, sharey=False ,gridspec_kw={'wspace':0.0 , 'hspace': 0.0},figsize=(7, 4))
    fs = 10
    plt.rc('font', weight='normal')
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)

    for i in range(len(dg_list)):
        axs.plot(xs/1000, dg_list[i], '-',linewidth=2, color='gray')

    # axs.plot(xs/1000, dg_list[i], 'k.-',linewidth=2, color='gray')
    dg_obs_Grid = dg_obs.reshape((Ndatapoints,Ndatapoints))
    axs.plot(xs/1000, dg_obs_Grid[iy_data, :], 'k.-',linewidth=2)


    # axs.add_patch(Rectangle((xs[0]/1000,dg_obs_Grid[iy_data, :].min()), (xs[-1]-xs[0])/1000,
    #                         dg_obs_Grid[iy_data, :].max()-dg_obs_Grid[iy_data, :].min(),
    #                         edgecolor='black',
    #                         facecolor='none',
    #                         lw=4))


    axs.set_ylabel('Gravity (mGal)',fontweight="normal", fontsize = fs)
    axs.set_xlabel('Easting (km)',fontweight="normal", fontsize = fs)


    # plt.show()
    
    fig.savefig(output_path)
    plt.close(fig) 

    
#################################################################################
def plot_prob_calib(mean_prob_list_rf0, true_prob_list_rf0, mean_prob_list_logreg, true_prob_list_logreg, output_path):
    
    fig, axs = plt.subplots(1,1 , sharex=False, sharey=False ,gridspec_kw={'wspace':0.0 , 'hspace': 0.0},figsize=(5, 5))
    fs = 10
    plt.rc('font', weight='normal')
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    
    
    axs.plot([0, 1], [0, 1], "k-", label="Perfectly calibrated")
    axs.set_ylabel("True probability")
    axs.set_xlabel("Predicted probability")

    axs.plot(mean_prob_list_rf0, true_prob_list_rf0, "s--", color = 'black' , label="%s" % ('Random Forest'))

    axs.plot(mean_prob_list_logreg, true_prob_list_logreg, "o-", color = 'black', label="%s" % ('Logistic Regression'))

#     axs.legend(loc="lower right")
#     plt.show()
    fig.savefig(output_path)

#     fig.savefig(os.path.join(fpath_Predictions_npy_Figs, 'prob_calib.pdf'))
    plt.close(fig) 

###################################################################################
def plot_reliability_diagram_multi(y_true_dict, y_prob_dict, n_bins=10, output_path=None):
    """
    Plot reliability diagrams for multiple classes on one figure.
    No legend; all lines black but different marker/linestyle per class.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    fontsize = 14

    # Define distinct styles per class
    styles = {
        "Salt": {"marker": "o", "linestyle": "-", "label": "Salt"},
        "Sediment": {"marker": "*", "linestyle": "--", "label": "Sediment"},
        "Basement": {"marker": "s", "linestyle": "-.", "label": "Basement"},
    }

    for label_name in y_true_dict.keys():
        y_true = y_true_dict[label_name]
        y_prob = y_prob_dict[label_name]
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        style = styles.get(label_name, {"marker": "o", "linestyle": "-"})
        ax.plot(
            prob_pred,
            prob_true,
            color='black',
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=5,
            linewidth=1.5,
        )

    # Reference perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)

    # Formatting
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Probability", fontsize=fontsize)
    ax.set_ylabel("Fraction of Positives", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Example usage:
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# y_prob = rf.predict_proba(X_test)[:, 1]
# plot_reliability_diagram(y_test, y_prob, n_bins=10, output_path="reliability_diagram.png")


#######################
# Ndatapoints = 32         # Number of total data in 1D array shape
#
# Gravity_Data = np.loadtxt('GRV_2D_Data.txt').astype('float32')
# Magnetic_Data = np.loadtxt('RTP_2D_Data.txt').astype('float32')
#
# xs = np.linspace(np.amin(Gravity_Data[:,0]),np.amax(Gravity_Data[:,0]),Ndatapoints)
# ys = np.flip(np.linspace(np.amin(Gravity_Data[:,1]),np.amax(Gravity_Data[:,1]), Ndatapoints),0)
#
# # ######################
#
# dg_obs = np.load(fpath_Arrays+'//'+'dg_obs.npy')
# dT_obs = np.load(fpath_Arrays+'//'+'dT_obs.npy')
#
# dg_fieldpred = np.load(fpath_Arrays+'//'+'dg_fieldpred.npy')
# dT_fieldpred = np.load(fpath_Arrays+'//'+'dT_fieldpred.npy')
#
# dg_validating = np.load(fpath_Arrays +'//'+'dg_validating.npy')
# dg_new_prediction = np.load(fpath_Arrays +'//'+'dg_new_prediction.npy')
#
# dT_validating = np.load(fpath_Arrays +'//'+'dT_validating.npy')
# dT_new_prediction = np.load(fpath_Arrays +'//'+'dT_new_prediction.npy')
#
# xmodel = np.load(fpath_Arrays +'//'+'xmodel.npy')
# ymodel = np.load(fpath_Arrays +'//'+'ymodel.npy')
# zmodel = np.load(fpath_Arrays +'//'+'zmodel.npy')
#
# field_density_prediction = np.load(fpath_Arrays+'//'+'field_density_prediction.npy')
# uncertainty_field = np.load(fpath_Arrays+'//'+'uncertainty_field.npy')
#
#
# DensityModel_validating = np.load(fpath_Arrays+'//'+'DensityModel_validating.npy')
# density_prediction = np.load(fpath_Arrays+'//'+'density_prediction.npy')
# uncertainty_validation = np.load(fpath_Arrays+'//'+'uncertainty_validation.npy')
#
# PMD_g = np.load(fpath_loaddesk+'//'+'PMD_g.npy')
#
#

# plot_datagrid(dg_obs, dg_fieldpred,  dT_obs, dT_fieldpred, xs, ys, Ndatapoints,
#               xticks=[650, 675, 700], yticks=[2690,2710,2730],
#               output_path=os.path.join(fpath_Figs, 'FieldData_Fit.pdf'))

# plot_datagrid(dg_validating, dg_new_prediction, dT_validating, dT_new_prediction, xs-xs.min(), ys-ys.min(), Ndatapoints,
#               xticks=[10, 30, 50], yticks=[0, 10, 20, 30, 40],
#               output_path=os.path.join(fpath_Figs, 'ValidData_Fit.pdf'))


# plot_field_data_model_slice(dg_obs, dg_fieldpred, dT_obs, dT_fieldpred, Ndatapoints,
#                             xs, field_density_prediction, uncertainty_field,
#                             xmodel, ymodel, zmodel, xticks=[650, 675, 700],
#                             output_path=os.path.join(fpath_Figs, 'Field_model.pdf'))


# plot_valid_data_model_slice(dg_validating, dg_new_prediction, dT_validating, dT_new_prediction, Ndatapoints,
#                             xs-xmodel.min(), ys-ymodel.min(), DensityModel_validating,
#                             density_prediction, uncertainty_validation,
#                             xmodel-xmodel.min(), ymodel-ymodel-min(), zmodel, xticks=[10, 30, 50],
#                             output_path=os.path.join(fpath_Figs, 'Validating_model.pdf'))



# plot_3dmodel(xmodel, ymodel, zmodel, field_density_prediction, uncertainty_field, PMD_g,
#              xticks=[650, 675, 700], yticks=[2690,2730],
#              output_path=os.path.join(fpath_Figs, '3Dmodel_one_view.pdf'))

###############################################################################################################

# Y, Z, X = np.meshgrid(ymodel, zmodel, xmodel)
# dx = abs(X[0,0,1]-X[0,0,0])
# dy = abs(Y[0,1,0]-Y[0,0,0])
# dz = abs(Z[1,0,0]-Z[0,0,0])
#
# # XYZ = np.column_stack((X.flatten(),Y.flatten(),Z.flatten())).astype('float32')
#
# x_min=np.min(X)-dx/2
# x_max=np.max(X)+dx/2
# y_min=np.min(Y)-dy/2
# y_max=np.max(Y)+dy/2
# z_min=np.min(Z)-dz/2
# z_max=np.max(Z)+dz/2
#
# Xn_3D = np.divide(X-x_min,x_max-x_min)
# Yn_3D = np.divide(Y-y_min,y_max-y_min)
# Zn_3D = np.divide(Z-z_min,z_max-z_min)
#
# Xn = Xn_3D.flatten()
# Yn = Yn_3D.flatten()
# Zn = Zn_3D.flatten()
# XnYnZn = np.column_stack((Xn,Yn,Zn)).astype('float32')
#
#
# Kernel_Grv_str = fpath_loaddesk +'//'+'Kernel_Grv'+'.npy'
# Kernel_Mag_str = fpath_loaddesk +'//'+'Kernel_Mag'+'.npy'
#
#
# Kernel_Grv = np.load(Kernel_Grv_str) # use this, if the latest result exists
# Kernel_Mag = np.load(Kernel_Mag_str) # use this, if the latest result exists
#
#
# inbox_threshold = 0.95
#
# fpath_training_dataset = os.getcwd()+'/training_dataset'
#
# inbox_threshold_folder = os.path.join(fpath_training_dataset, str(int(inbox_threshold*100)))
# npy_list = os.listdir(inbox_threshold_folder)
# npyfile = os.path.join(inbox_threshold_folder, npy_list[0])
#
# # plot_datainbox(npyfile, XnYnZn, Kernel_Grv, Kernel_Mag, Ndatapoints, xs,
# #                output_path=os.path.join(fpath_Figs, 'datainbox.pdf'))

    
