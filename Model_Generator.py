import numpy as np
import faiss
from FM_sparse import FM_sparse


def compute_dg_ratio(dg_ref, dg_new):
    """
    Compute the fraction of points in dg_new that lie within the min-max range of dg_ref.
    Returns a float in [0, 1], where higher is better.
    """
    dg_min, dg_max = dg_ref.min(), dg_ref.max()
    inside = (dg_new >= dg_min) & (dg_new <= dg_max)
    return np.sum(inside) / dg_new.size



def compute_dg_diff(dg_ref, dg_new):
    """Compute max and min differences between two gravity fields."""
    return abs(dg_ref.max() - dg_new.max()), abs(dg_ref.min() - dg_new.min())


def compute_density_model(x, y, z, rho, grid_points, kernel_gravity):
    """Compute density model and gravity response for given node configuration."""
    train_points = np.column_stack((x, y, z)).astype('float32')

    index = faiss.IndexFlatL2(3)
    index.add(train_points)

    _, indices = index.search(grid_points, 1)
    density_model = rho[indices[:, 0]].astype('float32')

    return FM_sparse(density_model, kernel_gravity)


def compute_dg(x, y, z, rho, grid_points, kernel_gravity, dg_obs):
    """Compute difference in dg range between observed and generated data."""
    dg_new = compute_density_model(x, y, z, rho, grid_points, kernel_gravity)
    return dg_new


def Model_Generator(XnYnZn, rho_sed, rho_salt, rho_base,
                    Kmin, Kmax, Kernel_Grv, dg_obs,
                    dg_field_std=0.2):
    """
    Generate a model whose simulated gravity field matches the observed field
    within the specified standard deviation threshold.
    """
    Chain = np.zeros(1 + Kmax * 4, dtype='float32')

    # Precompute bounds
    rho_salt_min, rho_salt_max = min(rho_salt), max(rho_salt)
    rho_base_min, rho_base_max = min(rho_base), max(rho_base)

    # Spatial bounds
    X1c, X2c = 0.2, 0.8
    Y1c, Y2c = 0.2, 0.8
    Z1c, Z2c = 0.5, 0.85

    while True:  # global loop: keep generating until a valid model is found
        # --- Step 1: Initialize random nodes until valid layering is achieved ---
        while True:
            Nnode = np.random.randint(Kmin, Kmax)
            xc, yc, zc = np.random.rand(3, Nnode).astype('float32')

            logic_sed = (xc < X1c) | (xc > X2c) | (yc < Y1c) | (yc > Y2c) | (zc < Z1c)
            logic_salt = (X1c <= xc) & (xc <= X2c) & (Y1c <= yc) & (yc <= Y2c) & (Z1c <= zc) & (zc <= Z2c)
            logic_base = (X1c <= xc) & (xc <= X2c) & (Y1c <= yc) & (yc <= Y2c) & (zc > Z2c)

            if np.any(logic_sed) and np.any(logic_salt) and np.any(logic_base):
                break

        rhoc = (logic_sed * np.random.choice(rho_sed, Nnode) +
                logic_salt * np.random.choice(rho_salt, Nnode) +
                logic_base * np.random.choice(rho_base, Nnode))

        dg_new = compute_dg(xc, yc, zc, rhoc, XnYnZn, Kernel_Grv, dg_obs)

        dg_ratio = compute_dg_ratio(dg_obs, dg_new)
        dg_max_diff, dg_min_diff = compute_dg_diff(dg_obs, dg_new)

        # --- Step 2: Iteratively perturb and accept better configurations ---
        for _ in range(10):
            for inode in range(Nnode):
                # Stop early if model already fits threshold
                if (dg_max_diff <= dg_field_std) and (dg_min_diff <= dg_field_std):
                    Chain[0] = Nnode
                    Chain[1:1 + Nnode * 4] = np.concatenate((xc, yc, zc, rhoc))
                    return Chain

                # Perturb node
                xp, yp, zp = xc.copy(), yc.copy(), zc.copy()

                xp[inode] = np.clip(np.random.normal(xp[inode], 0.2), 0, 1)
                yp[inode] = np.clip(np.random.normal(yp[inode], 0.2), 0, 1)
                zp[inode] = np.clip(np.random.normal(zp[inode], 0.2), 0, 1)

                logic_sed = (xp < X1c) | (xp > X2c) | (yp < Y1c) | (yp > Y2c) | (zp < Z1c)
                logic_salt = (X1c <= xp) & (xp <= X2c) & (Y1c <= yp) & (yp <= Y2c) & (Z1c <= zp) & (zp <= Z2c)
                logic_base = (X1c <= xp) & (xp <= X2c) & (Y1c <= yp) & (yp <= Y2c) & (zp > Z2c)

                # skip if invalid layering
                if not (np.any(logic_sed) and np.any(logic_salt) and np.any(logic_base)):
                    continue

                rhop = (logic_sed * np.random.choice(rho_sed, Nnode) +
                        logic_salt * np.random.choice(rho_salt, Nnode) +
                        logic_base * np.random.choice(rho_base, Nnode))

                dg_new_p = compute_dg(
                    xp, yp, zp, rhop, XnYnZn, Kernel_Grv, dg_obs
                )

                dg_ratio_p = compute_dg_ratio(dg_obs, dg_new_p)
                dg_max_diff_p, dg_min_diff_p = compute_dg_diff(dg_obs, dg_new_p)


                # Accept if both differences improve
                # if (dg_max_diff_p <= dg_max_diff) and (dg_min_diff_p <= dg_min_diff):
                if dg_ratio_p >= dg_ratio:
                    xc, yc, zc, rhoc = xp, yp, zp, rhop
                    dg_max_diff, dg_min_diff = dg_max_diff_p, dg_min_diff_p
                    dg_ratio = dg_ratio_p

