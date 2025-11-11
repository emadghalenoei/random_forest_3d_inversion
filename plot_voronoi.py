import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os

# -----------------------------
# Parameters
# -----------------------------
np.random.seed(42)
GRID_SIZE = 64
PLOT_GRID = 32  # downsampled for plotting
N_SEEDS = 50

# -----------------------------
# Step 1 — Random Voronoi nodes
# -----------------------------
seeds = np.random.rand(N_SEEDS, 3)

# -----------------------------
# Step 2 — Create 3D voxel grid
# -----------------------------
lin = np.linspace(0, 1, GRID_SIZE)
xg, yg, zg = np.meshgrid(lin, lin, lin, indexing='ij')
pts = np.column_stack((xg.ravel(), yg.ravel(), zg.ravel()))

# Assign each voxel to nearest Voronoi node
tree = cKDTree(seeds)
_, nearest_idx = tree.query(pts, k=1)
nearest_idx_grid = nearest_idx.reshape((GRID_SIZE, GRID_SIZE, GRID_SIZE))


# -----------------------------
# Step 3 — Random planes (two per axis)
# -----------------------------
def random_two_planes():
    a, b = np.random.rand(2) * 0.8 + 0.1
    return np.sort([a, b])


x_planes = random_two_planes()
y_planes = random_two_planes()
z_planes = random_two_planes()


# -----------------------------
# Step 4 — Classify Voronoi points based on planes
# -----------------------------
# Define middle sub-volume as: index 1 along x, y, z
def classify_point(pt, x_planes, y_planes, z_planes):
    # Determine index along each axis
    def axis_index(coord, planes):
        if coord < planes[0]:
            return 0
        elif coord < planes[1]:
            return 1
        else:
            return 2

    xi = axis_index(pt[0], x_planes)
    yi = axis_index(pt[1], y_planes)
    zi = axis_index(pt[2], z_planes)

    # Middle sub-volume
    if xi == 1 and yi == 1 and zi == 1:
        return 1
    # Lower-middle sub-volume: same x & y middle, z lower
    elif xi == 1 and yi == 1 and zi == 0:
        return 2
    else:
        return 0


vor_labels = np.array([classify_point(pt, x_planes, y_planes, z_planes) for pt in seeds])

# -----------------------------
# Step 5 — Transfer labels to voxels
# -----------------------------
voxel_labels = vor_labels[nearest_idx_grid]

# -----------------------------
# Step 6 — Downsample for plotting
# -----------------------------
inds = np.linspace(0, GRID_SIZE - 1, PLOT_GRID, dtype=int)
xg_ds = xg[inds][:, inds][:, :, inds]
yg_ds = yg[inds][:, inds][:, :, inds]
zg_ds = zg[inds][:, inds][:, :, inds]
voxel_labels_ds = voxel_labels[inds][:, inds][:, :, inds]

X = xg_ds.ravel()
Y = yg_ds.ravel()
Z = zg_ds.ravel()
labels_flat = voxel_labels_ds.ravel()

# Only plot voxels with label 1 or 2
mask = labels_flat != 0
X_plot = X[mask]
Y_plot = Y[mask]
Z_plot = Z[mask]
labels_plot = labels_flat[mask]

# Map labels to colors
color_map = {1: [0,0,1,0.75], 2: [1,0,0,0.75]}
colors = np.array([color_map[l] for l in labels_plot])


# -----------------------------
# Step 7 — Plotting
# -----------------------------
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1, 1, 1))

# Plot voxels
ax.scatter(X_plot, Y_plot, Z_plot, c=colors, marker='s', s=30, depthshade=True)





# Draw planes
def draw_plane(ax, axis, pos, color='cyan', alpha=0.4):
    ys = np.linspace(0, 1, 5)
    zs = np.linspace(0, 1, 5)
    xs = np.linspace(0, 1, 5)
    if axis == 'x':
        Y, Z = np.meshgrid(ys, zs)
        X = np.ones_like(Y) * pos
        ax.plot_surface(X, Y, Z, color=color, edgecolor='none', linewidth=0.8, alpha=alpha)
    elif axis == 'y':
        X, Z = np.meshgrid(xs, zs)
        Y = np.ones_like(X) * pos
        ax.plot_surface(X, Y, Z, color=color, edgecolor='none', linewidth=0.8, alpha=alpha)
    elif axis == 'z':
        X, Y = np.meshgrid(xs, ys)
        Z = np.ones_like(X) * pos
        ax.plot_surface(X, Y, Z, color=color, edgecolor='none', linewidth=0.8, alpha=alpha)


# Use bright colors for planes
plane_colors = ['grey', 'grey', 'grey'] #['red', 'green', 'blue']
draw_plane(ax, 'x', x_planes[0], plane_colors[0])
draw_plane(ax, 'x', x_planes[1], plane_colors[0])
draw_plane(ax, 'y', y_planes[0], plane_colors[1])
draw_plane(ax, 'y', y_planes[1], plane_colors[1])
draw_plane(ax, 'z', z_planes[0], plane_colors[2])
draw_plane(ax, 'z', z_planes[1], plane_colors[2])

# Plot Voronoi nodes
ax.scatter(seeds[:, 0], seeds[:, 1], seeds[:, 2], c='k', s=50)

# Axes and view
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=20, azim=30)
ax.grid(False)

# -----------------------------
# Step 8 — Save
# -----------------------------
out_dir = 'voronoi_classified_voxels'
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, 'voronoi_classified.png')
pdf_path = os.path.join(out_dir, 'voronoi_classified.pdf')
plt.savefig(png_path, dpi=200, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')
plt.close(fig)

print("Saved outputs:")
print(png_path)
print(pdf_path)
