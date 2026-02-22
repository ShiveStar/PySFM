import glob
import numpy as np
import pyvista as pv

# ================= CONFIG =================
GLOB_PATTERN = "../results/log_cabin_0.3_8000*/*.txt"
DELIMITER = ";"
MAX_POINTS = 100_000
CLIP_OUTLIERS = True
CLIP_LO = 5.0
CLIP_HI = 99.0

POINT_SIZE = 1.0
SEED = 0

WINDOW_SIZE = (1400, 900)
TITLE = "Combined PySFM Point Cloud"

N_XLABELS = 20
N_YLABELS = 20
N_ZLABELS = 12

# ----------------- ROTATION -----------------
ROTATE_ENABLE = True
ROTATE_DEGREES_X = 0.0
ROTATE_DEGREES_Y = 98.0
ROTATE_DEGREES_Z = 0.0
RECENTER_AFTER_ROTATION = True

# ----------------- SCALING ------------------
SCALE_ENABLE = False
SCALE_FACTOR = 100.0
# --------------------------------------------

SHIFT_TO_ZERO_MIN = True
BOUNDS_PAD_FRACTION = 0.2
# ===========================================


def load_points(path):
    P = np.loadtxt(path, delimiter=DELIMITER)
    if P.ndim == 1:
        P = P.reshape(1, -1)

    if P.shape[1] == 4:
        w = P[:, 3:4]
        P = P[:, :3] / w

    if P.shape[1] != 3:
        raise ValueError(f"{path}: expected 3 or 4 columns, got {P.shape}")

    return P.astype(np.float32)


def clean_points(P):
    return P[np.isfinite(P).all(axis=1)]


def clip_outliers(P, lo, hi):
    low = np.percentile(P, lo, axis=0)
    high = np.percentile(P, hi, axis=0)
    return P[np.all((P >= low) & (P <= high), axis=1)]


def downsample(P, max_points, seed=0):
    if len(P) <= max_points:
        return P
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(P), size=max_points, replace=False)
    return P[idx]


def rotation_matrix_xyz(dx, dy, dz):
    rx, ry, rz = np.deg2rad([dx, dy, dz])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float32)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float32)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float32)

    return (Rz @ Ry @ Rx).astype(np.float32)


def apply_rotation(P, dx, dy, dz, recenter):
    R = rotation_matrix_xyz(dx, dy, dz)
    if recenter:
        center = P.mean(axis=0, keepdims=True)
        return (P - center) @ R.T + center
    return P @ R.T


def compute_bounds(P, pad_fraction):
    mins = P.min(axis=0)
    maxs = P.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-9)
    pad = spans * pad_fraction
    xmin, ymin, zmin = mins - pad
    xmax, ymax, zmax = maxs + pad
    return (float(xmin), float(xmax),
            float(ymin), float(ymax),
            float(zmin), float(zmax))


def main():
    files = sorted(glob.glob(GLOB_PATTERN))
    if not files:
        raise SystemExit("No files matched.")

    import re, os
    files = sorted(files, key=lambda f: int(re.search(r"\d+", os.path.basename(f)).group()))

    all_points = []
    break_p = 1

    for i, f in enumerate(files, start=1):
        if i > break_p:
            P = load_points(f)
            P = clean_points(P)
            all_points.append(P)

    P = np.vstack(all_points)

    if CLIP_OUTLIERS:
        P = clip_outliers(P, CLIP_LO, CLIP_HI)

    P = downsample(P, MAX_POINTS, seed=SEED)

    if ROTATE_ENABLE:
        P = apply_rotation(P,
                           ROTATE_DEGREES_X,
                           ROTATE_DEGREES_Y,
                           ROTATE_DEGREES_Z,
                           RECENTER_AFTER_ROTATION)

    if SHIFT_TO_ZERO_MIN:
        P -= P.min(axis=0, keepdims=True)

    if SCALE_ENABLE:
        P *= SCALE_FACTOR
        print(f"Scaled cloud by {SCALE_FACTOR}x")

    bounds = compute_bounds(P, BOUNDS_PAD_FRACTION)

    cloud = pv.PolyData(P)
    plotter = pv.Plotter(window_size=WINDOW_SIZE)
    plotter.set_background("black")

    plotter.add_points(cloud,
                       render_points_as_spheres=True,
                       point_size=POINT_SIZE,
                       color="red",
                       opacity=0.95)

    plotter.add_text(TITLE, position="upper_left", color="white", font_size=14)

    plotter.show_bounds(
        bounds=bounds,
        grid="back",
        location="outer",
        ticks="outside",
        color="green",
        xtitle="X",
        ytitle="Y",
        ztitle="Z",
        n_xlabels=N_XLABELS,
        n_ylabels=N_YLABELS,
        n_zlabels=N_ZLABELS,
        all_edges=True,
    )

    # ---------------- CAMERA CONTROL ----------------
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    max_extent = max(xmax, ymax, zmax)

    # Place camera diagonally out from origin
    camera_position = (
        (max_extent * 2.5, max_extent * 2.5, max_extent * 2.5),  # camera location
        (0.0, 0.0, 0.0),                                         # focal point (origin)
        (0.0, 0.0, 1.0),                                         # up vector
    )

    plotter.camera_position = camera_position
    plotter.camera_set = True
    # ------------------------------------------------

    plotter.show()


if __name__ == "__main__":
    main()