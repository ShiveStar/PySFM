import glob
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIG =================
GLOB_PATTERN = "results/log_cabin_0.3_8000_really_good/*.txt"
DELIMITER = ";"

MAX_POINTS = 100_000
CLIP_OUTLIERS = True
CLIP_LO = 5.0
CLIP_HI = 99.0

SEED = 0

# Base rotation applied once before interactive rotation
ROTATE_ENABLE = True
ROTATE_DEGREES_X = 90.0
ROTATE_DEGREES_Y = 92.0
ROTATE_DEGREES_Z = 0.0
RECENTER_AFTER_BASE_ROTATION = False  # recenter around centroid before base rotation

# Interactive rotation (arrow keys)
# Left/Right  -> rotate about Y (yaw)
# Up/Down     -> rotate about X (pitch)
YAW_INIT_DEG = 0.0
PITCH_INIT_DEG = 0.0
YAW_STEP_DEG = 1.0
PITCH_STEP_DEG = 1.0
FAST_MULT = 10.0  # hold SHIFT for faster steps (shift+arrow)

# Centering (for “perfectly centered” view)
# - 'mean' centers by centroid
# - 'bbox' centers by bounding-box midpoint
CENTER_ENABLE = True
CENTER_MODE = "bbox"  # "mean" | "bbox"

# Optional scaling
SCALE_ENABLE = False
SCALE_FACTOR = 100.0

# Projection keys:
#   Z => XY (normal = +Z)
#   Y => XZ (normal = +Y)
#   X => YZ (normal = +X)
PROJECTION_INIT = "xz"  # "xy" | "xz" | "yz"

# Plot controls
FIGSIZE = (12, 8)
POINT_SIZE = 0.15
ALPHA = 1.0
DARKMODE = True
TITLE = "PySFM Point Cloud (2D projection + arrows)"
GRID_ALPHA = 0.2

# View padding (adds whitespace around the centered model)
PAD_FRACTION = 0.05  # 5%
# =========================================


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

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]], dtype=np.float32)

    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]], dtype=np.float32)

    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [0,   0,  1]], dtype=np.float32)

    # apply as P @ R.T
    return (Rz @ Ry @ Rx).astype(np.float32)


def apply_rotation(P, dx, dy, dz, recenter=False):
    R = rotation_matrix_xyz(dx, dy, dz)
    if recenter:
        c = P.mean(axis=0, keepdims=True)
        return (P - c) @ R.T + c
    return P @ R.T


def center_to_origin(P, mode="bbox"):
    mode = mode.lower().strip()
    if mode == "mean":
        c = P.mean(axis=0, keepdims=True)
        return P - c
    if mode == "bbox":
        mins = P.min(axis=0, keepdims=True)
        maxs = P.max(axis=0, keepdims=True)
        mid = (mins + maxs) / 2.0
        return P - mid
    raise ValueError("CENTER_MODE must be 'mean' or 'bbox'")


def project_2d(P, projection: str):
    projection = projection.lower().strip()
    if projection == "xy":
        return P[:, 0], P[:, 1], "X", "Y"
    if projection == "xz":
        return P[:, 0], P[:, 2], "X", "Z"
    if projection == "yz":
        return P[:, 1], P[:, 2], "Y", "Z"
    raise ValueError("projection must be 'xy', 'xz', or 'yz'")


def set_darkmode():
    if not DARKMODE:
        return
    plt.rcParams.update({
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
    })


def load_pipeline():
    files = sorted(glob.glob(GLOB_PATTERN))
    if not files:
        raise SystemExit(f"No files matched: {GLOB_PATTERN}")

    import re, os

    def extract_number(f):
        m = re.search(r"\d+", os.path.basename(f))
        return int(m.group()) if m else -1

    files = sorted(files, key=extract_number)

    all_points = []
    break_p = 1

    for i, f in enumerate(files, start=1):
        if i > break_p:
            P = load_points(f)
            P = clean_points(P)
            all_points.append(P)

    if not all_points:
        raise SystemExit("No point files loaded (check break_p / input files).")

    P = np.vstack(all_points)

    if CLIP_OUTLIERS:
        P = clip_outliers(P, CLIP_LO, CLIP_HI)

    P = downsample(P, MAX_POINTS, seed=SEED)

    if ROTATE_ENABLE:
        P = apply_rotation(
            P,
            ROTATE_DEGREES_X,
            ROTATE_DEGREES_Y,
            ROTATE_DEGREES_Z,
            recenter=RECENTER_AFTER_BASE_ROTATION,
        )

    if SCALE_ENABLE:
        P *= SCALE_FACTOR

    return P


def main():
    set_darkmode()
    P_base = load_pipeline()

    state = {
        "projection": PROJECTION_INIT.lower(),
        "yaw_deg": float(YAW_INIT_DEG),
        "pitch_deg": float(PITCH_INIT_DEG),
    }

    fig, ax = plt.subplots(figsize=FIGSIZE)

    def redraw():
        ax.clear()

        # Apply interactive rotations (pitch about X, yaw about Y)
        P = apply_rotation(P_base, state["pitch_deg"], state["yaw_deg"], 0.0, recenter=False)

        # Center “perfectly” (in 3D space), then project to 2D
        if CENTER_ENABLE:
            P = center_to_origin(P, CENTER_MODE)

        x2, y2, xlab, ylab = project_2d(P, state["projection"])

        # Also center in 2D (keeps the plot perfectly centered even if projection biases)
        x2 = x2 - np.mean(x2)
        y2 = y2 - np.mean(y2)

        ax.scatter(x2, y2, s=POINT_SIZE, alpha=ALPHA)

        ax.set_title(
            f"{TITLE} | proj={state['projection'].upper()} | "
            f"yaw(Y)={state['yaw_deg']:.1f}° | pitch(X)={state['pitch_deg']:.1f}°"
        )
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=GRID_ALPHA)

        # Tight, symmetric limits around 0 for “perfect centering”
        x_abs = np.max(np.abs(x2)) if x2.size else 1.0
        y_abs = np.max(np.abs(y2)) if y2.size else 1.0
        r = max(x_abs, y_abs)
        r = r * (1.0 + PAD_FRACTION)

        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)

        fig.tight_layout()
        fig.canvas.draw_idle()

    def is_shifted_key(key_str: str) -> bool:
        # Backends vary: sometimes "shift+left", sometimes just "left" (and shift is not included)
        return "shift" in (key_str or "").lower()

    def on_key(event):
        k_raw = event.key or ""
        k = k_raw.lower()

        # quit
        if k in ("escape", "q"):
            plt.close(fig)
            return

        # projection selection (persist rotation when switching planes)
        if k == "z":
            state["projection"] = "xy"
            redraw()
            return
        if k == "y":
            state["projection"] = "xz"
            redraw()
            return
        if k == "x":
            state["projection"] = "yz"
            redraw()
            return

        # reset
        if k == "r":
            state["yaw_deg"] = float(YAW_INIT_DEG)
            state["pitch_deg"] = float(PITCH_INIT_DEG)
            state["projection"] = PROJECTION_INIT.lower()
            redraw()
            return

        fast = FAST_MULT if is_shifted_key(k_raw) else 1.0

        # arrows:
        # left/right -> yaw about Y
        # up/down    -> pitch about X
        if "left" == k:
            state["yaw_deg"] -= YAW_STEP_DEG * fast
            redraw()
            return
        if "right" == k:
            state["yaw_deg"] += YAW_STEP_DEG * fast
            redraw()
            return
        if "up" == k:
            state["pitch_deg"] += PITCH_STEP_DEG * fast
            redraw()
            return
        if "down" == k:
            state["pitch_deg"] -= PITCH_STEP_DEG * fast
            redraw()
            return

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()


if __name__ == "__main__":
    main()