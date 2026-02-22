import os
import re
import numpy as np

from utils import *
from init_camera import initialize
from register_camera import register

# ============================================================
# CONFIG (edit everything here)
# ============================================================

# Dataset / naming
KEY = "log_cabin"
SAVE_NAME = KEY

# Paths
K_PATH = "data/k/K.txt"
IMG_GLOB = f"data/{KEY}/*.jpeg"
RESULTS_DIR = f"results/{SAVE_NAME}"

# Camera / distortion
K1 = 0.0
K2 = 0.0
DIST_COEFFS = np.array([K1, K2, 0.0, 0.0], dtype=np.float32)

# Image pipeline
DOWNSCALE = 1  # e.g. 0.5 halves resolution, 0.2 is aggressive downscale

# Feature extraction / matching (SFM_Params)
HESSIAN_THRESHOLD = 8000
DETECTOR = "SIFT"
MATCHER = "FLANN"

# Matching / geometry thresholds
RATIO_TEST_THRESHOLD = 0.60
PNP_THRESHOLD = 5
DIST_THRESHOLD = 100
POST_THRESHOLD = 5
PRE_THRESHOLD = 24

# Output cadence
SAVE_INITIAL_POINTS_FILENAME = "2.txt"   # after initialize()
SAVE_EACH_REGISTER_STEP = True           # save after each register() call

# Optional: stop early (None = run full)
MAX_REGISTER_STEPS = None  # int or None

# ============================================================


def extract_number(path: str) -> int:
    """
    Extract first integer found in filename.
    Example:
        '12.jpeg' -> 12
        'frame_003.jpeg' -> 3
    """
    name = os.path.basename(path)
    m = re.search(r"\d+", name)
    return int(m.group()) if m else -1


def load_intrinsics_k_txt(k_path: str) -> np.ndarray:
    """
    Load a 3x3 intrinsic matrix from a simple whitespace-separated text file.
    Expects at least 3 lines, each with >= 3 floats.
    """
    with open(k_path, "r") as f:
        lines = f.readlines()

    K = np.zeros((3, 3), dtype=float)
    for i in range(3):
        parts = lines[i].strip().split()
        if len(parts) < 3:
            raise ValueError(f"{k_path}: line {i+1} has < 3 columns")
        for j in range(3):
            K[i, j] = float(parts[j])
    return K


def build_config_and_cameras():
    # ---- intrinsics ----
    K = load_intrinsics_k_txt(K_PATH)

    # ---- sfm config ----
    config = SFM_Params(
        IMG_GLOB,
        hessianThreshold=HESSIAN_THRESHOLD,
        detector=DETECTOR,
        matcher=MATCHER,
    )

    # ---- IMPORTANT: numeric sort of image files ----
    config.files = sorted(config.files, key=extract_number)

    # ---- camera dict ----
    dict_cameras = {i: {} for i in range(config.n_cameras)}

    # ---- attach calibration + thresholds ----
    config.K = K
    config.DISTCOEFFS = DIST_COEFFS

    config.ratio_test_threshold = RATIO_TEST_THRESHOLD
    config.pnp_threshold = PNP_THRESHOLD
    config.dist_threshold = DIST_THRESHOLD
    config.post_threshold = POST_THRESHOLD
    config.pre_threshold = PRE_THRESHOLD

    # tracking list (used downstream)
    config.indice_registered_cameras = []

    return config, dict_cameras


def get_unique_point_mask(config, dict_cameras):
    """
    Build a unique index list of reconstructed points referenced by all registered cameras.
    Returns a sorted list of unique indices.
    """
    mask = []
    for cam_idx in config.indice_registered_cameras:
        # expects dict_cameras[cam_idx]['point_indice'] exists
        mask.extend(dict_cameras[cam_idx]["point_indice"].tolist())
    # unique + stable ordering
    return sorted(set(mask))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    config, dict_cameras = build_config_and_cameras()

    # ---- pipeline ----
    dict_cameras = load_images(config, dict_cameras, downscale=DOWNSCALE)
    dict_cameras = extract_features(config, dict_cameras)
    dict_cameras = match_features(config, dict_cameras)

    # ---- initialize ----
    config, dict_cameras = initialize(config, dict_cameras)

    # save initial
    init_out = os.path.join(RESULTS_DIR, SAVE_INITIAL_POINTS_FILENAME)
    np.savetxt(init_out, config.reconstructed_points_3d, delimiter=";")
    print(f"[saved] {init_out}  ({config.reconstructed_points_3d.shape[0]} points)")

    # ---- register remaining cameras ----
    n_cameras = config.n_cameras
    total_steps = n_cameras - 2
    if MAX_REGISTER_STEPS is not None:
        total_steps = min(total_steps, int(MAX_REGISTER_STEPS))

    for step in range(total_steps):
        cam_number = step + 3  # consistent with your original naming
        print("-" * 95)
        print(f"Registering camera {cam_number}...")

        config, dict_cameras, n_obs, n_pts3d = register(config, dict_cameras)
        print(f"{n_obs} observations, {n_pts3d} 3D points")

        if SAVE_EACH_REGISTER_STEP:
            mask = get_unique_point_mask(config, dict_cameras)
            reconstructed_points_3d = config.reconstructed_points_3d[mask]

            out_path = os.path.join(RESULTS_DIR, f"{cam_number}.txt")
            np.savetxt(out_path, reconstructed_points_3d, delimiter=";")
            print(f"[saved] {out_path}  ({reconstructed_points_3d.shape[0]} points)")


if __name__ == "__main__":
    main()