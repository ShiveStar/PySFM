# utils.py
# Compatible with modern OpenCV (4.x) + SciPy (modern Rotation API)
# - Uses cv.SIFT_create() when available (no contrib required for SIFT on modern OpenCV)
# - Only touches SURF if you explicitly request detector='SURF'
# - Fixes Rotation.from_dcm / as_dcm deprecations (uses from_matrix / as_matrix)

import numpy as np
import glob, os, sys, time
import cv2 as cv
from scipy.spatial.transform import Rotation as R


class SFM_Params:
    def __init__(
        self,
        img_dir,
        n_images=None,
        maxFeatures=None,
        hessianThreshold=200,
        detector="SIFT",
        matcher="FLANN",
    ):
        self.img_dir = img_dir
        files = sorted(glob.glob(self.img_dir))
        self.files = files[:n_images] if (n_images is not None) else files

        self.n_cameras = len(self.files)
        self.n_good_matches = np.zeros((self.n_cameras, self.n_cameras))

        detector = (detector or "").upper()
        matcher = (matcher or "").upper()

        # -------------------------
        # Feature detectors
        # -------------------------
        self.detector_name = detector
        self.descriptor_is_binary = False

        def _make_sift(nfeat):
            # Modern OpenCV
            if hasattr(cv, "SIFT_create"):
                return cv.SIFT_create(nfeatures=nfeat) if nfeat is not None else cv.SIFT_create()
            # Legacy contrib
            if hasattr(cv, "xfeatures2d") and hasattr(cv.xfeatures2d, "SIFT_create"):
                return cv.xfeatures2d.SIFT_create(nfeatures=nfeat) if nfeat is not None else cv.xfeatures2d.SIFT_create()
            return None

        def _make_surf(hess):
            # SURF is contrib-only (and may not be present even in contrib builds)
            if hasattr(cv, "xfeatures2d") and hasattr(cv.xfeatures2d, "SURF_create"):
                return cv.xfeatures2d.SURF_create(hessianThreshold=hess, extended=True)
            return None

        if detector == "SIFT":
            sift = _make_sift(int(maxFeatures) if maxFeatures is not None else None)
            if sift is None:
                raise RuntimeError(
                    "SIFT requested but not available in your cv2 build. "
                    "Install a modern OpenCV wheel (opencv-python) that includes SIFT, "
                    "or install opencv-contrib-python and use its SIFT if needed."
                )
            self.detector = sift

        elif detector == "SURF":
            surf = _make_surf(hessianThreshold)
            if surf is None:
                raise RuntimeError(
                    "SURF requested but cv2.xfeatures2d.SURF_create is unavailable. "
                    "Install opencv-contrib-python (note: some builds omit SURF), "
                    "or switch detector='SIFT'."
                )
            self.detector = surf

        elif detector == "ORB":
            # Optional: useful fallback; binary descriptors
            nfeat = int(maxFeatures) if maxFeatures is not None else 4000
            self.detector = cv.ORB_create(nfeatures=nfeat)
            self.descriptor_is_binary = True

        else:
            raise ValueError(f"Unknown detector='{detector}'. Use 'SIFT', 'SURF', or 'ORB'.")

        # -------------------------
        # Feature matcher
        # -------------------------
        if matcher == "FLANN":
            if self.descriptor_is_binary:
                # FLANN-LSH for ORB (binary descriptors)
                FLANN_INDEX_LSH = 6
                index_params = dict(
                    algorithm=FLANN_INDEX_LSH,
                    table_number=12,
                    key_size=20,
                    multi_probe_level=2,
                )
                search_params = dict(checks=50)
                self.matcher = cv.FlannBasedMatcher(index_params, search_params)
            else:
                # FLANN-KDTREE for SIFT/SURF (float descriptors)
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                self.matcher = cv.FlannBasedMatcher(index_params, search_params)

        elif matcher == "BF":
            if self.descriptor_is_binary:
                self.matcher = cv.BFMatcher(cv.NORM_HAMMING)
            else:
                self.matcher = cv.BFMatcher(cv.NORM_L2)

        else:
            raise ValueError(f"Unknown matcher='{matcher}'. Use 'FLANN' or 'BF'.")


def read_image(path, K, distcoeffs=None, undistort=False, ratio=1):
    """
    load images as gray images
    Input:
        path:       image path
        K:          calibration matrix
        distcoeffs: distortion coefficient
    Output:
        img:        loaded 2D array
    """
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if undistort:
        if distcoeffs is None:
            raise ValueError("undistort=True but distcoeffs is None")
        img = cv.undistort(img, K, distcoeffs)
    return img


def load_images(config, dict_cameras, downscale=1.0):
    """
    Read all images and optionally downscale.

    Parameters
    ----------
    downscale : float
        1.0  -> original size
        0.5  -> half resolution (recommended for speed)
        0.25 -> quarter resolution (very fast, lower accuracy)
    """

    if downscale <= 0 or downscale > 1:
        raise ValueError("downscale must be in range (0, 1].")

    # Scale intrinsic matrix if needed
    if downscale != 1.0:
        config.K[0, 0] *= downscale  # fx
        config.K[1, 1] *= downscale  # fy
        config.K[0, 2] *= downscale  # cx
        config.K[1, 2] *= downscale  # cy

    for i in range(config.n_cameras):
        img_path = config.files[i]
        img = read_image(img_path, config.K, distcoeffs=None, undistort=False)

        if downscale != 1.0:
            img = cv.resize(
                img,
                None,
                fx=downscale,
                fy=downscale,
                interpolation=cv.INTER_AREA
            )

        dict_cameras[i]["img"] = img

    print(f"Loaded in total {config.n_cameras} frames (scale={downscale})")
    return dict_cameras


def extract_features(config, dict_cameras):
    """
    extract features
    """
    t0 = time.time()
    for i in range(config.n_cameras):
        img = dict_cameras[i]["img"]
        kp, des = config.detector.detectAndCompute(img, None)
        dict_cameras[i]["kp"], dict_cameras[i]["des"] = kp, des
    t1 = time.time()
    print("Feature detection takes %d seconds" % (t1 - t0))
    return dict_cameras


def match_features(config, dict_cameras):
    """
    exhaustively match features
    """
    t0 = time.time()
    ratio_thr = getattr(config, "ratio_test_threshold", 0.75)

    for i in range(config.n_cameras):
        dict_cameras[i]["matches"] = []
        des1 = dict_cameras[i]["des"]

        for j in range(config.n_cameras):
            if i == j:
                dict_cameras[i]["matches"].append([])
                continue

            des2 = dict_cameras[j]["des"]
            if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
                config.n_good_matches[i, j] = 0
                dict_cameras[i]["matches"].append([])
                continue

            # FLANN-KDTREE needs float32 descriptors (SIFT/SURF)
            if not getattr(config, "descriptor_is_binary", False):
                if des1.dtype != np.float32:
                    des1m = des1.astype(np.float32, copy=False)
                else:
                    des1m = des1
                if des2.dtype != np.float32:
                    des2m = des2.astype(np.float32, copy=False)
                else:
                    des2m = des2
            else:
                des1m, des2m = des1, des2

            matches = config.matcher.knnMatch(des1m, des2m, k=2)
            matches = sorted(matches, key=lambda x: x[0].distance)

            good_match = []
            ind_train = []  # ensure unique trainIdx
            for m_n in matches:
                if len(m_n) < 2:
                    continue
                m, n = m_n
                if m.distance < ratio_thr * n.distance and m.trainIdx not in ind_train:
                    good_match.append(m)
                    ind_train.append(m.trainIdx)

            config.n_good_matches[i, j] = len(good_match)
            dict_cameras[i]["matches"].append(good_match)

    t1 = time.time()
    print("Feature matching takes %d seconds" % (t1 - t0))
    return dict_cameras


def reprojection_error(points_3d, points_2d, P, K):
    """
    calculate the reprojection error
    Input:
        points_3d:  4 x N
        points_2d:  N X 3
        P:          4 X 4
        K:          3 X 3
    Output:
        error:      N x 2
    """
    reprojected_2d = np.dot(P, points_3d)[:3, :].T
    reprojected_2d = reprojected_2d[:, :2] / reprojected_2d[:, 2, np.newaxis]
    error = reprojected_2d - points_2d[:, :2]
    error = np.dot(K[:2, :2], error.T).T
    return error


def linearTriangulation(P1, x1s, P2, x2s, K):
    """
    Given two projection matrices and calibrated image points, triangulate them to get 3-D points
    and also get the reprojection error [pixel]
    Input:
        P1:     4 x 4
        P2:     4 x 4
        x1s:    N x 3
        x2s:    N x 3
        K:      3 x 3
    Output:
        XS:     4 x N
        error:  N x 4
    """
    XS = np.zeros((4, x1s.shape[0]))
    for k in range(x1s.shape[0]):
        r1 = x1s[k, 0] * P1[2, :] - P1[0, :]
        r2 = x1s[k, 1] * P1[2, :] - P1[1, :]
        r3 = x2s[k, 0] * P2[2, :] - P2[0, :]
        r4 = x2s[k, 1] * P2[2, :] - P2[1, :]

        A = np.vstack((r1, r2, r3, r4))
        _, _, Vh = np.linalg.svd(A)
        XS[:, k] = Vh.T[:, -1] / Vh.T[3, 3]

    error_1 = reprojection_error(XS, x1s, P1, K)
    error_2 = reprojection_error(XS, x2s, P2, K)
    error = np.hstack((error_1, error_2))
    return XS, error


def decomposeE(E, x1s, x2s, K):
    """
    Given the essential and calibrated image points, get the second projection matrix and the index of inliers
    Input:
        E:      3 x 3
        x1s:    N x 3
        x2s:    N x 3
        K:      3 x 3

    Output:
        proj_mat:   4 x 4
        ind_inlier: N x 1 (boolean mask)
    """
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, S, Vh = np.linalg.svd(E)

    t = U[:, 2].reshape((-1, 1))

    R1 = np.dot(np.dot(U, W), Vh)
    R2 = np.dot(np.dot(U, W.T), Vh)
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    P1 = np.vstack((np.hstack((R1, t)), np.array([0, 0, 0, 1])))
    P2 = np.vstack((np.hstack((R1, -t)), np.array([0, 0, 0, 1])))
    P3 = np.vstack((np.hstack((R2, t)), np.array([0, 0, 0, 1])))
    P4 = np.vstack((np.hstack((R2, -t)), np.array([0, 0, 0, 1])))
    Ps = [P1, P2, P3, P4]

    n_inliers = []
    indice_inliers = []
    P = np.eye(4)

    for proj_mat in Ps:
        X, _ = linearTriangulation(P, x1s, proj_mat, x2s, K)
        p1X = np.dot(P, X)
        p2X = np.dot(proj_mat, X)

        indice_inlier = np.logical_and(p1X[2, :] > 0, p2X[2, :] > 0)
        indice_inliers.append(indice_inlier)
        n_inliers.append(indice_inlier.sum())

    n_inliers = np.array(n_inliers)
    index_proj_mat = n_inliers.argmax()

    ind_inlier = indice_inliers[index_proj_mat]
    proj_mat = Ps[index_proj_mat]
    return proj_mat, ind_inlier


def proj_mat_to_camera_vec(proj_mat):
    """
    decompose the projection matrix to camera params (rotation vector and translation vector)
    Input:
        proj_mat:       4 x 4
    Output:
        camera_vec:     1 x 6
    """
    rot_mat = proj_mat[:3, :3]
    if hasattr(R, "from_matrix"):
        r = R.from_matrix(rot_mat)
    else:
        r = R.from_dcm(rot_mat)  # legacy SciPy
    rot_vec = r.as_rotvec()
    t_vec = proj_mat[:3, 3]
    camera_vec = np.hstack((rot_vec, t_vec))
    return camera_vec


def recover_projection_matrix(camera_param):
    """
    given camera parameters, recover the projection matrix
    Input:
        camera_param:   1 x 6
    Output:
        P:              4 x 4
    """
    rot_vec = camera_param[:3]
    translate_vec = camera_param[3:]
    r = R.from_rotvec(rot_vec)
    rot_matrix = r.as_matrix() if hasattr(r, "as_matrix") else r.as_dcm()  # legacy SciPy
    P = np.eye(4)
    P[:3, :3] = rot_matrix
    P[:3, 3] = translate_vec.T
    return P
