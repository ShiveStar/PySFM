import numpy as np
from scipy.sparse import lil_matrix
import numpy.matlib

def rotate(points, rot_vecs, eps=1e-12):
    """
    Safe Rodrigues rotation (no divide-by-zero / NaNs when theta≈0).

    points:   (N,3)
    rot_vecs: (N,3)
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]  # (N,1)
    small = theta < eps

    # Safe unit axis
    v = rot_vecs / np.maximum(theta, eps)

    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotated = (
        cos_theta * points
        + sin_theta * np.cross(v, points)
        + (1.0 - cos_theta) * dot * v
    )

    # For tiny angles, use first-order approximation: R(p) ≈ p + w×p
    if np.any(small):
        idx = small[:, 0]
        rotated[idx] = points[idx] + np.cross(rot_vecs[idx], points[idx])

    return rotated
def project(points, camera_params):
    '''
    Convert 3-D points to 2-D by projecting onto images.
    '''
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj=points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    return points_proj

# get residuals
def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    '''
    get residuals
    '''
    # recover cameras params and points 3d
    camera_params=params[:n_cameras*6].reshape((n_cameras,6))
    points_3d=params[n_cameras*6:].reshape((n_points,3))
    
    # reproject 3d points
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    residual=points_proj-points_2d
    residual=np.dot(K[:2,:2],residual.T).T
    return residual.ravel()



def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    '''
    build sparse jacobian
    '''
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
    return A